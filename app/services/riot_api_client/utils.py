import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Hashable, Iterable
from typing import Any, NamedTuple

from app.core.config.constants import (
    JSON,
    REGION_TO_CONTINENT,
    Continent,
    Divisions,
    EliteTiers,
    JSONList,
    Queues,
    Region,
    Tiers,
)
from app.models.riot.league import (
    BasicBoundConfig,
    EliteBoundConfig,
)
from app.services.riot_api_client.base import RiotAPI

_ELITE_ORDER = list(EliteTiers)
_BASIC_RANKS = [(tier, div) for tier in Tiers for div in Divisions]

type JSONLike = JSON | JSONList | None
type UrlRegion = tuple[str, Region]
type UrlTuple = list[UrlRegion]


class PlayerCrawlState(NamedTuple):
    puuid: str
    queue_type: Queues
    continent: Continent
    next_page_start: int
    base_url: str

MAX_LOG_PREVIEW = 300

# Shared concurrency ceiling for fan-out fetch loops (iter_in_flight / worker pools).
MAX_IN_FLIGHT = 64


def compact_preview(payload: Any, *, max_len: int = MAX_LOG_PREVIEW) -> str:
    preview = repr(payload).replace("\n", " ")
    if len(preview) <= max_len:
        return preview
    return preview[: max_len - 3] + "..."


async def fetch_region_payload(
    *,
    url: str,
    region: Region,
    riot_api: RiotAPI,
    logger: logging.Logger,
    error_event: str = "LeagueFetchFailed",
) -> tuple[Region, JSONLike]:
    try:
        return region, await riot_api.fetch_json(
            url=url,
            location=region,
        )
    except Exception as exc:
        logger.warning(
            "%s region=%s error=%s",
            error_event,
            region.value,
            type(exc).__name__,
        )
        return region, None


def make_region_fetcher(
    riot_api: RiotAPI,
    logger: logging.Logger,
) -> Callable[[UrlRegion], Awaitable[tuple[Region, JSONLike]]]:
    """Build the (url, region) -> (region, payload) fetch closure shared by the
    elite / sub-elite league streamers."""

    async def fetch_one(job: UrlRegion) -> tuple[Region, JSONLike]:
        url, region = job
        return await fetch_region_payload(
            url=url,
            region=region,
            riot_api=riot_api,
            logger=logger,
        )

    return fetch_one


def validate_or_log[T](
    raw: Any,
    *,
    region: Region,
    convert: Callable[[Any], T],
    logger: logging.Logger,
) -> T | None:
    """Run ``convert(raw)``; on any failure log the shared LeagueUnexpectedFailed
    line and return None so the caller can skip the record."""
    try:
        return convert(raw)
    except Exception as exc:
        logger.info(
            "LeagueUnexpectedFailed region=%s error=%s preview=%r",
            region.value,
            type(exc).__name__,
            compact_preview(raw),
        )
        return None


async def iter_in_flight[T, R](
    items: Iterable[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    max_in_flight: int,
) -> AsyncIterator[R]:
    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be > 0")

    iterator = iter(items)
    pending: set[asyncio.Future[R]] = set()

    def submit_next() -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False

        pending.add(asyncio.ensure_future(worker(item)))
        return True

    for _ in range(max_in_flight):
        if not submit_next():
            break

    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            done_tasks = list(done)
            for _ in done_tasks:
                submit_next()
            for task in done_tasks:
                yield await task
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def bounded_elite_tiers(cfg: EliteBoundConfig) -> list[EliteTiers]:
    """Return elite tiers between cfg.upper and cfg.lower (inclusive)."""
    if not cfg.collect:
        return []

    start = 0 if cfg.upper is None else _ELITE_ORDER.index(cfg.upper)
    end = len(_ELITE_ORDER) - 1 if cfg.lower is None else _ELITE_ORDER.index(cfg.lower)

    if start > end:
        raise ValueError("Elite bounds: upper must not be below lower")

    return _ELITE_ORDER[start : end + 1]


def bounded_sub_elite_tiers(cfg: BasicBoundConfig) -> list[tuple[Tiers, Divisions]]:
    """Return (tier, division) pairs between upper_* and lower_* (inclusive)."""
    if not cfg.collect:
        return []

    upper = (
        (cfg.upper_tier, cfg.upper_division)
        if cfg.upper_tier is not None and cfg.upper_division is not None
        else None
    )
    lower = (
        (cfg.lower_tier, cfg.lower_division)
        if cfg.lower_tier is not None and cfg.lower_division is not None
        else None
    )

    start = 0 if upper is None else _BASIC_RANKS.index(upper)
    end = len(_BASIC_RANKS) - 1 if lower is None else _BASIC_RANKS.index(lower)

    if start > end:
        raise ValueError("Basic bounds: upper must not be below lower")

    return _BASIC_RANKS[start : end + 1]


def spreading[P, K: Hashable](items: Iterable[P], key_fn: Callable[[P], K]) -> list[P]:
    buckets: dict[K, deque[P]] = defaultdict(deque)
    for item in items:
        buckets[key_fn(item)].append(item)

    keys = list(buckets.keys())
    out: list[P] = []
    while True:
        made = False
        for k in keys:
            dq = buckets[k]
            if dq:
                out.append(dq.popleft())
                made = True
        if not made:
            break
    return out


def spreading_region(
    items: Iterable[UrlRegion],
) -> list[UrlRegion]:
    """
    spreading() adapter for (url, region) items.
    Selects continent using REGION_TO_CONTINENT[region].
    """
    return spreading(
        items,
        key_fn=lambda ur: REGION_TO_CONTINENT[ur[1]],
    )
