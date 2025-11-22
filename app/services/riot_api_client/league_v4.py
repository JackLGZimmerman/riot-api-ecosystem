from __future__ import annotations

import asyncio
import itertools
from typing import (
    AsyncIterator,
    Iterable,
    Iterator,
    Tuple,
    TypeAlias,
    TypeVar,
)

from pydantic import SecretStr, ValidationError

from app.core.config.constants import (
    ENDPOINTS,
    JSON,
    Divisions,
    EliteTiers,
    JSONList,
    Queues,
    Regions,
    Tiers,
    URLTemplate,
)
from app.models import (
    BasicBoundConfig,
    BasicBoundsConfig,
    EliteBoundConfig,
    EliteBoundsConfig,
    LeagueEntryDTO,
    LeagueListDTO,
    MinifiedLeagueEntryDTO,
)
from app.services import RiotAPI
from app.services.errors import LeagueListDTOError, MinifiedLeagueEntryError
from app.services.riot_api_client.base import get_riot_api
from app.services.utils import reraise

LEAGUE_PAGE_UPPER_BOUND: int = 1024
MAX_IN_FLIGHT: int = 128
REQUEST_TIMEOUT: int = 10

riot_api: RiotAPI = get_riot_api()
api_key: SecretStr = riot_api.api_key

_ELITE_ORDER = list(EliteTiers)
_BASIC_RANKS = [(tier, div) for tier in Tiers for div in Divisions]

T = TypeVar("T")

UrlRegion: TypeAlias = tuple[str, Regions]
UrlTuple: TypeAlias = list[UrlRegion]
PageKey = Tuple[Regions, Queues, Tiers, Divisions]

# ==================================== Helpers ====================================


def chunked(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
    """Yield consecutive n-sized chunks from `iterable` (last may be smaller)."""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


async def _fetch_with_region_obj(url: str, region: Regions) -> tuple[Regions, JSON]:
    data = await asyncio.wait_for(
        riot_api.fetch_json(url=url, location=region),
        timeout=REQUEST_TIMEOUT,
    )
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict JSON, got {type(data)}")
    return region, data


async def _fetch_with_region_list(
    url: str, region: Regions
) -> tuple[Regions, JSONList]:
    data = await asyncio.wait_for(
        riot_api.fetch_json(url=url, location=region),
        timeout=REQUEST_TIMEOUT,
    )
    if not isinstance(data, list):
        raise TypeError(f"Expected list[dict] JSON, got {type(data)}")
    return region, data


def bounded_elite_tiers(cfg: EliteBoundConfig) -> list[EliteTiers]:
    """Return elite tiers between cfg.upper and cfg.lower (inclusive)."""
    if not cfg.collect:
        return []

    start = 0 if cfg.upper is None else _ELITE_ORDER.index(cfg.upper)
    end = len(_ELITE_ORDER) - 1 if cfg.lower is None else _ELITE_ORDER.index(cfg.lower)

    if start > end:
        raise ValueError("Elite bounds: upper must not be below lower")

    return _ELITE_ORDER[start : end + 1]


def bounded_basic_brackets(cfg: BasicBoundConfig) -> list[tuple[Tiers, Divisions]]:
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


# ==================================================================================


async def stream_elite_players(
    queue_bounds: EliteBoundsConfig,
) -> AsyncIterator[MinifiedLeagueEntryDTO]:
    urls: UrlTuple = []
    template: URLTemplate = ENDPOINTS["league"]["elite"]

    for queue, bounds in queue_bounds.items():
        if not bounds.collect:
            continue

        tiers: list[EliteTiers] = bounded_elite_tiers(bounds)

        for tier in tiers:
            urls.extend(
                (
                    template.format(
                        elite_tier=tier.lower(),
                        region=region,
                        queue=queue,
                        api_key=api_key,
                    ),
                    region,
                )
                for region in Regions
            )

    for batch in chunked(urls, MAX_IN_FLIGHT):
        tasks: list[asyncio.Task[tuple[Regions, JSON]]] = [
            asyncio.create_task(_fetch_with_region_obj(url, region))
            for url, region in batch
        ]

        for future in asyncio.as_completed(tasks):
            region, resp = await future
            with reraise(
                LeagueListDTOError,
                f"Failed to build LeagueListDTO (region={region.value})",
            ):
                dto = LeagueListDTO(**resp)

            with reraise(
                MinifiedLeagueEntryError,
                f"Failed to create entries via MinifiedLeagueEntryDTO.from_list (region={region.value})",
            ):
                for entry in MinifiedLeagueEntryDTO.from_list(dto, region=region):
                    yield entry


async def stream_sub_elite_players(
    queue_bounds: BasicBoundsConfig,
) -> AsyncIterator[MinifiedLeagueEntryDTO]:
    page_bounds: dict[PageKey, int] = await _discover_page_bounds(queue_bounds)
    template: URLTemplate = ENDPOINTS["league"]["by_queue_tier_division"]

    for (region, queue, tier, div), last_page in page_bounds.items():
        pages = range(1, last_page + 1)
        for batch in chunked(pages, MAX_IN_FLIGHT):
            tasks: list[asyncio.Task[tuple[Regions, JSONList]]] = [
                asyncio.create_task(
                    _fetch_with_region_list(
                        template.format(
                            region=region,
                            queue=queue,
                            tier=tier,
                            division=div,
                            page=pg,
                            api_key=api_key,
                        ),
                        region,
                    )
                )
                for pg in batch
            ]

            for future in asyncio.as_completed(tasks):
                region, records = await future
                for raw in records:
                    try:
                        dto = LeagueEntryDTO(**raw)
                    except ValidationError:
                        continue
                    yield MinifiedLeagueEntryDTO.from_entry(dto, region=region)


async def _discover_page_bounds(queue_bounds: BasicBoundsConfig) -> dict[PageKey, int]:
    template: URLTemplate = ENDPOINTS["league"]["by_queue_tier_division"]

    async def probe(
        region: Regions,
        queue: Queues,
        tier: Tiers,
        div: Divisions,
    ) -> tuple[PageKey, int]:
        low, high = 1, LEAGUE_PAGE_UPPER_BOUND + 1
        while low + 1 < high:
            mid = (low + high) // 2
            url = template.format(
                region=region,
                queue=queue,
                tier=tier,
                division=div,
                page=mid,
                api_key=api_key,
            )
            _: JSON | JSONList = await riot_api.fetch_json(url=url, location=region)
            low, high = (mid, high) if _ else (low, mid)

        return (region, queue, tier, div), low

    tasks: list[asyncio.Task[tuple[PageKey, int]]] = []
    for region in Regions:
        for queue, bounds in queue_bounds.items():
            if not bounds.collect:
                continue

            brackets: list[tuple[Tiers, Divisions]] = bounded_basic_brackets(bounds)

            for tier, division in brackets:
                tasks.append(asyncio.create_task(probe(region, queue, tier, division)))

    bounds_map: dict[PageKey, int] = {}
    for finished in asyncio.as_completed(tasks):
        page_key, page_num = await finished
        bounds_map[page_key] = page_num

    return bounds_map
