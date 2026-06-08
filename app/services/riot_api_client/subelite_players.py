from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.core.config.constants import (
    ENDPOINTS,
    Divisions,
    PLAYERS_REGIONS,
    Queues,
    Region,
    Tiers,
    URLTemplate,
)
from app.models import (
    BasicBoundsConfig,
    MinifiedLeagueEntryDTO,
)
from app.models.riot.league import LeagueEntryDTO
from app.services.riot_api_client.base import FetchOutcome, RiotAPI
from app.services.riot_api_client.utils import (
    MAX_IN_FLIGHT,
    UrlTuple,
    bounded_sub_elite_tiers,
    iter_in_flight,
    make_region_fetcher,
    spreading,
    validate_or_log,
)

logger = logging.getLogger(__name__)

LEAGUE_PAGE_UPPER_BOUND: int = 1024
PAGE_NOT_FOUND_STATUS: int = 404

type PageKey = tuple[Region, Queues, Tiers, Divisions]


async def discover_page_bounds(
    queue_bounds: BasicBoundsConfig,
    riot_api: RiotAPI,
) -> list[tuple[PageKey, int]]:
    template: URLTemplate = ENDPOINTS["league"]["by_queue_tier_division"]

    async def probe(
        region: Region,
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
            )

            result = await riot_api.fetch_json_detailed(url=url, location=region)

            if result.outcome is FetchOutcome.OK:
                payload = result.data
                if not isinstance(payload, list):
                    raise RuntimeError(
                        "LeagueBoundProbeUnexpectedPayload "
                        f"region={region.value} queue={queue.value} tier={tier.value} "
                        f"division={div.value} page={mid} status={result.status}"
                    )
                low, high = (mid, high) if payload else (low, mid)
                continue

            if (
                result.outcome is FetchOutcome.HTTP_NON_RETRYABLE
                and result.status == PAGE_NOT_FOUND_STATUS
            ):
                return (region, queue, tier, div), low

            raise RuntimeError(
                "LeagueBoundProbeRequestFailed "
                f"region={region.value} queue={queue.value} tier={tier.value} "
                f"division={div.value} page={mid} outcome={result.outcome.value} "
                f"status={result.status}"
            )

        return (region, queue, tier, div), low

    work: list[tuple[Region, Queues, Tiers, Divisions]] = []
    for region in PLAYERS_REGIONS:
        for queue, bounds in queue_bounds.items():
            if not bounds.collect:
                continue
            for tier, division in bounded_sub_elite_tiers(bounds):
                work.append((region, queue, tier, division))

    spread_work = spreading(work, key_fn=lambda x: x[0])

    results: list[tuple[PageKey, int]] = []

    async def probe_safe(
        args: tuple[Region, Queues, Tiers, Divisions],
    ) -> tuple[
        tuple[Region, Queues, Tiers, Divisions],
        tuple[PageKey, int] | None,
        Exception | None,
    ]:
        try:
            return args, await probe(*args), None
        except Exception as exc:
            return args, None, exc

    async for args, item, exc in iter_in_flight(
        spread_work,
        probe_safe,
        max_in_flight=MAX_IN_FLIGHT,
    ):
        if exc is not None:
            region, queue, tier, div = args
            logger.warning(
                "LeagueBoundProbeFailed region=%s queue=%s tier=%s division=%s error=%s",
                region.value,
                queue.value,
                tier.value,
                div.value,
                type(exc).__name__,
            )
            continue
        if item is not None:
            results.append(item)

    return results


async def stream_sub_elite_players(
    queue_bounds: BasicBoundsConfig,
    riot_api: RiotAPI,
) -> AsyncIterator[MinifiedLeagueEntryDTO]:
    page_bounds: list[tuple[PageKey, int]] = await discover_page_bounds(
        queue_bounds=queue_bounds,
        riot_api=riot_api,
    )

    template: URLTemplate = ENDPOINTS["league"]["by_queue_tier_division"]

    jobs: UrlTuple = []
    for (region, queue, tier, div), last_page in page_bounds:
        for page in range(1, last_page + 1):
            url = template.format(
                region=region,
                queue=queue,
                tier=tier,
                division=div,
                page=page,
            )
            jobs.append((url, region))

    spread_jobs = spreading(jobs, key_fn=lambda ur: ur[1])
    del jobs
    del page_bounds

    async for region, records in iter_in_flight(
        spread_jobs,
        make_region_fetcher(riot_api, logger),
        max_in_flight=MAX_IN_FLIGHT,
    ):
        if not isinstance(records, list) or not records:
            continue

        for raw in records:
            entry = validate_or_log(
                raw,
                region=region,
                convert=lambda r: MinifiedLeagueEntryDTO.from_entry(
                    LeagueEntryDTO(**r), region=region
                ),
                logger=logger,
            )
            if entry is None:
                continue
            yield entry
