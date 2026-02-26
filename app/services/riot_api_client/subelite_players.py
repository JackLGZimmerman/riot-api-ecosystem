from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, TypeAlias

from app.core.config.constants import (
    ENDPOINTS,
    Divisions,
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
    UrlTuple,
    bounded_sub_elite_tiers,
    chunked,
    compact_preview,
    fetch_region_payload,
    spreading,
)

logger = logging.getLogger(__name__)

LEAGUE_PAGE_UPPER_BOUND: int = 1024
MAX_IN_FLIGHT: int = 128
PAGE_NOT_FOUND_STATUS: int = 404

PageKey: TypeAlias = tuple[Region, Queues, Tiers, Divisions]


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
    for region in Region:
        for queue, bounds in queue_bounds.items():
            if not bounds.collect:
                continue
            for tier, division in bounded_sub_elite_tiers(bounds):
                work.append((region, queue, tier, division))

    spread_work = spreading(work, key_fn=lambda x: x[0])

    results: list[tuple[PageKey, int]] = []
    for batch in chunked(spread_work, MAX_IN_FLIGHT):
        probed = await asyncio.gather(
            *(probe(*args) for args in batch),
            return_exceptions=True,
        )
        for args, item in zip(batch, probed, strict=True):
            if isinstance(item, Exception):
                region, queue, tier, div = args
                logger.warning(
                    "LeagueBoundProbeFailed region=%s queue=%s tier=%s division=%s error=%s",
                    region.value,
                    queue.value,
                    tier.value,
                    div.value,
                    type(item).__name__,
                )
                continue
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

    for batch in chunked(spread_jobs, MAX_IN_FLIGHT):
        tasks = [
            fetch_region_payload(
                url=url,
                region=region,
                riot_api=riot_api,
                logger=logger,
            )
            for url, region in batch
        ]
        for region, records in await asyncio.gather(*tasks):
            if not isinstance(records, list) or not records:
                continue

            for raw in records:
                try:
                    dto = LeagueEntryDTO(**raw)
                    entry = MinifiedLeagueEntryDTO.from_entry(dto, region=region)
                except Exception as exc:
                    logger.info(
                        "LeagueUnexpectedFailed region=%s error=%s preview=%r",
                        region.value,
                        type(exc).__name__,
                        compact_preview(raw),
                    )
                    continue
                yield entry
