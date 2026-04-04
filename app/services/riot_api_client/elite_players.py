from __future__ import annotations

import logging
from typing import AsyncIterator

from app.core.config.constants import (
    ENDPOINTS,
    PLAYERS_REGIONS,
    Region,
    URLTemplate,
)
from app.models import EliteBoundsConfig, LeagueListDTO, MinifiedLeagueEntryDTO
from app.services.riot_api_client.base import RiotAPI
from app.services.riot_api_client.utils import (
    JSONLike,
    UrlRegion,
    UrlTuple,
    bounded_elite_tiers,
    compact_preview,
    fetch_region_payload,
    iter_in_flight,
    spreading_region,
)

logger = logging.getLogger(__name__)

MAX_IN_FLIGHT: int = 64


async def stream_elite_players(
    queue_bounds: EliteBoundsConfig,
    riot_api: RiotAPI,
) -> AsyncIterator[MinifiedLeagueEntryDTO]:
    urls: UrlTuple = []
    template: URLTemplate = ENDPOINTS["league"]["elite"]

    for queue, bounds in queue_bounds.items():
        if not bounds.collect:
            continue

        for tier in bounded_elite_tiers(bounds):
            urls.extend(
                (
                    template.format(
                        elite_tier=tier.lower(),
                        region=region,
                        queue=queue,
                    ),
                    region,
                )
                for region in PLAYERS_REGIONS
            )

    spread_urls = spreading_region(urls)

    async def fetch_one(job: UrlRegion) -> tuple[Region, JSONLike]:
        url, region = job
        return await fetch_region_payload(
            url=url,
            region=region,
            riot_api=riot_api,
            logger=logger,
        )

    async for region, resp in iter_in_flight(
        spread_urls,
        fetch_one,
        max_in_flight=MAX_IN_FLIGHT,
    ):
        if resp is None:
            continue

        try:
            dto = LeagueListDTO.model_validate(resp)
            entries = MinifiedLeagueEntryDTO.from_list(dto, region=region)
        except Exception as exc:
            logger.info(
                "LeagueUnexpectedFailed region=%s error=%s preview=%r",
                region.value,
                type(exc).__name__,
                compact_preview(resp),
            )
            continue

        for entry in entries:
            yield entry
