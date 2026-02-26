from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from app.core.config.constants import (
    ENDPOINTS,
    Region,
    URLTemplate,
)
from app.models import EliteBoundsConfig, LeagueListDTO, MinifiedLeagueEntryDTO
from app.services.riot_api_client.base import RiotAPI
from app.services.riot_api_client.utils import (
    UrlTuple,
    bounded_elite_tiers,
    chunked,
    compact_preview,
    fetch_region_payload,
    spreading_region,
)

logger = logging.getLogger(__name__)

MAX_IN_FLIGHT: int = 128


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
                for region in Region
            )

    spread_urls = spreading_region(urls)

    for batch in chunked(spread_urls, MAX_IN_FLIGHT):
        tasks = [
            fetch_region_payload(
                url=url,
                region=region,
                riot_api=riot_api,
                logger=logger,
            )
            for url, region in batch
        ]
        for region, resp in await asyncio.gather(*tasks):
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
