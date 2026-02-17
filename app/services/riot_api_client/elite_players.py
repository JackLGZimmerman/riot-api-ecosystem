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
    fetch_json_with_carry_over,
    spreading_region,
)

logger = logging.getLogger(__name__)

MAX_IN_FLIGHT: int = 128
MAX_PREVIEW: int = 200


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

    async def fetch_one(url: str, region: Region) -> tuple[Region, object | None]:
        try:
            _, data = await fetch_json_with_carry_over(
                url=url,
                location=region,
                riot_api=riot_api,
                carry_over=(region,),
            )
            return region, data
        except Exception as e:
            logger.warning(
                "LeagueFetchFailed region=%s error=%s",
                region.value,
                type(e).__name__,
            )
            return region, None

    for batch in chunked(spread_urls, MAX_IN_FLIGHT):
        tasks = [fetch_one(url, region) for url, region in batch]
        for region, resp in await asyncio.gather(*tasks):

            try:
                dto = LeagueListDTO.model_validate(resp)
                entries = MinifiedLeagueEntryDTO.from_list(dto, region=region)
            except Exception as e:
                logger.info(
                    "LeagueUnexpectedFailed region=%s error=%s preview=%r",
                    region.value,
                    type(e).__name__,
                    str(resp),
                )
                continue

            for entry in entries:
                yield entry
