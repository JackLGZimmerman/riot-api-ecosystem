from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.core.config.constants import (
    ENDPOINTS,
    PLAYERS_REGIONS,
    URLTemplate,
)
from app.models import EliteBoundsConfig, LeagueListDTO, MinifiedLeagueEntryDTO
from app.services.riot_api_client.base import RiotAPI
from app.services.riot_api_client.utils import (
    MAX_IN_FLIGHT,
    UrlTuple,
    bounded_elite_tiers,
    iter_in_flight,
    make_region_fetcher,
    spreading_region,
    validate_or_log,
)

logger = logging.getLogger(__name__)


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

    async for region, resp in iter_in_flight(
        spread_urls,
        make_region_fetcher(riot_api, logger),
        max_in_flight=MAX_IN_FLIGHT,
    ):
        if resp is None:
            continue

        entries = validate_or_log(
            resp,
            region=region,
            convert=lambda r: MinifiedLeagueEntryDTO.from_list(
                LeagueListDTO.model_validate(r), region=region
            ),
            logger=logger,
        )
        if entries is None:
            continue

        for entry in entries:
            yield entry
