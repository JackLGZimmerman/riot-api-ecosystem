from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.core.config.constants import (
    ENDPOINTS,
    PLAYERS_REGIONS,
    URLTemplate,
)
from app.models import EliteBoundsConfig, LeagueListDTO, MinifiedLeagueEntryDTO
from app.services.riot_api_client.base import FetchOutcome, RiotAPI
from app.services.riot_api_client.utils import (
    MAX_IN_FLIGHT,
    UrlTuple,
    bounded_elite_tiers,
    iter_in_flight,
    spreading_region,
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

    async def fetch_one(job):
        url, region = job
        result = await riot_api.fetch_json_detailed(url=url, location=region)
        if result.outcome is not FetchOutcome.OK:
            raise RuntimeError(
                "EliteLeagueFetchFailed "
                f"region={region.value} outcome={result.outcome.value} "
                f"status={result.status}"
            )
        if not isinstance(result.data, dict):
            raise RuntimeError(
                "EliteLeagueUnexpectedPayload "
                f"region={region.value} type={type(result.data).__name__}"
            )
        return region, result.data

    async for region, resp in iter_in_flight(
        spread_urls,
        fetch_one,
        max_in_flight=MAX_IN_FLIGHT,
    ):
        entries = MinifiedLeagueEntryDTO.from_list(
            LeagueListDTO.model_validate(resp),
            region=region,
        )

        for entry in entries:
            yield entry
