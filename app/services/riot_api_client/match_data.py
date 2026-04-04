from __future__ import annotations

import logging
from typing import Any, AsyncIterator, NamedTuple

from app.core.config.constants import (
    ENDPOINTS,
)
from app.core.config.constants.geography import REGION_TO_CONTINENT, Continent, Region
from app.services.riot_api_client.base import RiotAPI
from app.services.riot_api_client.utils import iter_in_flight, spreading

MAX_IN_FLIGHT = 64
logger = logging.getLogger(__name__)


class MatchWork(NamedTuple):
    match_id: str
    continent: Continent


async def yield_match_data(
    matchids: list[str],
    endpoint_type: str,
    riot_api: RiotAPI,
) -> AsyncIterator[dict[str, Any]]:
    endpoint = ENDPOINTS["match"][endpoint_type]

    work_items: list[MatchWork] = []
    for match_id in matchids:
        shard = match_id.split("_", 1)[0].lower()
        try:
            region = Region(shard)
        except ValueError:
            logger.error(
                "Unknown region shard encountered: shard=%s match_id=%s",
                shard,
                match_id,
            )
            raise ValueError(
                f"Unknown region shard '{shard}' in match_id '{match_id}'"
            )

        continent = REGION_TO_CONTINENT[region]
        work_items.append(MatchWork(match_id, continent))

    shuffled = spreading(work_items, lambda w: w.continent)

    async def fetch_one(work: MatchWork) -> Any:
        return await riot_api.fetch_json(
            url=endpoint.format(
                continent=work.continent,
                matchId=work.match_id,
            ),
            location=work.continent,
        )

    async for data in iter_in_flight(
        shuffled,
        fetch_one,
        max_in_flight=MAX_IN_FLIGHT,
    ):
        if isinstance(data, dict):
            yield data


async def stream_non_timeline_data(
    matchids: list[str],
    *,
    riot_api: RiotAPI,
) -> AsyncIterator[dict[str, Any]]:
    async for data in yield_match_data(
        matchids,
        endpoint_type="by_match_id",
        riot_api=riot_api,
    ):
        yield data


async def stream_timeline_data(
    matchids: list[str],
    *,
    riot_api: RiotAPI,
) -> AsyncIterator[dict[str, Any]]:
    async for data in yield_match_data(
        matchids,
        endpoint_type="timeline_by_match_id",
        riot_api=riot_api,
    ):
        yield data
