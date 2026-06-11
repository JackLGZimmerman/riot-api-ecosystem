from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from app.core.config.constants import (
    ENDPOINTS,
)
from app.core.config.constants.geography import REGION_TO_CONTINENT, Continent, Region
from app.services.riot_api_client.base import RiotAPI
from app.services.riot_api_client.utils import (
    MAX_IN_FLIGHT,
    iter_in_flight,
    spreading,
)

logger = logging.getLogger(__name__)
type MatchEndpointType = Literal["by_match_id", "timeline_by_match_id"]


class MatchWork(NamedTuple):
    match_id: str
    region: Region
    continent: Continent


@dataclass(frozen=True)
class MatchFetchResult:
    match_id: str
    data: dict[str, Any] | None
    status: int | None


async def stream_match_data(
    matchids: list[str],
    endpoint_type: MatchEndpointType,
    riot_api: RiotAPI,
) -> AsyncIterator[MatchFetchResult]:
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
        work_items.append(MatchWork(match_id, region, continent))

    shuffled = spreading(work_items, lambda w: w.region)

    async def fetch_one(work: MatchWork) -> MatchFetchResult:
        result = await riot_api.fetch_json_detailed(
            url=endpoint.format(
                continent=work.continent,
                matchId=work.match_id,
            ),
            location=work.continent,
        )
        data = result.data if isinstance(result.data, dict) else None
        return MatchFetchResult(
            match_id=work.match_id,
            data=data,
            status=result.status,
        )

    async for result in iter_in_flight(
        shuffled,
        fetch_one,
        max_in_flight=MAX_IN_FLIGHT,
    ):
        yield result
