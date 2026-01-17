from __future__ import annotations

import asyncio
from typing import AsyncGenerator, NamedTuple

from app.core.config.constants import (
    ENDPOINTS,
)
from app.core.config.constants.generic import JSON, JSONList
from app.core.config.constants.geography import REGION_TO_CONTINENT, Continent, Region
from app.services.riot_api_client.base import RiotAPI
from app.services.riot_api_client.utils import chunked, spreading
from app.worker.pipelines.matchdata_orchestrator import MatchDataCollectorState

SENTINEL: object = object()
MAX_PAGE_START = 900
MAX_PAGE_COUNT = 100
MAX_IN_FLIGHT = 16


class MatchWork(NamedTuple):
    match_id: str
    continent: Continent


async def yield_match_data(
    state: MatchDataCollectorState,
    endpoint_type: str,
    riot_api: RiotAPI,
):
    endpoint = ENDPOINTS["match"][endpoint_type]

    work_items: list[MatchWork] = []
    for match_id in state.matchids:
        region = Region(match_id.split("_", 1)[0].lower())
        continent = REGION_TO_CONTINENT[region]
        work_items.append(MatchWork(match_id, continent))

    shuffled = spreading(work_items, lambda w: w.continent)

    for batch in chunked(shuffled, MAX_IN_FLIGHT):
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    riot_api.fetch_json(
                        url=endpoint.format(match_id=w.match_id),
                        location=w.continent,
                    )
                )
                for w in batch
            ]

        for t in tasks:
            data = t.result()
            if data is not None:
                yield data


async def stream_non_timeline_data(
    state: MatchDataCollectorState, *, riot_api: RiotAPI
) -> AsyncGenerator[JSON | JSONList, None]:
    """
    Fetch non-timeline match payloads.
    Results are consumed internally (e.g. stored, processed, emitted elsewhere).
    """
    async for data in yield_match_data(
        state,
        endpoint_type="by_match_id",
        riot_api=riot_api,
    ):
        if data is not None:
            yield data


async def stream_timeline_data(
    state: MatchDataCollectorState,
    *,
    riot_api: RiotAPI,
) -> AsyncGenerator[JSON | JSONList, None]:
    """
    Stream timeline payloads as raw dicts.
    """
    async for data in yield_match_data(
        state,
        endpoint_type="timeline_by_match_id",
        riot_api=riot_api,
    ):
        if data is not None:
            yield data


"""
Stream ids with regular order based on continent
Batch the matchids and extract the data (Different batch sizing for timeline vs non-timeline)
Return a full batch list, to be parsed

Parser needs to be comprehensive, probably a builder
"""
