from __future__ import annotations

import asyncio
import itertools
from typing import Dict, Tuple, Mapping, List, Final
from pydantic import ValidationError

from config.constants import (
    ENDPOINTS,
    REGION_TO_CONTINENT,
    ELITE_TIER_MAPPING,
    TIER_MAPPING,
    DIVISION_MAPPING,
    REGIONS,      # tuple[str, ...]
    QUEUES,       # tuple[str, ...]
    TIERS,        # tuple[str, ...]
    DIVISIONS,    # tuple[str, ...]
    ELITE_TIERS,  # tuple[str, ...]
)
from config import settings
from models.riot.league import LeagueEntryDTO, LeagueListDTO, MinifiedLeagueEntryDTO
from services import RiotAPI


# ---------- helpers -----------------------------------------------------------

def chunked(iterable, n: int):
    """Yield consecutive *n*-sized chunks of *iterable* (last may be smaller)."""
    it = iter(iterable)
    while (batch := list(itertools.islice(it, n))):
        yield batch


EliteQueueBound = Dict[str, str | None]              # queue -> base elite tier | None
SubEliteQueueBound = Dict[str, Tuple[str, str] | None]  # queue -> (tier, division) | None


def _make_minified(
    *,
    puuid: str,
    queue_type: str,
    tier: str,
    rank: str,
    wins: int,
    losses: int,
    region: str,
) -> MinifiedLeagueEntryDTO:
    return MinifiedLeagueEntryDTO(
        puuid,
        queue_type,
        tier,
        rank,
        wins,
        losses,
        region,
        REGION_TO_CONTINENT[region],
    )

    
def _list_to_minified(dto: LeagueListDTO, region: str) -> list[MinifiedLeagueEntryDTO]:
    return [
        _make_minified(
            puuid=e.puuid,
            queue_type=dto.queue,
            tier=dto.tier,
            rank=e.rank,
            wins=e.wins,
            losses=e.losses,
            region=region,
        )
        for e in dto.entries
        if e and e.puuid
    ]


def _entry_to_minified(dto: LeagueEntryDTO, region: str) -> MinifiedLeagueEntryDTO:
    return _make_minified(
        puuid=dto.puuid,
        queue_type=dto.queueType,
        tier=dto.tier,
        rank=dto.rank,
        wins=dto.wins,
        losses=dto.losses,
        region=region,
    )

class LeagueV4(RiotAPI):
    """Streams elite and sub-elite league pages."""

    def __init__(self, *, max_in_flight: int = 32):
        super().__init__()
        self.max_in_flight = max_in_flight
        self.league_page_upper_bound: int = settings.league_page_upper_bound


    async def stream_elite_players(self, queue_bounds: EliteQueueBound):
        urls: list[tuple[str, str]] = []

        for queue, base_tier in queue_bounds.items():
            if base_tier is None:
                continue
            for tier in ELITE_TIER_MAPPING[base_tier]:
                template = getattr(ENDPOINTS.league, tier.lower())
                urls.extend(
                    (
                        template.format(region=region, queue=queue, api_key=self._api_key),
                        region,
                    )
                    for region in REGIONS
                )

        for batch in chunked(urls, self.max_in_flight):
            async with asyncio.TaskGroup() as tg:
                tasks = {
                    tg.create_task(self.fetch_json(url=u, location=r, scope="region")): r
                    for u, r in batch
                }

            for task, region in tasks.items():
                resp = task.result()
                try:
                    dto = LeagueListDTO(**resp)
                except ValidationError:
                    continue
                for entry in _list_to_minified(dto, region):
                    yield entry


    async def stream_sub_elite_players(self, queue_bounds: SubEliteQueueBound):
        """Yield `MinifiedLeagueEntryDTO` objects for all requested sub-elite pages."""
        page_bounds = await self._discover_page_bounds(queue_bounds)
        tmpl = str(ENDPOINTS.league.by_queue_tier_division)

        for (region, queue, tier, div), last_page in page_bounds.items():
            for batch in chunked(range(1, last_page + 1), self.max_in_flight):
                async with asyncio.TaskGroup() as tg:
                    tasks = {
                        tg.create_task(
                            self.fetch_json(
                                url=tmpl.format(
                                    region=region,
                                    queue=queue,
                                    tier=tier,
                                    division=div,
                                    page=pg,
                                    api_key=self._api_key,
                                ),
                                location=region,
                                scope="region",
                            )
                        ): region
                        for pg in batch
                    }

                for task, region in tasks.items():
                    records = task.result()
                    for raw in records:
                        try:
                            dto = LeagueEntryDTO(**raw)
                        except ValidationError:
                            continue
                        yield _entry_to_minified(dto, region)


    async def _discover_page_bounds(self, queue_bounds: SubEliteQueueBound):
        tmpl = str(ENDPOINTS.league.by_queue_tier_division)

        async def probe(region, queue, tier, div):
            low, high = 1, self.league_page_upper_bound + 1
            while low + 1 < high:
                mid = (low + high) // 2
                url = tmpl.format(
                    region=region,
                    queue=queue,
                    tier=tier,
                    division=div,
                    page=mid,
                    api_key=self._api_key,
                )
                data = await self.fetch_json(url=url, location=region, scope="region")
                low, high = (mid, high) if data else (low, mid)
            return (region, queue, tier, div), low

        async with asyncio.TaskGroup() as tg:
            task_map: dict[asyncio.Task, Tuple[str, str, str, str]] = {}
            for region in REGIONS:
                for queue, opt in queue_bounds.items():
                    if opt is None:
                        continue
                    base_tier, base_div = opt
                    for tier in TIER_MAPPING[base_tier]:
                        for div in DIVISION_MAPPING[base_div]:
                            task = tg.create_task(probe(region, queue, tier, div))
                            task_map[task] = (region, queue, tier, div)

        return {task_map[task]: task.result() for task in task_map}


    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()
