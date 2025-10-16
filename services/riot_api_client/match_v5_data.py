
import asyncio
import random
from pathlib import Path
from .base import RiotAPI
from config import settings
from utils import storages
from dataclasses import dataclass
from typing import AsyncGenerator, List, Any, Dict
from config.constants import ENDPOINTS, REGION_TO_CONTINENT, Continents
from itertools import islice
from enum import StrEnum


class StreamKind(StrEnum):
    MATCH = "match"
    TIMELINE = "timeline"

@dataclass
class Message:
    kind: StreamKind
    payload: Dict[str, Any]

# --- helper functions for the class ---

def chunked(iterator: List[any], n: int):
    it = iter(iterator)
    while (batch:= list(islice(it, n))):
         yield batch

def bucketed_matchids(matchids: List[str]):
    continent_buckets = {k: [] for k in Continents}
    for matchid in matchids:
        region_code = matchid.split("_", 1)[0].lower()
        continent  = REGION_TO_CONTINENT[region_code]
        continent_buckets[continent].append(matchid)
    return continent_buckets

def get_cycled_matchids(continent_buckets: dict) -> list:
    match_ids = []
    continents = list(Continents)
    start = 0
    n = len(continents)
    while any(continent_buckets[c] for c in continents):
        c = continents[start % n]
        if continent_buckets[c]:
            match_ids.append(continent_buckets[c].pop())
        start += 1

    return match_ids

# --- class initialisation for ---

class MatchV5Data(RiotAPI):
    def __init__(self, *, max_in_flight: int = 16):
        super().__init__(max_in_flight=max_in_flight)

    async def get_match_data(self) -> AsyncGenerator[Dict[str, Any], None]:
        endpoint = ENDPOINTS.match.by_match_id
        
        async for match_ids in self.get_match_ids():
            match_ids = bucketed_matchids(match_ids)
            for batch in chunked(match_ids, self.max_in_flight):
                tasks: List[asyncio.Task] = []
                for match_id in batch:
                    region_code = match_id.split("_", 1)[0].lower()
                    continent  = REGION_TO_CONTINENT[region_code]

                    url = endpoint.format(
                        continent   = continent,
                        matchId     = match_id,
                        api_key     = self._api_key,
                    )

                    tasks.append(
                        asyncio.create_task(
                            self.fetch_json(
                                url         = url,
                                location    = continent,
                                scope       = "continent",
                            )
                        )
                    )

                for fut in asyncio.as_completed(tasks):
                    match_data: Dict[str, Any] = await fut
                    yield Message(kind=StreamKind.MATCH, payload=match_data)

    async def get_timeline_data(self) -> AsyncGenerator[Dict[str, Any], None]:
        endpoint = ENDPOINTS.match.timeline_by_match_id
        
        async for match_ids in self.get_match_ids():
            match_ids = bucketed_matchids(match_ids)
            for batch in chunked(match_ids, self.max_in_flight):
                tasks: List[asyncio.Task] = []
                for match_id in batch:
                    region_code = match_id.split("_", 1)[0].lower()
                    continent  = REGION_TO_CONTINENT[region_code]

                    url = endpoint.format(
                        continent   = continent,
                        matchId     = match_id,
                        api_key     = self._api_key,
                    )

                    tasks.append(
                        asyncio.create_task(
                            self.fetch_json(
                                url         = url,
                                location    = continent,
                                scope       = "continent",
                            )
                        )
                    )

                for fut in asyncio.as_completed(tasks):
                    timeline_data: Dict[str, Any] = await fut
                    yield Message(kind=StreamKind.TIMELINE, payload=timeline_data)


    @staticmethod
    async def get_match_ids() -> AsyncGenerator[List[str], None]:
        in_path: Path = (
            rf'{settings.base_project_path}' \
            'data/database/raw/match' \
            'matchids.csv.zst'
        )
        for chunk in storages["match_v5"]["ids"]["load"](path=in_path, chunk_size=100):
                flat: List[str] = [matchid for matchid_list in chunk for matchid in matchid_list]
                yield flat

