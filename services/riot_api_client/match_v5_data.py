import asyncio
import itertools
from pathlib import Path
from .base import RiotAPI
from services.riot_api_client.factories.base_factory import get_riot_api
from pydantic import SecretStr
from config import settings
from utils import storages
from dataclasses import dataclass
from typing import AsyncGenerator, List, Any, Dict, Iterable, Iterator, Tuple, TypeVar
from config.constants import (
    ENDPOINTS,
    REGION_TO_CONTINENT,
    Continents,
    Regions,
    JSON,
    JSONList,
)
from enum import StrEnum

riot_api: RiotAPI = get_riot_api()
api_key: SecretStr = riot_api.api_key

T = TypeVar("T")


class StreamKind(StrEnum):
    MATCH = "match"
    TIMELINE = "timeline"


@dataclass
class Message:
    kind: StreamKind
    payload: Dict[str, Any]


# --- helper functions for the class ---


def chunked(iterable: Iterable[Any], n: int) -> Iterator[list[Any]]:
    """Yield consecutive n-sized chunks from `iterable` (last may be smaller)."""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


def bucketed_matchids(matchids: list[str]) -> dict[Continents, list[str]]:
    """Group match IDs by continent using the matchId prefix (e.g., 'EUW1_...')."""
    buckets: dict[Continents, list[str]] = {c: [] for c in Continents}
    for matchid in matchids:
        region_code: Regions = Regions(matchid.split("_", 1)[0].lower())
        continent = REGION_TO_CONTINENT[region_code]
        buckets[continent].append(matchid)
    return buckets


def get_cycled_matchids(buckets: dict[Continents, list[str]]) -> list[str]:
    """Interleave match IDs across continents to avoid per-continent bursts."""
    out: list[str] = []
    continents = list(Continents)
    i = 0
    n = len(continents)
    while any(buckets[c] for c in continents):
        c = continents[i % n]
        if buckets[c]:
            out.append(buckets[c].pop())
        i += 1
    return out


# --- class initialisation for ---


async def _fetch_with_continent_obj(
    url: str, continent: Continents
) -> Tuple[Continents, JSON | JSONList]:
    data: JSON | JSONList = await riot_api.fetch_json(url=url, location=continent)
    return continent, data


# ---------------- Data streams ----------------


async def get_match_ids() -> AsyncGenerator[List[str], None]:
    """
    Yields lists of match IDs from your storage (already chunked by the loader).
    """
    in_path: Path = Path(
        f"{settings.base_project_path}data/database/raw/matchmatchids.csv.zst"
    )
    for chunk in storages["match_v5"]["ids"]["load"](path=in_path, chunk_size=100):
        flat: List[str] = [mid for sub in chunk for mid in sub]
        yield flat


async def get_match_data(*, max_in_flight: int = 16) -> AsyncGenerator[Message, None]:
    """
    Stream match payloads as Message(kind=MATCH, payload=dict).
    """
    endpoint = ENDPOINTS.match.by_match_id

    async for batch_ids in get_match_ids():
        buckets = bucketed_matchids(batch_ids)
        ordered = round_robin_ids(buckets)

        for batch in chunked(ordered, max_in_flight):
            tasks: List[asyncio.Task[tuple[Continents, JSONObj]]] = []
            for match_id in batch:
                region_code = match_id.split("_", 1)[0].lower()
                continent = REGION_TO_CONTINENT[region_code]
                url = endpoint.format(
                    continent=continent.value,  # ensure strings in URL
                    matchId=match_id,
                    api_key=riot_api._api_key,  # follows your existing pattern
                )
                tasks.append(
                    asyncio.create_task(_fetch_with_continent_obj(url, continent))
                )

            for fut in asyncio.as_completed(tasks):
                _, match_data = await fut
                yield Message(kind=StreamKind.MATCH, payload=match_data)


async def get_timeline_data(
    *, max_in_flight: int = 16
) -> AsyncGenerator[Message, None]:
    """
    Stream timeline payloads as Message(kind=TIMELINE, payload=dict).
    """
    endpoint = ENDPOINTS.match.timeline_by_match_id

    async for batch_ids in get_match_ids():
        buckets = bucketed_matchids(batch_ids)
        ordered = round_robin_ids(buckets)

        for batch in chunked(ordered, max_in_flight):
            tasks: List[asyncio.Task[tuple[Continents, JSONObj]]] = []
            for match_id in batch:
                region_code = match_id.split("_", 1)[0].lower()
                continent = REGION_TO_CONTINENT[region_code]
                url = endpoint.format(
                    continent=continent.value,
                    matchId=match_id,
                    api_key=riot_api._api_key,
                )
                tasks.append(
                    asyncio.create_task(_fetch_with_continent_obj(url, continent))
                )

            for fut in asyncio.as_completed(tasks):
                _, timeline_data = await fut
                yield Message(kind=StreamKind.TIMELINE, payload=timeline_data)
