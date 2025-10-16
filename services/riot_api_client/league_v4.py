from __future__ import annotations

import asyncio
import itertools
from typing import Sequence, Tuple, Optional, List
from pydantic import ValidationError
from utils.helpers import reraise
from services.errors import LeagueListDTOError, MinifiedLeagueEntryError
from config.constants import (
    ENDPOINTS,
    cumulative_elite_tier_mapping,
    cumulative_tier_mapping,
    cumulative_division_mapping,
    Regions,
    Queues,
    Tiers,
    Divisions,
    EliteTiers,
)
from config import settings
from models.riot.league import LeagueEntryDTO, LeagueListDTO, MinifiedLeagueEntryDTO
from services import RiotAPI

from base import RiotAPI
from services.riot_api_client.factories.base_factory import get_riot_api

EliteQueueBound = Sequence[Tuple[Queues, Optional[EliteTiers]]]
SubEliteQueueBound = Sequence[Tuple[Queues, Optional[Tuple[Tiers, Divisions]]]]

LEAGUE_PAGE_UPPER_BOUND: int = settings.league_page_upper_bound
MAX_IN_FLIGHT: int = 128
REQUEST_TIMEOUT: int = 10

riot_api: RiotAPI = get_riot_api()
api_key = riot_api.get_api_key()


# ==================================== Helpers ====================================

def chunked(iterable, n: int):
    """Yield consecutive n-sized chunks of iterable (last may be smaller)."""
    it = iter(iterable)
    while (batch := list(itertools.islice(it, n))):
        yield batch

async def _fetch_with_region(url: str, region: Regions):
    data = await asyncio.wait_for(
        riot_api.fetch_json(url=url, location=region),
        timeout=REQUEST_TIMEOUT,
    )
    return region, data

# ==================================================================================


async def stream_elite_players(queue_bounds: EliteQueueBound):
    urls: List[Tuple[str, str]] = []
    template = ENDPOINTS.league.elite

    for queue, base_tier in queue_bounds:
        if base_tier is None:
            continue
        for tier in cumulative_elite_tier_mapping()[base_tier]:
            urls.extend(
                (
                    template.format(
                        elite_tier=tier.lower(),
                        region=region,
                        queue=queue,
                        api_key=api_key,
                    ),
                    region,
                )
                for region in Regions
            )

    for batch in chunked(urls, MAX_IN_FLIGHT):
        tasks = [
            asyncio.create_task(_fetch_with_region(url, region))
            for url, region in batch
        ]

        for future in asyncio.as_completed(tasks):
            region, resp = await future

            with reraise(LeagueListDTOError, f"Failed to build LeagueListDTO (region={region})"):
                dto = LeagueListDTO(**resp)

            with reraise(
                MinifiedLeagueEntryError,
                f"Failed to create entries via MinifiedLeagueEntryDTO.from_list (region={region})",
            ):
                for entry in MinifiedLeagueEntryDTO.from_list(dto, region):
                    yield entry

async def stream_sub_elite_players(queue_bounds: SubEliteQueueBound):
    page_bounds = await _discover_page_bounds(queue_bounds)
    tmpl = str(ENDPOINTS.league.by_queue_tier_division)

    for (region, queue, tier, div), last_page in page_bounds.items():
        pages = range(1, last_page + 1)
        for batch in chunked(pages, MAX_IN_FLIGHT):
            tasks = [
                asyncio.create_task(
                    _fetch_with_region(
                        tmpl.format(
                            region=region,
                            queue=queue,
                            tier=tier,
                            division=div,
                            page=pg,
                            api_key=api_key,
                        ),
                        region,
                    )
                )
                for pg in batch
            ]

            for future in asyncio.as_completed(tasks):
                region, records = await future
                for raw in records:
                    try:
                        dto = LeagueEntryDTO(**raw)
                    except ValidationError:
                        continue
                    yield MinifiedLeagueEntryDTO.from_entry(dto, region)

async def _discover_page_bounds(queue_bounds: SubEliteQueueBound):
    tmpl = str(ENDPOINTS.league.by_queue_tier_division)

    async def probe(region, queue, tier, div):
        low, high = 1, LEAGUE_PAGE_UPPER_BOUND + 1
        while low + 1 < high:
            mid = (low + high) // 2
            url = tmpl.format(
                region=region,
                queue=queue,
                tier=tier,
                division=div,
                page=mid,
                api_key=api_key,
            )
            data = await riot_api.fetch_json(url=url, location=region, scope="region")
            low, high = (mid, high) if data else (low, mid)
        return (region, queue, tier, div), low

    tasks: List[asyncio.Task] = []
    for region in Regions:
        for queue, opt in queue_bounds:
            if opt is None:
                continue
            base_tier, base_div = opt
            for tier in cumulative_tier_mapping()[base_tier]:
                for div in cumulative_division_mapping()[base_div]:
                    tasks.append(
                        asyncio.create_task(probe(region, queue, tier, div))
                    )

    bounds_map: dict[Tuple, int] = {}
    for finished in asyncio.as_completed(tasks):
        key, page = await finished
        bounds_map[key] = page

    return bounds_map
