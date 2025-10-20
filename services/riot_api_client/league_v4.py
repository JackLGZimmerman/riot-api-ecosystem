from __future__ import annotations

import asyncio
import itertools
from typing import (
    Tuple,
    List,
    Iterable,
    TypeAlias,
    TypeVar,
    Iterator,
    AsyncIterator,
)
from pydantic import ValidationError
from utils import reraise
from pydantic import SecretStr
from services.errors import LeagueListDTOError, MinifiedLeagueEntryError
from config.constants import (
    ENDPOINTS,
    JSON,
    JSONList,
    EliteTiers,
    Tiers,
    Regions,
    Divisions,
    Queues,
    URLTemplate,
)
from models import (
    LeagueEntryDTO,
    LeagueListDTO,
    MinifiedLeagueEntryDTO,
    EliteBoundsConfig,
    BasicBoundsConfig,
    BasicBoundSubConfig,
)
from services import RiotAPI
from services.riot_api_client.factories.base_factory import get_riot_api


LEAGUE_PAGE_UPPER_BOUND: int = 1024
MAX_IN_FLIGHT: int = 128
REQUEST_TIMEOUT: int = 10

riot_api: RiotAPI = get_riot_api()
api_key: SecretStr = riot_api.api_key

T = TypeVar("T")

UrlRegion: TypeAlias = tuple[str, Regions]
UrlTuple: TypeAlias = list[UrlRegion]
PageKey = Tuple[Regions, Queues, Tiers, Divisions]

# ==================================== Helpers ====================================


def chunked(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
    """Yield consecutive n-sized chunks from `iterable` (last may be smaller)."""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


async def _fetch_with_region_obj(url: str, region: Regions) -> tuple[Regions, JSON]:
    data = await asyncio.wait_for(
        riot_api.fetch_json(url=url, location=region),
        timeout=REQUEST_TIMEOUT,
    )
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict JSON, got {type(data)}")
    return region, data


async def _fetch_with_region_list(
    url: str, region: Regions
) -> tuple[Regions, JSONList]:
    data = await asyncio.wait_for(
        riot_api.fetch_json(url=url, location=region),
        timeout=REQUEST_TIMEOUT,
    )
    if not isinstance(data, list):
        raise TypeError(f"Expected list[dict] JSON, got {type(data)}")
    return region, data


def bounded_items(
    upper: EliteTiers | None, lower: EliteTiers | None
) -> List[EliteTiers]:
    result: List[EliteTiers] = []
    collecting = upper is None
    for item in list(EliteTiers):
        if not collecting:
            if item == upper:
                collecting = True
                result.append(item)
        else:
            result.append(item)
            if lower is not None and item == lower:
                break
    return result


def bounded_items_basic(
    upper: BasicBoundSubConfig,
    lower: BasicBoundSubConfig,
) -> list[tuple[Tiers, Divisions]]:
    """
    Collect (tier, division) pairs from the 'upper' bound down to 'lower' (inclusive).

    Defaults:
      - if upper.tier/division is None -> start immediately (DIAMOND I)
      - if lower.tier/division is None -> run until the very end (IRON IV)
    """
    result: list[tuple[Tiers, Divisions]] = []
    collecting = upper.tier is None and upper.division is None

    for tier in Tiers:
        for div in Divisions:
            pair = (tier, div)

            if not collecting:
                if tier == upper.tier and div == upper.division:
                    collecting = True
                    result.append(pair)
            else:
                result.append(pair)
                if (
                    lower.tier is not None
                    and lower.division is not None
                    and tier == lower.tier
                    and div == lower.division
                ):
                    return result

    return result


# ==================================================================================


async def stream_elite_players(
    queue_bounds: EliteBoundsConfig,
) -> AsyncIterator[MinifiedLeagueEntryDTO]:
    urls: UrlTuple = []
    template: URLTemplate = ENDPOINTS["league"]["elite"]

    for queue, bounds in queue_bounds.root.items():
        if hasattr(bounds, "collect") and not bounds.collect:
            continue

        upper: EliteTiers | None = bounds.upper
        lower: EliteTiers | None = bounds.lower
        tiers: list[EliteTiers] = bounded_items(upper, lower)

        for tier in tiers:
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
        tasks: list[asyncio.Task[tuple[Regions, JSON]]] = [
            asyncio.create_task(_fetch_with_region_obj(url, region))
            for url, region in batch
        ]

        for future in asyncio.as_completed(tasks):
            region, resp = await future  # region: Regions, resp: JSON

            with reraise(
                LeagueListDTOError,
                f"Failed to build LeagueListDTO (region={region.value})",
            ):
                dto = LeagueListDTO(**resp)

            with reraise(
                MinifiedLeagueEntryError,
                f"Failed to create entries via MinifiedLeagueEntryDTO.from_list (region={region.value})",
            ):
                for entry in MinifiedLeagueEntryDTO.from_list(dto, region=region):
                    yield entry


async def stream_sub_elite_players(
    queue_bounds: BasicBoundsConfig,
) -> AsyncIterator[MinifiedLeagueEntryDTO]:
    page_bounds: dict[PageKey, int] = await _discover_page_bounds(queue_bounds)
    template: URLTemplate = ENDPOINTS["league"]["by_queue_tier_division"]

    for (region, queue, tier, div), last_page in page_bounds.items():
        pages = range(1, last_page + 1)
        for batch in chunked(pages, MAX_IN_FLIGHT):
            tasks: list[asyncio.Task[tuple[Regions, JSONList]]] = [
                asyncio.create_task(
                    _fetch_with_region_list(
                        template.format(
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
                    yield MinifiedLeagueEntryDTO.from_entry(dto, region=region)


async def _discover_page_bounds(queue_bounds: BasicBoundsConfig) -> dict[PageKey, int]:
    template: URLTemplate = ENDPOINTS["league"]["by_queue_tier_division"]

    async def probe(
        region: Regions,
        queue: Queues,
        tier: Tiers,
        div: Divisions,
    ) -> tuple[PageKey, int]:
        low, high = 1, LEAGUE_PAGE_UPPER_BOUND + 1
        while low + 1 < high:
            mid = (low + high) // 2
            url = template.format(
                region=region,
                queue=queue,
                tier=tier,
                division=div,
                page=mid,
                api_key=api_key,
            )
            _: JSON | JSONList = await riot_api.fetch_json(url=url, location=region)
            low, high = (mid, high) if _ else (low, mid)

        return (region, queue, tier, div), low

    tasks: list[asyncio.Task[tuple[PageKey, int]]] = []
    for region in Regions:
        for queue, bounds in queue_bounds.root.items():
            upper: BasicBoundSubConfig = bounds.upper
            lower: BasicBoundSubConfig = bounds.lower
            brackets: list[tuple[Tiers, Divisions]] = bounded_items_basic(upper, lower)

            for tier, division in brackets:
                tasks.append(asyncio.create_task(probe(region, queue, tier, division)))

    bounds_map: dict[PageKey, int] = {}
    for finished in asyncio.as_completed(tasks):
        page_key, page_num = await finished
        bounds_map[page_key] = page_num

    return bounds_map


if __name__ == "__main__":
    # elite_bounds = EliteBoundsConfig.model_validate(
    #     {
    #         "collect": True,
    #         "RANKED_SOLO_5x5": {"upper": None, "lower": "GRANDMASTER"},
    #         "RANKED_FLEX_SR": {"upper": None, "lower": "MASTER"},
    #     }
    # )
    # basic = BasicBoundsConfig.model_validate(
    #     {
    #         "RANKED_SOLO_5x5": {
    #             "collect": True,
    #             "upper": {"tier": "DIAMOND", "division": "I"},
    #             "lower": {"tier": "EMERALD", "division": "II"},
    #         },
    #         "RANKED_FLEX_SR": {
    #             "collect": True,
    #             "upper": None,
    #             "lower": {"tier": "DIAMOND", "division": "II"},
    #         },
    #     }
    # )

    # async def main() -> None:
    #     async for entry in stream_sub_elite_players(basic):
    #         print(entry)

    # asyncio.run(main())
    pass
