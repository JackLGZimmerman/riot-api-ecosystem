from __future__ import annotations

import asyncio

from app.core.config.constants import Continent, Queues, Region
from app.services.riot_api_client.base import FetchJSONResult, FetchOutcome
from app.services.riot_api_client.match_ids import stream_match_ids
from app.services.riot_api_client.utils import PlayerCrawlState


class FakeRiotAPI:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def fetch_json_detailed(self, *, url, location):
        self.urls.append(url)
        return FetchJSONResult([], FetchOutcome.OK, 200)


def _state(puuid: str, region: Region, continent: Continent) -> PlayerCrawlState:
    return PlayerCrawlState(
        puuid=puuid,
        queue_type=Queues.RANKED_SOLO_5x5,
        region=region,
        continent=continent,
        next_page_start=0,
        base_url=f"https://example.test/{puuid}?start={{start}}",
    )


def test_matchid_initial_work_is_spread_by_region_not_continent() -> None:
    api = FakeRiotAPI()
    states = [
        _state("na", Region.NA1, Continent.AMERICAS),
        _state("la", Region.LA1, Continent.AMERICAS),
        _state("br", Region.BR1, Continent.AMERICAS),
        _state("eu", Region.EUW1, Continent.EUROPE),
    ]

    async def consume() -> None:
        async for _ in stream_match_ids(api, initial_states=states, max_in_flight=1):
            pass

    asyncio.run(consume())

    assert api.urls == [
        "https://example.test/na?start=0",
        "https://example.test/la?start=0",
        "https://example.test/br?start=0",
        "https://example.test/eu?start=0",
    ]
