from __future__ import annotations
# ruff: noqa: E402

import asyncio
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.core.config.constants.geography import Continent
from app.services.riot_api_client.base import FetchJSONResult, FetchOutcome, RiotAPI
from app.services.riot_api_client.match_data import stream_match_data


class FakeRiotAPI(RiotAPI):
    def __init__(self, responses: list[FetchJSONResult]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    async def fetch_json_detailed(self, *, url: str, location: str) -> FetchJSONResult:
        self.calls.append((url, location))
        return next(self._responses)


def test_stream_match_data_surfaces_404_status_and_correct_route() -> None:
    riot_api = FakeRiotAPI(
        [
            FetchJSONResult(
                data=None,
                outcome=FetchOutcome.HTTP_NON_RETRYABLE,
                status=404,
            ),
            FetchJSONResult(
                data=None,
                outcome=FetchOutcome.HTTP_NON_RETRYABLE,
                status=404,
            ),
        ]
    )

    async def collect():
        return [
            item
            async for item in stream_match_data(
                ["TW2_191351674", "EUW1_6914045680"],
                endpoint_type="by_match_id",
                riot_api=riot_api,
            )
        ]

    results = asyncio.run(collect())

    assert {result.match_id for result in results} == {
        "TW2_191351674",
        "EUW1_6914045680",
    }
    assert {result.status for result in results} == {404}
    assert {location for _, location in riot_api.calls} == {
        Continent.SEA.value,
        Continent.EUROPE.value,
    }
