from __future__ import annotations

import aiohttp
from contextlib import asynccontextmanager
from typing import Final, Dict, Optional, Literal

from config.settings import settings
from utils.rate_limiter import RateLimiter
from config.constants import Region, Continent, REGION_TO_CONTINENT
from config.constants.http import RETRYABLE_STATUS_CODES, ERROR_MESSAGES

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

LocationScope = Literal["region", "continent"]

class RiotAPI:
    _api_key: str = settings.api_key

    def __init__(
        self,
        *,
        calls_per_two_minutes: int = settings.calls_per_two_minutes,
        time_period_two_minutes: int = settings.time_period_two_minutes

    ) -> None:
        self.calls_per_two_minutes: Final[int] = calls_per_two_minutes
        self.time_period_two_minutes: Final[int] = time_period_two_minutes

        self._session: Optional[aiohttp.ClientSession] = None
        self._region_limiters: Dict[Region, RateLimiter] = {}
        self._continent_limiters: Dict[Continent, RateLimiter] = {}

    @classmethod
    def get_api_key(cls):
        return cls._api_key

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def _region_limiter(self, region: Region) -> RateLimiter:
        if region not in self._region_limiters:
            self._region_limiters[region] = RateLimiter(
                max_calls=self.calls_per_two_minutes, 
                time_period=self.time_period_two_minutes
            )
        return self._region_limiters[region]

    def _continent_limiter(self, location: Region | Continent) -> RateLimiter:
        if isinstance(location, Continent):
            continent = location
        else:
            continent = REGION_TO_CONTINENT[location]
        if continent not in self._continent_limiters:
            self._continent_limiters[continent] = RateLimiter(
                max_calls=self.calls_per_two_minutes, 
                time_period=self.time_period_two_minutes,
                print_func=lambda msg: print(f"[{continent}] {msg}")
            )
        return self._continent_limiters[continent]

    @asynccontextmanager
    async def _with_limiter(self, location: Region | Continent, scope: LocationScope):
        if scope == "region":
            limiter = self._region_limiter(location)
        elif scope == "continent":
            limiter = self._continent_limiter(location)
        else:
            raise ValueError(f"Invalid scope {scope}; must be 'region' or 'continent'")

        async with limiter:
            yield

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(
            lambda e: isinstance(e, aiohttp.ClientResponseError)
            and e.status in RETRYABLE_STATUS_CODES
        ),
    )
    async def fetch_json(
        self,
        *,
        url: str,
        location: Region | Continent,
        scope: LocationScope = "region",
    ) -> dict:
        async with self._with_limiter(location, scope):
            sess = await self.session()
            async with sess.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()


