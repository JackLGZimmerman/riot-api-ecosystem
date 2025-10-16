from __future__ import annotations

import aiohttp
from typing import Final, Literal
from http import HTTPStatus
from functools import lru_cache

from config.settings import settings
from utils.rate_limiter import RateLimiter
from config.constants import Regions, Continents, REGION_TO_CONTINENT

RETRYABLE = {
    HTTPStatus.TOO_MANY_REQUESTS.value,      # 429
    HTTPStatus.INTERNAL_SERVER_ERROR.value,  # 500
    HTTPStatus.BAD_GATEWAY.value,            # 502
    HTTPStatus.SERVICE_UNAVAILABLE.value,    # 503
    HTTPStatus.GATEWAY_TIMEOUT.value,        # 504
}

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

LocationScope = Literal["region", "continent"]

# ---------- LRU-cached limiter factories (singletons per key) ----------

@lru_cache(maxsize=None)
def _limiter(
    location_key: Regions | Continents,
    calls_per_two_minutes: int,
    time_period_two_minutes: int,
) -> RateLimiter:
    return RateLimiter(
        max_calls=calls_per_two_minutes,
        time_period=time_period_two_minutes,
    )


class RiotAPI:
    _api_key: str = settings.api_key

    def __init__(
        self,
        *,
        calls_per_two_minutes: int = settings.calls_per_two_minutes,
        time_period_two_minutes: int = settings.time_period_two_minutes,
    ) -> None:
        self.calls_per_two_minutes: Final[int] = calls_per_two_minutes
        self.time_period_two_minutes: Final[int] = time_period_two_minutes
        self._session: aiohttp.ClientSession | None = None

    @property
    def get_api_key(cls):
        return cls._api_key

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
    
    # ==========================================================

    def _get_limiter(self, location: Regions | Continents) -> RateLimiter:
        return _limiter(location, self.calls_per_two_minutes, self.time_period_two_minutes)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(
            lambda e: isinstance(e, aiohttp.ClientResponseError) and e.status in RETRYABLE
        ),
    )
    async def fetch_json(
        self,
        *,
        url: str,
        location: Regions | Continents,
    ) -> dict:
        limiter = self._get_limiter(location)
        async with limiter:
            sess = await self.session()
            async with sess.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

