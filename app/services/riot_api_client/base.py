from __future__ import annotations

from functools import lru_cache
from http import HTTPStatus
from typing import Final, Literal

import aiohttp
from pydantic import PositiveFloat, PositiveInt, SecretStr
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from app.services.utils.rate_limiter import RateLimiter

from app.core.config.constants import JSON, Continents, JSONList, Regions
from app.core.config.settings import settings

RETRYABLE = {
    HTTPStatus.TOO_MANY_REQUESTS.value,  # 429
    HTTPStatus.INTERNAL_SERVER_ERROR.value,  # 500
    HTTPStatus.BAD_GATEWAY.value,  # 502
    HTTPStatus.SERVICE_UNAVAILABLE.value,  # 503
    HTTPStatus.GATEWAY_TIMEOUT.value,  # 504
}


LocationScope = Literal["region", "continent"]

# ---------- LRU-cached limiter factories (singletons per key) ----------


@lru_cache(maxsize=None)
def _limiter(
    location_key: Regions | Continents,
    calls: int,
    time_period: float,
) -> RateLimiter:
    return RateLimiter(max_calls=calls, time_period=time_period)


class RiotAPI:
    _api_key: SecretStr = settings.api_key

    def __init__(
        self,
        *,
        calls: PositiveInt = settings.rate_limit_calls,
        time_period: PositiveFloat = settings.rate_limit_period,
    ) -> None:
        self.calls: Final[PositiveInt] = calls
        self.time_period: Final[PositiveFloat] = time_period
        self._session: aiohttp.ClientSession | None = None

    @property
    def api_key(cls):
        return cls._api_key

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    # ==========================================================

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(
            lambda e: isinstance(e, aiohttp.ClientResponseError)
            and e.status in RETRYABLE
        ),
    )
    async def fetch_json(
        self,
        *,
        url: str,
        location: Regions | Continents,
    ) -> JSON | JSONList:
        limiter: RateLimiter = _limiter(location)
        async with limiter:
            sess = await self.session()
            async with sess.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


riot_api = RiotAPI()


@lru_cache(maxsize=1)
def get_riot_api() -> RiotAPI:
    """Global RiotHTTP client (dual-window rate limiter inside)."""
    return RiotAPI(
        calls=settings.rate_limit_calls,
        time_period=settings.rate_limit_period,
    )
