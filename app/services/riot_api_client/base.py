# app/services/riot_api_client/base.py

from __future__ import annotations

import logging
from functools import lru_cache
from http import HTTPStatus
from typing import Final, Literal

import aiohttp
from pydantic import PositiveFloat, PositiveInt
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config.constants import JSON, Continents, JSONList, Regions
from app.core.config.settings import settings
from app.services.utils.rate_limiter import RateLimiter

RETRYABLE = {
    HTTPStatus.TOO_MANY_REQUESTS.value,  # 429
    HTTPStatus.INTERNAL_SERVER_ERROR.value,  # 500
    HTTPStatus.BAD_GATEWAY.value,  # 502
    HTTPStatus.SERVICE_UNAVAILABLE.value,  # 503
    HTTPStatus.GATEWAY_TIMEOUT.value,  # 504
}

LocationScope = Literal["region", "continent"]

logger = logging.getLogger("limiter")

# ---------- LRU-cached limiter factories (singletons per key) ----------


@lru_cache(maxsize=None)
def _limiter(
    location_key: Regions | Continents,
    calls: int,
    time_period: float,
) -> RateLimiter:
    """
    One RateLimiter instance per (location, calls, period) tuple.
    """

    def log_with_location(msg: str) -> None:
        logger.debug("[%s] %s", location_key.value, msg)

    return RateLimiter(
        max_calls=calls,
        time_period=time_period,
        print_func=log_with_location,
    )


class RiotAPI:
    """
    Thin wrapper around aiohttp + Riot rate limiting.

    Designed for **per-task lifetime**:

      - You typically create it inside a pipeline run, e.g.:

            async with RiotAPI() as riot:
                data = await riot.fetch_json(...)

      - Optionally accept an external aiohttp.ClientSession to share
        connections with other code, in which case this class will NOT
        close that session.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        calls: PositiveInt = settings.rate_limit_calls,
        time_period: PositiveFloat = settings.rate_limit_period,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._api_key: Final[str] = api_key or settings.api_key.get_secret_value()
        self.calls: Final[PositiveInt] = calls
        self.time_period: Final[PositiveFloat] = time_period

        self._session: aiohttp.ClientSession | None = session
        self._external_session: Final[bool] = session is not None

    # -------- lifecycle / context manager --------

    async def __aenter__(self) -> "RiotAPI":
        """
        Async context manager entry.

        Ensures there is an open ClientSession for this instance.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """
        Async context manager exit.

        Closes the internal session if this instance created it.
        """
        await self.close()

    @property
    def api_key(self) -> str:
        """Expose the configured API key as a plain string."""
        return self._api_key

    async def session(self) -> aiohttp.ClientSession:
        """
        Return a ClientSession for this instance.

        If no session exists (or it was closed), a new one is created.
        The session is tied to the lifetime of this RiotAPI instance,
        not globally to the process.
        """
        if self._session is None or self._session.closed:
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
        """
        Fetch JSON from Riot API with:
        - per-(location, calls, period) rate limiter
        - retry on transient HTTP errors (429/5xx)
        """
        limiter: RateLimiter = _limiter(
            location,
            int(self.calls),
            float(self.time_period),
        )

        async with limiter:
            sess = await self.session()
            async with sess.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def close(self) -> None:
        """
        Close the internal session if we own it.

        If a session was injected via the constructor, we assume the caller
        is responsible for closing it.
        """
        if (
            self._session is not None
            and not self._session.closed
            and not self._external_session
        ):
            await self._session.close()
        if not self._external_session:
            self._session = None


def get_riot_api(
    *,
    api_key: str | None = None,
    calls: PositiveInt | None = None,
    time_period: PositiveFloat | None = None,
    session: aiohttp.ClientSession | None = None,
) -> RiotAPI:
    """
    Convenience factory for creating a RiotAPI instance.

    This is now just a thin wrapper (no caching):
      - Use it when you want a slightly nicer call site, or
      - Construct RiotAPI(...) directly.

    Typical usage in a pipeline:

        async with get_riot_api() as riot:
            data = await riot.fetch_json(...)
    """
    return RiotAPI(
        api_key=api_key,
        calls=calls or settings.rate_limit_calls,
        time_period=time_period or settings.rate_limit_period,
        session=session,
    )
