# app/services/riot_api_client/base.py

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Final

import aiohttp
from pydantic import PositiveFloat, PositiveInt
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.api.v1.metrics.telemetry import (
    export_http_error_code_counter,
    export_location_event,
)
from app.core.config.constants import JSON, Continent, JSONList, Region
from app.core.config.constants.generic import RETRYABLE
from app.core.config.settings import settings
from app.services.riot_api_client.rate_limiter import (
    RateLimitSpec,
    Limiter,
    TelemetryLimiter,
)

MAX_BODY_PREVIEW = 200

logger = logging.getLogger(__name__)


class FetchOutcome(str, Enum):
    OK = "ok"
    HTTP_NON_RETRYABLE = "http_non_retryable"
    NON_JSON = "non_json"
    RETRY_EXHAUSTED = "retry_exhausted"


@dataclass(frozen=True)
class FetchJSONResult:
    data: JSON | JSONList | None
    outcome: FetchOutcome
    status: int | None = None


def mask_api_key(url: str) -> str:
    return re.sub(r"(api_key=)[^&]+", r"\1*", url)


def _is_retryable_fetch_exception(e: BaseException) -> bool:
    if isinstance(e, aiohttp.ClientResponseError):
        return e.status in RETRYABLE
    return isinstance(e, (aiohttp.ClientConnectionError, asyncio.TimeoutError))


def _retry_exhausted_result(_) -> FetchJSONResult:
    return FetchJSONResult(
        data=None,
        outcome=FetchOutcome.RETRY_EXHAUSTED,
    )


@lru_cache(maxsize=None)
def _limiter(location_key: Region | Continent, calls: int, time_period: float):
    sustained_calls = int(calls)
    sustained_period = float(time_period)

    core = Limiter(
        RateLimitSpec(
            location=location_key,
            calls=sustained_calls,
            period_s=sustained_period,
        ),
        debug=True,
    )

    return TelemetryLimiter(
        core,
        location=location_key,
        period=sustained_period,
        export=export_location_event,
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
    ) -> None:
        self._api_key: Final[str] = api_key or settings.api_key.get_secret_value()
        self.calls: Final[PositiveInt] = calls
        self.time_period: Final[PositiveFloat] = time_period

        self._session: aiohttp.ClientSession | None = None

    @property
    def api_key(self):
        return self._api_key

    async def __aenter__(self) -> RiotAPI:
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
        if self._session is not None and not self._session.closed:
            await self._session.close()

    # ==========================================================

    @retry(
        reraise=False,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_fetch_exception),
        retry_error_callback=_retry_exhausted_result,
    )
    async def fetch_json_detailed(
        self,
        *,
        url: str,
        location: Region | Continent,
    ) -> FetchJSONResult:
        """
        Fetch JSON from Riot API with:
        - per-(location, calls, period) rate limiter
        - retry on transient HTTP errors (429/5xx)
        - single enriched log line per granted token from RateLimiter
        """

        limiter = _limiter(
            location,
            self.calls,
            self.time_period,
        )

        if self._session is None or self._session.closed:
            raise RuntimeError(
                "RiotAPI session is not initialised. "
                "Use `async with RiotAPI()` when calling fetch_json."
            )

        session = self._session

        async with limiter:
            headers = {"X-Riot-Token": self.api_key}
            try:
                async with session.get(url, headers=headers) as resp:
                    status: int = resp.status

                    if not 200 <= status < 300:
                        export_http_error_code_counter(status)

                        if status in RETRYABLE:
                            resp.raise_for_status()

                        logger.warning(
                            "NonRetryableHTTP status=%s url=%s location=%s",
                            status,
                            mask_api_key(str(resp.url)),
                            location,
                        )
                        return FetchJSONResult(
                            data=None,
                            outcome=FetchOutcome.HTTP_NON_RETRYABLE,
                            status=status,
                        )
                    try:
                        return FetchJSONResult(
                            data=await resp.json(),
                            outcome=FetchOutcome.OK,
                            status=status,
                        )
                    except (aiohttp.ContentTypeError, Exception):
                        body = await resp.text()
                        preview = body.replace("\n", " ")[:MAX_BODY_PREVIEW]
                        logger.warning(
                            "NonJSONResponse status=%s url=%s location=%s len=%d preview=%r",
                            status,
                            mask_api_key(str(resp.url)),
                            location,
                            len(body),
                            preview,
                        )
                        return FetchJSONResult(
                            data=None,
                            outcome=FetchOutcome.NON_JSON,
                            status=status,
                        )
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                raise

    async def fetch_json(
        self,
        *,
        url: str,
        location: Region | Continent,
    ) -> JSON | JSONList | None:
        result = await self.fetch_json_detailed(url=url, location=location)
        if result.outcome is FetchOutcome.RETRY_EXHAUSTED:
            logger.warning(
                "RetryExhaustedHTTP url=%s location=%s",
                mask_api_key(url),
                location,
            )
        return result.data


def get_riot_api(
    *,
    api_key: str | None = None,
    calls: PositiveInt | None = None,
    time_period: PositiveFloat | None = None,
) -> RiotAPI:
    """
    Factory for creating a RiotAPI instance.
    """
    return RiotAPI(
        api_key=api_key,
        calls=calls or settings.rate_limit_calls,
        time_period=time_period or settings.rate_limit_period,
    )
