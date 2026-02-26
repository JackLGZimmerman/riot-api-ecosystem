from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from app.core.config.constants import Continent, Region

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitSpec:
    location: Region | Continent
    calls: int = 100
    period_s: float = 120.0


class _LimiterDebugMetrics:
    __slots__ = (
        "location",
        "interval_s",
        "expected_rate_per_s",
        "period_s",
        "_lock",
        "_period_start",
        "_permits",
    )

    def __init__(
        self,
        *,
        location: Region | Continent,
        interval_s: float,
        expected_rate_per_s: float,
        period_s: float,
    ) -> None:
        self.location = location
        self.interval_s = interval_s
        self.expected_rate_per_s = expected_rate_per_s
        self.period_s = period_s

        self._lock = asyncio.Lock()
        self._period_start: float | None = None
        self._permits: int = 0

    async def record(self, now: float) -> None:
        async with self._lock:
            if self._period_start is None:
                self._period_start = now

            elapsed_in_window_s = now - self._period_start
            if elapsed_in_window_s >= self.period_s:
                periods = int(elapsed_in_window_s // self.period_s)
                self._period_start += periods * self.period_s
                self._permits = 0
                elapsed_in_window_s = now - self._period_start

            self._permits += 1
            permits = self._permits
            period_start = self._period_start

        elapsed_since_window_start_s = max(0.0, now - period_start)
        observed_rate_per_s = (
            permits / elapsed_since_window_start_s
            if elapsed_since_window_start_s > 0
            else 0.0
        )
        delta_rate_per_s = observed_rate_per_s - self.expected_rate_per_s

        logger.debug(
            "LimiterStats",
            extra={
                "location": self.location,
                "permits": permits,
                "interval_s": round(self.interval_s, 3),
                "avg_rate_per_s": round(observed_rate_per_s, 3),
                "expected_rate_per_s": round(self.expected_rate_per_s, 3),
                "delta_rate_per_s": round(delta_rate_per_s, 3),
                "elapsed_s": round(elapsed_since_window_start_s, 3),
            },
        )


class Limiter:
    """
    Steady stream limiter (no bursts):
      - emits 1 permit every (period_s / calls) seconds

    Concurrency-safe: each caller is assigned a scheduled slot on a single timeline.
    Uses monotonic time (loop.time()).
    """

    def __init__(self, spec: RateLimitSpec, *, debug: bool = False) -> None:
        self._location: Final[Region | Continent] = spec.location
        self._calls: Final[int] = int(spec.calls)
        self._period: Final[float] = float(spec.period_s)

        self._interval: Final[float] = self._period / self._calls

        self._lock = asyncio.Lock()
        self._next_at: float | None = None

        self._debug: _LimiterDebugMetrics | None = None
        if debug:
            self._debug = _LimiterDebugMetrics(
                location=self._location,
                interval_s=self._interval,
                expected_rate_per_s=self._calls / self._period,
                period_s=self._period,
            )

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()

        async with self._lock:
            now = loop.time()
            if self._next_at is None:
                self._next_at = now

            base = self._next_at if self._next_at >= now else now

            scheduled = base
            self._next_at = scheduled + self._interval

            delay = scheduled - now

        if delay > 0:
            await asyncio.sleep(delay)

        debug_metrics = self._debug
        if debug_metrics is not None:
            await debug_metrics.record(loop.time())

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *_exc):
        return False


class TelemetryLimiter:
    """
    Minimal telemetry:
      - store permit timestamps in a deque
      - drop items older than (now - period)
      - export rate = len(window) / period
    """

    def __init__(
        self,
        wrapped_limiter,
        *,
        location: Continent | Region,
        period: float,
        export: Callable[..., None],
    ) -> None:
        self._wrapped_limiter = wrapped_limiter
        self._location = location
        self._period: Final[float] = float(period)
        self._export = export

        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def _record_and_export(self, now: float) -> None:
        async with self._lock:
            self._times.append(now)
            cutoff = now - self._period
            while self._times and self._times[0] <= cutoff:
                self._times.popleft()

            rate = len(self._times) / self._period
        self._export(location=self._location, rate=rate)

    async def acquire(self) -> None:
        await self._wrapped_limiter.acquire()
        now = asyncio.get_running_loop().time()
        await self._record_and_export(now)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *_exc):
        return
