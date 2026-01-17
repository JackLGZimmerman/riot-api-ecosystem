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
class DualWindowSpec:
    location: Region | Continent
    long_calls: int = 100
    long_period_s: float = 120.0


class _DebugState:
    __slots__ = (
        "location",
        "interval",
        "expected_rate",
        "period",
        "_lock",
        "_period_start",
        "_permits",
    )

    def __init__(
        self, *, location: str, interval: float, expected_rate: float, period: float
    ) -> None:
        self.location = location
        self.interval = interval
        self.expected_rate = expected_rate
        self.period = period

        self._lock = asyncio.Lock()
        self._period_start: float | None = None
        self._permits: int = 0

    async def record(self, now: float) -> None:
        async with self._lock:
            if self._period_start is None:
                self._period_start = now

            elapsed = now - self._period_start
            if elapsed >= self.period:
                periods = int(elapsed // self.period)
                self._period_start += periods * self.period
                self._permits = 0
                elapsed = now - self._period_start

            self._permits += 1
            permits = self._permits
            period_start = self._period_start

        elapsed2 = max(0.0, now - period_start)
        observed = (permits / elapsed2) if elapsed2 > 0 else 0.0
        delta = observed - self.expected_rate

        logger.debug(
            "LimiterStats",
            extra={
                "location": self.location,
                "permits": permits,
                "interval_s": round(self.interval, 3),
                "avg_rate_per_s": round(observed, 3),
                "expected_rate_per_s": round(self.expected_rate, 3),
                "delta_rate_per_s": round(delta, 3),
                "elapsed_s": round(elapsed2, 3),
            },
        )

        print(
            f"[{self.location}] "
            f"permits={permits} "
            f"interval={self.interval:.3f}s "
            f"avg_rate={observed:.3f}/s exp={self.expected_rate:.3f}/s Δ={delta:+.3f}/s "
            f"t={elapsed2:.3f}s"
        )


class Limiter:
    """
    Steady stream limiter (no bursts):
      - emits 1 permit every (long_period_s / long_calls) seconds

    Concurrency-safe: each caller is assigned a scheduled slot on a single timeline.
    Uses monotonic time (loop.time()).
    """

    def __init__(self, spec: DualWindowSpec, *, debug: bool = False) -> None:
        self._location: Final[str] = spec.location
        self._calls: Final[int] = int(spec.long_calls)
        self._period: Final[float] = float(spec.long_period_s)

        self._interval: Final[float] = self._period / self._calls

        self._lock = asyncio.Lock()
        self._next_at: float | None = None

        self._debug: _DebugState | None = None
        if debug:
            self._debug = _DebugState(
                location=self._location,
                interval=self._interval,
                expected_rate=self._calls / self._period,
                period=self._period,
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

        dbg = self._debug
        if dbg is not None:
            await dbg.record(loop.time())

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
        inner,
        *,
        location: Continent | Region,
        period: float,
        export: Callable[..., None],
    ) -> None:
        self._inner = inner
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
        await self._inner.acquire()
        now = asyncio.get_running_loop().time()
        await self._record_and_export(now)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *_exc):
        return


async def main():
    async def dummy_task(limiter: Limiter):
        async with limiter:
            await asyncio.sleep(200 / 1000)

    sustained_calls = 100
    sustained_period = 120

    spec = DualWindowSpec(
        location=Region.EUW1,
        long_calls=sustained_calls,
        long_period_s=sustained_period,
    )

    limiter = Limiter(spec, debug=True)
    print("Starting the limiter test")
    async with asyncio.TaskGroup() as tg:
        for _ in range(500):
            tg.create_task(dummy_task(limiter))
    print("Finished the limiter test")


if __name__ == "__main__":
    asyncio.run(main())
