from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


class RateLimiter:
    """
    Even-spread async token-bucket that **prints metrics for every call** when
    `debug=True`.

    Metrics line format:
        {n}/{capacity} | {rate:.2f} calls/s | {expected:.1f} expected | {drift:+.1f} drift
    """

    __slots__ = (
        "_capacity",
        "_tokens",
        "_fill_rate",
        "_last_checked",
        "_lock",
        "_debug",
        "_print",
        "_cycle_calls",
        "_cycle_start",
        "_time_period",
    )

    # ─────────────────────────── construction ─────────────────────────── #

    def __init__(
        self,
        *,
        max_calls: int,
        time_period: float,
        debug: bool = True,
        print_func: Callable[[str], None] = print,
    ) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be > 0")
        if time_period <= 0.0:
            raise ValueError("time_period must be > 0")

        self._capacity: int = max_calls
        self._time_period: float = time_period
        self._tokens: float = 0.0
        self._fill_rate: float = max_calls / time_period
        self._last_checked: float = time.perf_counter()
        self._lock = asyncio.Lock()

        self._debug: bool = debug
        self._print = print_func
        self._cycle_calls: int = 0
        self._cycle_start: float = self._last_checked

    # ───────────────────────── token acquisition ──────────────────────── #

    async def acquire(self) -> None:
        """
        Block until one token is available, then consume it.
        """
        while True:
            async with self._lock:
                now = time.perf_counter()
                elapsed = now - self._last_checked
                self._last_checked = now

                self._tokens += elapsed * self._fill_rate

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    self._after_grant(now)
                    return

                wait_time = (1.0 - self._tokens) / self._fill_rate

            await asyncio.sleep(wait_time)

    # ───────────────────── context-manager helpers ────────────────────── #

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *_exc) -> None:
        return

    # ───────────────────────────── decorator ──────────────────────────── #

    def decorator(self, fn: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
        """
        Decorate an **async** function so each invocation is rate-limited
        without needing an explicit ``async with limiter`` inside the function.
        """

        async def wrapper(*args: _P.args, **kwargs: _P.kwargs):
            async with self:
                return await fn(*args, **kwargs)

        return wrapper

    # ────────────────────────── private helpers ───────────────────────── #

    def _after_grant(self, now: float) -> None:
        elapsed = now - self._cycle_start

        if elapsed >= self._time_period:
            self._cycle_calls = 0
            self._cycle_start = now
            elapsed = 0.0

            self._tokens = 0.0

        self._cycle_calls += 1

        ideal = elapsed * self._fill_rate

        drift = self._cycle_calls - ideal

        if self._debug:
            self._print(
                f"{self._cycle_calls}/{self._capacity} | "
                f"ideal≈{ideal:.2f} | "
                f"drift={drift:+.2f}"
            )

    # ──────────────────────────── diagnostics ─────────────────────────── #

    @property
    def remaining(self) -> int:
        """
        Number of immediately available tokens (0 or 1 in this burst-free
        implementation).
        """
        now = time.perf_counter()
        elapsed = now - self._last_checked
        tokens = min(1.0, self._tokens + elapsed * self._fill_rate)
        return int(tokens)

    def __repr__(self) -> str:
        interval = 1 / self._fill_rate
        return (
            f"{self.__class__.__name__}(interval≈{interval:.4f}s, "
            f"token={self.remaining})"
        )
