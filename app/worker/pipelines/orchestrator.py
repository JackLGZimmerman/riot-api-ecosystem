from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Protocol,
)


@dataclass(frozen=True)
class OrchestrationContext:
    orchestration_start_time: int


@dataclass(frozen=True)
class SaveSpec:
    save: Callable[[], Awaitable[None]]


class Loader(Protocol):
    def load(self, ts: int) -> Any: ...


class Collector(Protocol):
    def collect(self, state: Any) -> AsyncIterator[Any]: ...


class Saver(Protocol):
    async def save(self, *specs: SaveSpec) -> None: ...


class Orchestrator:
    def __init__(self, loader: Loader, collector: Collector, saver: Saver):
        self.loader = loader
        self.collector = collector
        self.saver = saver

    async def run(self) -> None:
        raise NotImplementedError
