from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol
from uuid import UUID


@dataclass(frozen=True)
class OrchestrationContext:
    ts: int
    run_id: UUID
    pipeline: str


class Loader(Protocol):
    def load(self, ctx: OrchestrationContext) -> Any: ...


class Collector(Protocol):
    def collect(self, state: Any, ctx: OrchestrationContext) -> AsyncIterator[Any]: ...


class Saver(Protocol):
    async def save(
        self, items: AsyncIterator[Any], state: Any, ctx: OrchestrationContext
    ) -> None: ...


class Orchestrator:
    def __init__(
        self, pipeline: str, loader: Loader, collector: Collector, saver: Saver
    ):
        self.pipeline = pipeline
        self.loader = loader
        self.collector = collector
        self.saver = saver

    async def run(self) -> None:
        raise NotImplementedError
