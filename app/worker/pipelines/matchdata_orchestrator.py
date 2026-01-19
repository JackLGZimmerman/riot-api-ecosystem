from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol, TypeVar

from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.match_data import (
    stream_non_timeline_data,
    stream_timeline_data,
)
from app.services.riot_api_client.parsers.non_timeline import (
    MatchDataNonTimelineParsingOrchestrator,
    NonTimelineTables,
)
from app.services.riot_api_client.parsers.timeline import (
    MatchDataTimelineParsingOrchestrator,
    TimelineTables,
)
from app.worker.pipelines.orchestrator import (
    Collector,
    Loader,
    Orchestrator,
    Saver,
    SaveSpec,
)
from database.clickhouse.operations.matchids import load_matchids

TRaw = TypeVar("TRaw", contravariant=True)
TParsed = TypeVar("TParsed", covariant=True)


class Parser(Protocol[TRaw, TParsed]):
    def parse(self, raw: TRaw) -> TParsed: ...


@dataclass(frozen=True)
class MatchDataCollectorState:
    matchids: list[str]


@dataclass(frozen=True)
class NonTimelineParser(Parser[dict[str, Any], NonTimelineTables]):
    orch: MatchDataNonTimelineParsingOrchestrator

    def parse(self, raw: dict[str, Any]) -> NonTimelineTables:
        return self.orch.run(raw)


@dataclass(frozen=True)
class TimelineParser(Parser[dict[str, Any], TimelineTables]):
    orch: MatchDataTimelineParsingOrchestrator

    def parse(self, raw: dict[str, Any]) -> TimelineTables:
        return self.orch.run(raw)


class MatchDataOrchestrator(Orchestrator):
    def __init__(
        self,
        *,
        loader: Loader,
        non_timeline_collector: Collector,
        timeline_collector: Collector,
        non_timeline_parser: MatchDataNonTimelineParsingOrchestrator,
        timeline_parser: MatchDataTimelineParsingOrchestrator,
        saver: Saver,
    ) -> None:
        super().__init__(loader, non_timeline_collector, saver)
        self.non_timeline_collector = non_timeline_collector
        self.timeline_collector = timeline_collector
        self.non_timeline_parser = non_timeline_parser
        self.timeline_parser = timeline_parser

    async def run(self) -> None:
        state: MatchDataCollectorState = self.loader.load(-1)

        non_timeline_raw = self.non_timeline_collector.collect(state)
        timeline_raw = self.timeline_collector.collect(state)

        async def save_non_timeline() -> None:
            async for raw in non_timeline_raw:
                tables = await asyncio.to_thread(self.non_timeline_parser.run, raw)
                await persist_non_timeline_tables(tables)  # you implement

        async def save_timeline() -> None:
            async for raw in timeline_raw:
                tables = await asyncio.to_thread(self.timeline_parser.run, raw)
                await persist_timeline_tables(tables)  # you implement

        await self.saver.save(
            SaveSpec(save=save_non_timeline),
            SaveSpec(save=save_timeline),
        )


class MatchDataLoader(Loader):
    def __init__(self, max_workers: int = 16) -> None:
        self.max_workers = max_workers

    def load(self, ts: int) -> MatchDataCollectorState:
        matchids: list[str] = load_matchids()
        return MatchDataCollectorState(matchids=matchids)


class MatchDataNonTimelineCollector(Collector):
    def __init__(self, riot_api: RiotAPI) -> None:
        self.riot_api = riot_api

    async def collect(
        self, state: MatchDataCollectorState
    ) -> AsyncIterator[dict[str, Any]]:
        async for raw in stream_non_timeline_data(
            state.matchids, riot_api=self.riot_api
        ):
            yield raw


class MatchDataTimelineCollector(Collector):
    def __init__(self, riot_api: RiotAPI) -> None:
        self.riot_api = riot_api

    async def collect(
        self, state: MatchDataCollectorState
    ) -> AsyncIterator[dict[str, Any]]:
        async for raw in stream_timeline_data(state.matchids, riot_api=self.riot_api):
            yield raw


class MatchDataSaver(Saver):
    async def save(self, *specs: SaveSpec) -> None:
        async with asyncio.TaskGroup() as tg:
            for spec in specs:
                tg.create_task(spec.save())


if __name__ == "__main__":
    riot_api: RiotAPI = get_riot_api()

    loader = MatchDataLoader()

    non_timeline_collector = MatchDataNonTimelineCollector(riot_api=riot_api)
    timeline_collector = MatchDataTimelineCollector(riot_api=riot_api)

    non_timeline_parser = MatchDataNonTimelineParsingOrchestrator()
    timeline_parser = MatchDataTimelineParsingOrchestrator()

    saver = MatchDataSaver()

    orchestrator = MatchDataOrchestrator(
        loader=loader,
        non_timeline_collector=non_timeline_collector,
        timeline_collector=timeline_collector,
        non_timeline_parser=non_timeline_parser,
        timeline_parser=timeline_parser,
        saver=saver,
    )

    asyncio.run(orchestrator.run())
