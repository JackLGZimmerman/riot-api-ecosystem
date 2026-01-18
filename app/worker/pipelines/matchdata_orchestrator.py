from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.match_data import (
    stream_non_timeline_data,
    stream_timeline_data,
)
from app.worker.pipelines.orchestrator import (
    Collector,
    Loader,
    Orchestrator,
    Saver,
    SaveSpec,
)
from database.clickhouse.operations.matchids import load_matchids
from app.services.riot_api_client.parsers.non_timeline import (
    NonTimeline,
    MatchDataNonTimelineParsingOrchestrator,
)

@dataclass(frozen=True)
class MatchDataCollectorState:
    matchids: list[str]


class MatchDataOrchestrator(Orchestrator):
    def __init__(
        self,
        *,
        loader: Loader,
        non_timeline_collector: Collector,
        timeline_collector: Collector,
        saver: Saver,
    ) -> None:
        super().__init__(loader, non_timeline_collector, saver)
        self.non_timeline_collector = non_timeline_collector
        self.timeline_collector = timeline_collector

    async def run(self) -> None:
        state: MatchDataCollectorState = self.loader.load(-1)

        non_timeline_stream: AsyncIterator[dict[str, Any]] = (
            self.non_timeline_collector.collect(state.matchids)
        )
        timeline_stream: AsyncIterator[dict[str, Any]] = (
            self.timeline_collector.collect(state.matchids)
        )


        async def save_matchdata_matchids() -> None:
            await ...

        async def save_non_timeline() -> None:
            await ...

        async def save_timeline() -> None:
            await ...

        await self.saver.save(
            SaveSpec(save=save_matchdata_matchids),
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

    async def collect(self, state: MatchDataCollectorState) -> AsyncIterator[dict[str, Any]]:
        async for non_timeline in stream_non_timeline_data(
            state.matchids, riot_api=self.riot_api
        ):
            yield non_timeline


class MatchDataTimelineCollector(Collector):
    def __init__(self, riot_api: RiotAPI) -> None:
        self.riot_api = riot_api

    async def collect(self, state: MatchDataCollectorState) -> AsyncIterator[Any]:
        async for non_timeline in stream_timeline_data(
            state.matchids, riot_api=self.riot_api
        ):
            yield non_timeline


class MatchDataSaver(Saver):
    async def save(self, *specs: SaveSpec) -> None:
        for spec in specs:
            await spec.save()


if __name__ == "__main__":
    riot_api: RiotAPI = get_riot_api()

    loader = MatchDataLoader()

    non_timeline_collector = MatchDataNonTimelineCollector(riot_api=riot_api)
    timeline_collector = MatchDataTimelineCollector(riot_api=riot_api)

    saver = MatchDataSaver()

    orchestrator = MatchDataOrchestrator(
        loader=loader,
        non_timeline_collector=non_timeline_collector,
        timeline_collector=timeline_collector,
        saver=saver,
    )

    asyncio.run(orchestrator.run())
