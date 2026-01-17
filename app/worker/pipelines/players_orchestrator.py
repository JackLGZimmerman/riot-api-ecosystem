import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator

from app.infrastructure.files.utils import atomic_outputs
from app.models import MinifiedLeagueEntryDTO
from app.models.riot.league import (
    BASIC_BOUNDS,
    ELITE_BOUNDS,
    BasicBoundsConfig,
    EliteBoundsConfig,
)
from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.elite_players import stream_elite_players
from app.services.riot_api_client.subelite_players import stream_sub_elite_players
from app.worker.pipelines.orchestrator import (
    Collector,
    Loader,
    Orchestrator,
    Saver,
    SaveSpec,
)


@dataclass(frozen=True)
class PlayerCollectorState:
    elite_bounds: EliteBoundsConfig
    subelite_bounds: BasicBoundsConfig


class PlayersOrchestrator(Orchestrator):
    def __init__(
        self,
        loader: Loader,
        collector: Collector,
        saver: Saver,
    ):
        super().__init__(loader, collector, saver)

    async def run(self) -> None:
        state: PlayerCollectorState = self.loader.load()

        players_stream: AsyncIterator[MinifiedLeagueEntryDTO] = self.collector.collect(
            state
        )

        async def save_players() -> None:
            await asyncio.to_thread(_write_players_jsonl_zstd, players_stream)

        await self.saver.save(
            SaveSpec(save=save_players),
        )


class PlayerLoader(Loader):
    def load(self) -> PlayerCollectorState:
        return PlayerCollectorState(
            elite_bounds=ELITE_BOUNDS,
            subelite_bounds=BASIC_BOUNDS,
        )


class PlayerCollector(Collector):
    def __init__(self, riot_api: RiotAPI):
        self.riot_api = riot_api

    async def collect(
        self, state: PlayerCollectorState
    ) -> AsyncGenerator[MinifiedLeagueEntryDTO, None]:
        async for player in stream_elite_players(
            state.elite_bounds, riot_api=self.riot_api
        ):
            yield player

        async for player in stream_sub_elite_players(
            state.subelite_bounds, riot_api=self.riot_api
        ):
            yield player


class PlayerSaver(Saver):
    async def save(self, *specs: SaveSpec):
        for spec in specs:
            await spec.save()


if __name__ == "__main__":
    riot_api: RiotAPI = get_riot_api()

    loader = PlayerLoader()
    collector = PlayerCollector(riot_api)
    saver = PlayerSaver()

    orchestrator = PlayersOrchestrator(loader=loader, collector=collector, saver=saver)

    asyncio.run(orchestrator.run())
