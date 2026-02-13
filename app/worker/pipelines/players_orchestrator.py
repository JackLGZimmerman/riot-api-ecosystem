import asyncio
from dataclasses import dataclass
import time
from typing import AsyncIterator, Any
from uuid import uuid4
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
    OrchestrationContext,
)
from database.clickhouse.operations.players import (
    insert_players_stream_in_batches,
    delete_partial_players_run,
)


@dataclass(frozen=True)
class PlayerCollectorState:
    elite_bounds: EliteBoundsConfig
    subelite_bounds: BasicBoundsConfig


class PlayersOrchestrator(Orchestrator):
    def __init__(
        self, pipeline: str, loader: Loader, collector: Collector, saver: Saver
    ):
        super().__init__(pipeline=pipeline, loader=loader, collector=collector, saver=saver)

    async def run(self) -> None:
        ctx = OrchestrationContext(
            ts=int(time.time()),
            run_id=uuid4(),
            pipeline=self.pipeline,
        )

        state: PlayerCollectorState = self.loader.load(ctx)
        players_stream: AsyncIterator[MinifiedLeagueEntryDTO] = self.collector.collect(
            state, ctx
        )

        await self.saver.save(players_stream, state, ctx)


class PlayerLoader(Loader):
    def load(self, ctx: OrchestrationContext) -> PlayerCollectorState:
        return PlayerCollectorState(
            elite_bounds=ELITE_BOUNDS,
            subelite_bounds=BASIC_BOUNDS,
        )


class PlayerCollector(Collector):
    def __init__(self, riot_api: RiotAPI):
        self.riot_api = riot_api

    async def collect(
        self,
        state: PlayerCollectorState,
        ctx: OrchestrationContext,
    ) -> AsyncIterator[MinifiedLeagueEntryDTO]:
        async for player in stream_elite_players(
            state.elite_bounds, riot_api=self.riot_api
        ):
            yield player

        async for player in stream_sub_elite_players(
            state.subelite_bounds, riot_api=self.riot_api
        ):
            yield player


class PlayerSaver(Saver):
    async def save(
        self,
        items: AsyncIterator[MinifiedLeagueEntryDTO],
        state: Any,
        ctx: OrchestrationContext,
    ) -> None:
        try:
            await insert_players_stream_in_batches(
                items,
                ts=ctx.ts,
                run_id=ctx.run_id,
            )
        except Exception:
            delete_partial_players_run(ctx.run_id)
            raise Exception(
                "Failure inserting data in player pipeline, removing partial data..."
            )


if __name__ == "__main__":
    riot_api: RiotAPI = get_riot_api()

    loader = PlayerLoader()
    collector = PlayerCollector(riot_api)
    saver = PlayerSaver()

    orchestrator = PlayersOrchestrator(
        pipeline="players",
        loader=loader,
        collector=collector,
        saver=saver,
    )

    asyncio.run(orchestrator.run())
