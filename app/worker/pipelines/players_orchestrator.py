from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator
from uuid import UUID, uuid4

from app.models import MinifiedLeagueEntryDTO
from app.models.riot.league import BASIC_BOUNDS, ELITE_BOUNDS
from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.elite_players import stream_elite_players
from app.services.riot_api_client.subelite_players import stream_sub_elite_players
from app.worker.pipelines.orchestrator import (
    Collector,
    Loader,
    OrchestrationContext,
    Orchestrator,
    Saver,
)
from app.worker.pipelines.recovery_utils import run_sync_with_retry
from app.worker.pipelines.stop_flag import raise_if_stop_requested
from database.clickhouse.operations.players import (
    delete_failed_players_snapshot_ts,
    delete_old_players_snapshot_ts,
    delete_partial_players_run,
    insert_players_stream_in_batches,
    upsert_players_snapshot_ts,
)

logger = logging.getLogger(__name__)


class PlayersOrchestrator(Orchestrator):
    loader: PlayerLoader
    collector: PlayerCollector
    saver: PlayerSaver

    def __init__(
        self,
        pipeline: str,
        loader: PlayerLoader,
        collector: PlayerCollector,
        saver: PlayerSaver,
    ):
        super().__init__(pipeline=pipeline, loader=loader, collector=collector, saver=saver)

    async def run(self) -> None:
        # Players intentionally does not use resumable state tracking.
        ctx = OrchestrationContext(
            ts=int(time.time()),
            run_id=uuid4(),
            pipeline=self.pipeline,
        )
        state = self.loader.load(ctx)

        logger.info("Players run start run_id=%s", ctx.run_id)
        await self.saver.save(self.collector.collect(state, ctx), state, ctx)
        await self.saver.finalize_cycle(cycle_ts=ctx.ts, run_id=ctx.run_id)
        logger.info("Players run complete run_id=%s", ctx.run_id)


class PlayerLoader(Loader):
    def load(self, ctx: OrchestrationContext) -> None:
        _ = ctx
        return None


class PlayerCollector(Collector):
    def __init__(self, riot_api: RiotAPI):
        self.riot_api = riot_api

    async def collect(
        self,
        state: None,
        ctx: OrchestrationContext,
    ) -> AsyncIterator[MinifiedLeagueEntryDTO]:
        _ = state
        _ = ctx
        raise_if_stop_requested(stage="players:start")
        async for player in stream_elite_players(
            ELITE_BOUNDS,
            riot_api=self.riot_api,
        ):
            raise_if_stop_requested(stage="players:elite")
            yield player

        raise_if_stop_requested(stage="players:subelite-start")
        async for player in stream_sub_elite_players(
            BASIC_BOUNDS,
            riot_api=self.riot_api,
        ):
            raise_if_stop_requested(stage="players:subelite")
            yield player


class PlayerSaver(Saver):
    async def save(
        self,
        items: AsyncIterator[MinifiedLeagueEntryDTO],
        state: None,
        ctx: OrchestrationContext,
    ) -> None:
        _ = state
        try:
            await insert_players_stream_in_batches(
                items,
                ts=ctx.ts,
                run_id=ctx.run_id,
            )
        except Exception:
            await run_sync_with_retry(
                logger=logger,
                component="Players",
                op_name="delete_partial_players_run",
                func=delete_partial_players_run,
                args=(ctx.run_id,),
            )
            logger.exception("Players save failed run_id=%s", ctx.run_id)
            raise

    async def finalize_cycle(self, *, cycle_ts: int, run_id: UUID) -> None:
        try:
            await run_sync_with_retry(
                logger=logger,
                component="Players",
                op_name="upsert_players_snapshot_ts",
                func=upsert_players_snapshot_ts,
                args=(cycle_ts, run_id),
            )
        except Exception:
            await run_sync_with_retry(
                logger=logger,
                component="Players",
                op_name="delete_failed_players_snapshot_ts",
                func=delete_failed_players_snapshot_ts,
                args=(run_id,),
            )
            raise

        await run_sync_with_retry(
            logger=logger,
            component="Players",
            op_name="delete_old_players_snapshot_ts",
            func=delete_old_players_snapshot_ts,
            args=(run_id,),
        )


if __name__ == "__main__":
    async def _main() -> None:
        async with get_riot_api() as riot_api:
            orchestrator = PlayersOrchestrator(
                pipeline="players",
                loader=PlayerLoader(),
                collector=PlayerCollector(riot_api),
                saver=PlayerSaver(),
            )

            await orchestrator.run()

    asyncio.run(_main())
