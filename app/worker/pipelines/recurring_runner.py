from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from app.core.config.constants.paths import (
    MATCH_DATA_MATCH_IDS_DIR,
    MATCH_DATA_NO_TIMELINE_PATH,
    MATCH_DATA_TIMELINE_PATH,
    MATCH_IDS_DATA_DIR,
    PLAYER_INFO,
    PLAYER_PUUIDS,
    PUUIDS_FOR_MATCH_IDS,
    PUUIDS_FOR_MATCH_IDS_CHECKPOINT,
)
from app.core.config.settings import settings
from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.worker.pipelines.matchdata_orchestrator import (
    MatchDataLoader,
    MatchDataNonTimelineCollector,
    MatchDataOrchestrator,
    MatchDataSaver,
    MatchDataTimelineCollector,
)
from app.worker.pipelines.matchids_orchestrator import (
    MatchIDCollector,
    MatchIDLoader,
    MatchIDOrchestrator,
    MatchIDSaver,
    read_players_jsonl_zst,
)
from app.worker.pipelines.players_orchestrator import (
    PlayerCollector,
    PlayerLoader,
    PlayerSaver,
    PlayersOrchestrator,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    run: Callable[[], Awaitable[None]]


def _install_signal_handlers(stop: asyncio.Event) -> None:
    def _stop() -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _stop())


async def _sleep_cancelable(stop: asyncio.Event, seconds: float) -> None:
    if seconds <= 0:
        return
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return


def _build_steps(riot_api: RiotAPI) -> Sequence[PipelineStep]:
    async def run_players() -> None:
        orchestrator = PlayersOrchestrator(
            loader=PlayerLoader(),
            collector=PlayerCollector(riot_api=riot_api),
            saver=PlayerSaver(),
            collected_player_puuids_path=PLAYER_PUUIDS,
            collected_players_path=PLAYER_INFO,
        )
        await orchestrator.run()

    async def run_match_ids() -> None:
        loader = MatchIDLoader(
            all_player_details_path=PLAYER_INFO,
            collected_puuids_path=PUUIDS_FOR_MATCH_IDS,
            collected_puuids_checkpoint_path=PUUIDS_FOR_MATCH_IDS_CHECKPOINT,
            read_players_jsonl_zst=read_players_jsonl_zst,
        )
        orchestrator = MatchIDOrchestrator(
            loader=loader,
            collector=MatchIDCollector(riot_api=riot_api),
            saver=MatchIDSaver(),
            collected_match_ids_dir=MATCH_IDS_DATA_DIR,
            collected_puuids_path=PUUIDS_FOR_MATCH_IDS,
            checkpoint_path=PUUIDS_FOR_MATCH_IDS_CHECKPOINT,
        )
        await orchestrator.run()

    async def run_match_data() -> None:
        loader = MatchDataLoader(match_ids_dir=MATCH_DATA_MATCH_IDS_DIR)

        non_timeline_collector = MatchDataNonTimelineCollector(riot_api=riot_api)
        timeline_collector = MatchDataTimelineCollector(riot_api=riot_api)

        saver = MatchDataSaver()

        orchestrator = MatchDataOrchestrator(
            loader=loader,
            non_timeline_collector=non_timeline_collector,
            timeline_collector=timeline_collector,
            saver=saver,
            out_matchids=MATCH_DATA_MATCH_IDS_DIR,
            out_non_timeline=MATCH_DATA_NO_TIMELINE_PATH,
            out_timeline=MATCH_DATA_TIMELINE_PATH,
        )
        await orchestrator.run()

    return (
        PipelineStep("players", run_players),
        PipelineStep("match_ids", run_match_ids),
        PipelineStep("match_data", run_match_data),
    )


async def _run_cycle(steps: Sequence[PipelineStep]) -> None:
    for step in steps:
        logger.info("Step start: %s", step.name)
        start = time.monotonic()
        await step.run()
        logger.info("Step done: %s (%.2fs)", step.name, time.monotonic() - start)


async def main() -> None:
    stop = asyncio.Event()
    _install_signal_handlers(stop)

    interval_seconds: float = float(
        getattr(settings, "pipeline_interval_seconds", 6 * 60 * 60)
    )
    backoff_seconds: float = 60.0
    backoff_cap: float = 15 * 60.0

    async with get_riot_api() as riot_api:
        while not stop.is_set():
            cycle_start = time.monotonic()
            steps = _build_steps(riot_api)

            try:
                logger.info("Pipeline cycle start")
                await _run_cycle(steps)
                logger.info("Pipeline cycle success")

                backoff_seconds = 60.0

                elapsed = time.monotonic() - cycle_start
                sleep_for = max(0.0, interval_seconds - elapsed)
                logger.info("Sleeping %.1fs", sleep_for)
                await _sleep_cancelable(stop, sleep_for)

            except Exception:
                logger.exception(
                    "Pipeline cycle failed; backing off %.1fs", backoff_seconds
                )
                await _sleep_cancelable(stop, backoff_seconds)
                backoff_seconds = min(backoff_cap, backoff_seconds * 2)


if __name__ == "__main__":
    asyncio.run(main())
