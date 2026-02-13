from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from prefect import flow

from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.parsers.non_timeline import (
    MatchDataNonTimelineParsingOrchestrator,
)
from app.services.riot_api_client.parsers.timeline import (
    MatchDataTimelineParsingOrchestrator,
)
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


def _build_steps(riot_api: RiotAPI) -> Sequence[PipelineStep]:
    async def run_players() -> None:
        orchestrator = PlayersOrchestrator(
            pipeline="players",
            loader=PlayerLoader(),
            collector=PlayerCollector(riot_api=riot_api),
            saver=PlayerSaver(),
        )
        await orchestrator.run()

    async def run_match_ids() -> None:
        orchestrator = MatchIDOrchestrator(
            pipeline="match_ids",
            loader=MatchIDLoader(),
            collector=MatchIDCollector(riot_api=riot_api),
            saver=MatchIDSaver(),
        )
        await orchestrator.run()

    async def run_match_data() -> None:
        orchestrator = MatchDataOrchestrator(
            pipeline="match_ids",
            loader=MatchDataLoader(),
            non_timeline_collector=MatchDataNonTimelineCollector(riot_api=riot_api),
            timeline_collector=MatchDataTimelineCollector(riot_api=riot_api),
            saver=MatchDataSaver(
                non_timeline_parser=MatchDataNonTimelineParsingOrchestrator(),
                timeline_parser=MatchDataTimelineParsingOrchestrator(),
            ),
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


@flow(name="riot-pipeline")
async def riot_pipeline() -> None:
    """
    One Prefect flow run = one full pipeline cycle.
    Repetition is handled by Prefect Automation (run again on completion).
    """
    logger.info("Pipeline run start")
    start = time.monotonic()

    async with get_riot_api() as riot_api:
        steps = _build_steps(riot_api)
        await _run_cycle(steps)

    logger.info("Pipeline run success (%.2fs)", time.monotonic() - start)
