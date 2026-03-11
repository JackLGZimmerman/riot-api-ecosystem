from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from prefect import flow

from app.core.logging import setup_logging_config
from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.parsers.non_timeline import (
    MatchDataNonTimelineParsingOrchestrator,
)
from app.services.riot_api_client.parsers.timeline import (
    MatchDataTimelineParsingOrchestrator,
)
from app.worker.pipelines.matchdata_orchestrator import (
    MatchDataLoader,
    MatchDataOrchestrator,
    MatchDataSaver,
    MatchDataStreamCollector,
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
from app.worker.pipelines.stop_flag import raise_if_stop_requested

setup_logging_config()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    run: Callable[[], Awaitable[None]]


def _build_steps(riot_api: RiotAPI) -> Sequence[PipelineStep]:
    players = PlayersOrchestrator(
        pipeline="players",
        loader=PlayerLoader(),
        collector=PlayerCollector(riot_api=riot_api),
        saver=PlayerSaver(),
    )
    match_ids = MatchIDOrchestrator(
        pipeline="match_ids",
        loader=MatchIDLoader(),
        collector=MatchIDCollector(riot_api=riot_api),
        saver=MatchIDSaver(),
    )
    match_data = MatchDataOrchestrator(
        pipeline="match_data",
        loader=MatchDataLoader(),
        non_timeline_collector=MatchDataStreamCollector(
            riot_api=riot_api,
            stream="non_timeline",
        ),
        timeline_collector=MatchDataStreamCollector(
            riot_api=riot_api,
            stream="timeline",
        ),
        saver=MatchDataSaver(
            non_timeline_parser=MatchDataNonTimelineParsingOrchestrator(),
            timeline_parser=MatchDataTimelineParsingOrchestrator(),
        ),
    )

    return (
        PipelineStep("players", players.run),
        PipelineStep("match_ids", match_ids.run),
        PipelineStep("match_data", match_data.run),
    )


async def _run_cycle(steps: Sequence[PipelineStep]) -> None:
    for step in steps:
        raise_if_stop_requested(stage=f"pipeline:{step.name}:start")
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
