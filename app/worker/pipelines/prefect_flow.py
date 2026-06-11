from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from collections.abc import Awaitable, Callable, Sequence

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


def _build_steps(
    riot_api: RiotAPI, *, matchdata_only: bool = False
) -> Sequence[PipelineStep]:
    if matchdata_only:
        return (_build_match_data_step(riot_api),)

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

    return (
        PipelineStep("players", players.run),
        PipelineStep("match_ids", match_ids.run),
        _build_match_data_step(riot_api),
    )


def _build_match_data_step(riot_api: RiotAPI) -> PipelineStep:
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
    return PipelineStep("match_data", match_data.run)


async def _run_cycle(steps: Sequence[PipelineStep]) -> None:
    for step in steps:
        raise_if_stop_requested(stage=f"pipeline:{step.name}:start")
        logger.info("Step start: %s", step.name)
        start = time.monotonic()
        await step.run()
        logger.info("Step done: %s (%.2fs)", step.name, time.monotonic() - start)


@flow(name="riot-pipeline")
async def riot_pipeline(matchdata_only: bool = False) -> None:
    """
    One Prefect flow run = one full pipeline cycle.
    matchdata_only skips upstream collection and drains the matchdata queue.
    Repetition is handled by Prefect Automation (run again on completion).
    """
    logger.info("Pipeline run start matchdata_only=%s", matchdata_only)
    start = time.monotonic()

    async with get_riot_api() as riot_api:
        steps = _build_steps(riot_api, matchdata_only=matchdata_only)
        await _run_cycle(steps)

    logger.info("Pipeline run success (%.2fs)", time.monotonic() - start)
