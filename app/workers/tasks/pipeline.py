# app/workers/tasks/pipeline.py

import asyncio
import time
from datetime import datetime, timezone
from logging import Logger

from celery import Task
from celery.utils.log import get_task_logger

from app.models import (
    BasicBoundsConfig,
    EliteBoundsConfig,
    parse_basic_bounds,
    parse_elite_bounds,
)
from app.services.pipelines.players import run_player_collection_pipeline
from app.services.riot_api_client.base import get_riot_api
from app.workers.app import celery_app

logger: Logger = get_task_logger(__name__)


@celery_app.task(name="demo.long_running", bind=True)
def long_running_task(self: Task, n: int) -> int:
    """
    Very simple demo task that logs progress and sleeps.
    Runs in the background on the Celery worker.
    """

    logger.info(
        "Starting long_running_task | n=%s | task_id=%s | worker=%s",
        n,
        self.request.id,
        self.request.hostname,
    )

    for i in range(1, n + 1):
        logger.info("Progress %s/%s (task_id=%s)", i, n, self.request.id)
        time.sleep(1)

    logger.info(
        "Finished long_running_task | n=%s | task_id=%s | worker=%s",
        n,
        self.request.id,
        self.request.hostname,
    )

    return n


# -----------------------------------------------------------
# PIPELINE TASK WITH FULL LOGGING
# -----------------------------------------------------------


@celery_app.task(name="pipelines.player_collection", bind=True)
def player_collection_task(self: Task) -> str:
    """
    Celery task that runs the league snapshot pipeline
    (elite + sub-elite, dedup, export).
    Runs in the background on the Celery worker.
    """

    start_ts = datetime.now(timezone.utc)
    logger.info(
        "Starting player_collection_task | task_id=%s | worker=%s | timestamp=%s",
        self.request.id,
        self.request.hostname,
        start_ts.isoformat(),
    )

    elite_bounds: EliteBoundsConfig = parse_elite_bounds(
        {
            "RANKED_SOLO_5x5": {
                "collect": True,
                "upper": None,
                "lower": None,
            },
            "RANKED_FLEX_SR": {
                "collect": True,
                "upper": None,
                "lower": None,
            },
        }
    )

    sub_elite_bounds: BasicBoundsConfig = parse_basic_bounds(
        {
            "RANKED_SOLO_5x5": {
                "collect": True,
                "upper_tier": None,
                "upper_division": None,
                "lower_tier": None,
                "lower_division": None,
            },
            "RANKED_FLEX_SR": {
                "collect": True,
                "upper_tier": None,
                "upper_division": None,
                "lower_tier": None,
                "lower_division": None,
            },
        }
    )

    logger.info(
        "Bounds parsed successfully | task_id=%s | elite_keys=%s | sub_elite_keys=%s",
        self.request.id,
        list(elite_bounds.keys()),
        list(sub_elite_bounds.keys()),
    )

    async def _main() -> None:
        async with get_riot_api() as riot_api:
            await run_player_collection_pipeline(
                elite_bounds=elite_bounds,
                sub_elite_bounds=sub_elite_bounds,
                riot_api=riot_api,
            )

    try:
        asyncio.run(_main())

        end_ts = datetime.now(timezone.utc)
        logger.info(
            "Finished player_collection_task | task_id=%s | runtime=%.2fs | worker=%s",
            self.request.id,
            (end_ts - start_ts).total_seconds(),
            self.request.hostname,
        )
        return "ok"

    except Exception as e:
        logger.exception(
            "player_collection_task failed | task_id=%s | worker=%s | error=%s",
            self.request.id,
            self.request.hostname,
            str(e),
        )
        raise
