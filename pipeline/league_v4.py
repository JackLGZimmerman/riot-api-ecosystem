# pipelines/league_pipeline.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

from services.riot_api_client.league_v4 import LeagueV4
from config.constants import Queue, Tier, Division, EliteTier
from utils.async_pipeline import enqueue, consumer_loop
from utils.file_storage import league_v4_save

SENTINEL = object()


@dataclass(slots=True)
class PipelineConfig:
    max_in_flight: int = 64
    bulk_batch_size: int = 500
    queue_maxsize: int = 1000
    monitor_interval: float = 1.0
    out_path: Path = Path("data/database/raw/league/league_players.csv.zst")


async def monitor_resources(stop: asyncio.Event, interval: float) -> None:
    import psutil, os

    proc = psutil.Process(os.getpid())
    while not stop.is_set():
        rss = proc.memory_info().rss / (1024 * 1024)
        cpu = proc.cpu_percent(None)
        print(f"[RES] CPU%: {cpu:5.1f} | RSS: {rss:.2f} MB")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


async def run_league_pipeline(profile: bool = False) -> None:
    cfg = PipelineConfig()
    stop_evt = asyncio.Event()
    monitor_task = asyncio.create_task(monitor_resources(stop_evt, cfg.monitor_interval))

    q: asyncio.Queue = asyncio.Queue(maxsize=cfg.queue_maxsize)

    try:
        async with LeagueV4(max_in_flight=cfg.max_in_flight) as api:
            prod_elite = asyncio.create_task(
                enqueue(
                    api.stream_elite_players(
                        {
                            Queue.RANKED_SOLO_5x5: EliteTier.MASTER,
                            Queue.RANKED_FLEX_SR: EliteTier.CHALLENGER,
                        }
                    ),
                    q,
                    sentinel=SENTINEL,
                )
            )
            prod_sub_elite = asyncio.create_task(
                enqueue(
                    api.stream_sub_elite_players(
                        {
                            Queue.RANKED_SOLO_5x5: (Tier.DIAMOND, Division.I),
                            Queue.RANKED_FLEX_SR: None,
                        }
                    ),
                    q,
                    sentinel=SENTINEL,
                )
            )
            consumer = asyncio.create_task(
                consumer_loop(
                    q=q,
                    save_func=league_v4_save,
                    out_path=str(cfg.out_path),
                    batch_size=cfg.bulk_batch_size,
                    num_producers=2,
                    sentinel=SENTINEL,
                )
            )

            await prod_elite
            await prod_sub_elite
            await consumer
    finally:
        stop_evt.set()
        await monitor_task
