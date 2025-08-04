from __future__ import annotations

from typing import List, Tuple, Iterator, Dict, AsyncGenerator
import os
from pathlib import Path
import random
import time
import psutil
from datetime import datetime, timezone
import memory_profiler
import itertools
import asyncio
from collections import namedtuple
from .base import RiotAPI
from dataclasses import dataclass, field
from config.constants import Continent, ENDPOINTS, Queue, QUEUE_TYPE_TO_QUEUE_CODE
from config import db_settings, settings
from utils import (
    init_db,
    close_db,
    league_v4,
    match_v5
)

from models import MatchIds

SENTINEL: object = object()
MAX_PAGE_START = 900
MAX_PAGE_COUNT = 100
CrawlKey = Tuple[str, str]  # (puuid, queueType)
CrawlState = namedtuple("CrawlState", ["puuid", "queue_type", "continent", "start_time", "start"])

@dataclass(slots=True)
class PipelineConfig:
    max_in_flight: int = 15_000
    max_batch_size: int = 20_000
    max_queue_size: int = 40_000
    monitor_interval: float = 1.0
    current_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    out_path: Path = field(init=False)

    def __post_init__(self):
        date_str = self.current_date.strftime("%Y%m%d") 
        self.out_path = Path(
            r"C:\Users\Jack\Documents\GitHub\fifth-time-lucky-api"
            r"\data\database\raw\league"
        ) / f"matchids-{date_str}.csv.zst"

def chunked(iterable, n: int):
    it = iter(iterable)
    while (batch := list(itertools.islice(it, n))):
        yield batch

class MatchV5(RiotAPI):
    def __init__(self, *, max_in_flight: int = 32):
        super().__init__()
        self.max_in_flight = max_in_flight

    @staticmethod
    async def get_players(dev: bool = False) -> AsyncGenerator[List[List[str]], None]:
        in_path = (
            r"C:\Users\Jack\Documents\GitHub\fifth-time-lucky-api"
            r"\data\database\raw\league"
            r"\league_players.csv.zst"
        )
        try:
            for chunk in league_v4["load"](path=in_path, indexes=[0, 1, 7], chunk_size=None):
                for row in chunk:
                    if len(row) < 3:
                        continue  # defensive
                    yield (row[0], row[1], row[2])
        except Exception as e:
            raise ValueError(f"Unable to load player data from {in_path}: {e}") from e


    async def get_match_ids(
        self,         
        dev: bool = False
    ) -> AsyncGenerator[MatchIds, None]:
        match_v5_puuid_endpoint = ENDPOINTS.match.by_puuid

        active: dict[CrawlKey, CrawlState] = {}

        async for player in self.get_players(dev):
            puuid, queueType, continent = player
            key: CrawlKey = (puuid, queueType)
            existing_start_time = 0
            active[key] = CrawlState(
                puuid=puuid,
                queue_type=queueType,
                continent=Continent(continent),
                start_time=existing_start_time,
                start=0,
            )

        while active:
            pending = []
            for key, state in list(active.items()):
                queue_code = QUEUE_TYPE_TO_QUEUE_CODE.get(state.queue_type)
                if queue_code is None:
                    active.pop(key, None)
                    continue

                url = self._build_puuid_url(
                    url=match_v5_puuid_endpoint,
                    puuid=state.puuid,
                    continent=state.continent,
                    queueType=queue_code,
                    startTime=state.start_time,
                    start=state.start,
                )
                pending.append((url, key, state))

            random.shuffle(pending)
            for batch in chunked(pending, self.max_in_flight):
                task_map = {}
                async with asyncio.TaskGroup() as tg:
                    for url, key, state in batch:
                        task = tg.create_task(
                            self.fetch_json(url=url, location=state.continent, scope="continent")
                        )
                        task_map[task] = (key, state)

                for task, (key, state) in task_map.items():
                    try:
                        data = task.result()
                    except Exception as e:
                        print(f"Fetch failed for {key}: {e}")
                        active.pop(key, None)
                        continue

                    match_ids: MatchIds = data

                    yield match_ids

                    if state.start != MAX_PAGE_START and len(match_ids) == MAX_PAGE_COUNT:
                        updated = state._replace(start=state.start + len(match_ids))
                        active[key] = updated
                    else:
                        active.pop(key, None)
    
    def _build_puuid_url(
            self, 
            url: str,
            puuid: str, 
            continent: Continent, 
            queueType: Queue, 
            startTime: int = 0, 
            start: int = 0) -> str:
        return url.format(
            continent=continent,
            puuid=puuid,
            startTime=startTime,
            endTime=int(time.time()),
            queue=queueType,
            type="ranked",
            start=start,
            api_key=self._api_key
        )   
    

    @staticmethod
    async def get_match_ids() -> AsyncGenerator[str, None]:
        parent_dir = r"C:\Users\Jack\Documents\GitHub\fifth-time-lucky-api\data\database\raw\league"
        
        try:
            files = os.listdir(parent_dir)
            match_id_file = [file for file in files if file.startswith('matchids-')]
            async for match_id in match_v5["load"](match_id_file):
                pass
        except Exception as e:
            pass

    async def get_match_data(self):
        pass
    
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()


async def monitor_resources(stop: asyncio.Event, interval: float):
    proc = psutil.Process(os.getpid())
    while not stop.is_set():
        rss = proc.memory_info().rss
        cpu = proc.cpu_percent(interval=None)
        print(f"[RES] CPU%: {cpu:5.1f} | RSS: {rss/1024/1024:.2f} MB")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


async def enqueue(
    gen: AsyncGenerator[MatchIds, None],
    queue: asyncio.Queue,
    sentinel: object = SENTINEL,
):
    try:
        async for entry in gen:
            await queue.put(entry)
    finally:
        await queue.put(sentinel)


async def consumer_loop(q: asyncio.Queue, batch_size: int = 100, path: str | None = None):
    buffer: List[MatchIds] = []
    
    while True:
        try:
            item = await q.get()
            if item is SENTINEL:
                if buffer:
                    match_v5["save"](path, buffer)
                break
            
            buffer.append(item)

            if len(buffer) >= batch_size:
                match_v5["save"](
                    path,
                    buffer,
                )
                buffer.clear()
        except:
            print("Error during the consumer loop for match_v5")



async def _demo(profile: bool = False, dev: bool = False):
    await init_db()
    cfs = PipelineConfig()


    q: asyncio.Queue = asyncio.Queue(maxsize=cfs.max_queue_size)
    stop = asyncio.Event()


    mon = asyncio.create_task(monitor_resources(stop=stop, interval=cfs.monitor_interval))


    async with MatchV5(max_in_flight=cfs.max_in_flight) as api:
        
        prod1 = asyncio.create_task(
            enqueue(
                api.get_match_ids(dev=True),
                q
            )
        )
        cons = asyncio.create_task(
            consumer_loop(
                q=q,
                batch_size=cfs.max_batch_size,
                path=cfs.out_path
            )
        )

        await prod1
        await cons
        stop.set()
        await mon

    await close_db()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    asyncio.run(_demo(profile=args.profile, dev=args.dev))
