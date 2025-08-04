from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import itertools
import os
import pstats
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import psutil
from pydantic import ValidationError

from config.constants import (
    DIVISION_MAPPING,
    ELITE_TIER_MAPPING,
    ENDPOINTS,
    REGION_TO_CONTINENT,
    Division,
    EliteTier,
    Queue,
    Region,
    Tier,
    TIER_MAPPING,
)

from config import settings
from utils import league_v4
from models.riot.league import (
    LeagueEntryDTO,
    LeagueListDTO,
    MinifiedLeagueEntryDTO,
)
from services import RiotAPI


SENTINEL: object = object()


def chunked(iterable, n: int):
    """Yield lists of *n* items from *iterable*."""
    it = iter(iterable)
    while (batch := list(itertools.islice(it, n))):
        yield batch


PageKey = Tuple[Region, Queue, Tier, Division]
PageBound = Dict[PageKey, int]
EliteQueueBound = Dict[Queue, EliteTier | None]
SubEliteQueueBound = Dict[Queue, Tuple[Tier, Division] | None]


@dataclass(slots=True)
class PipelineConfig:
    max_in_flight: int = 64
    bulk_batch_size: int = 500
    queue_maxsize: int = 1_000
    monitor_interval: float = 1.0
    out_path: Path = Path("data/database/raw/league/league_players.csv.zst")


class LeagueV4(RiotAPI):
    """API client streaming elite/sub-elite league pages."""

    def __init__(self, *, max_in_flight: int = 32):
        super().__init__()
        self.max_in_flight = max_in_flight
        self.league_page_upper_bound: int = settings.league_page_upper_bound,


    @staticmethod
    def _listDTO_to_minified_entry(
        league: LeagueListDTO, region: Region
    ) -> List[MinifiedLeagueEntryDTO]:
        return [
            MinifiedLeagueEntryDTO(
                puuid=entry.puuid,
                queueType=league.queue,
                tier=league.tier,
                rank=entry.rank,
                wins=entry.wins,
                losses=entry.losses,
                region=region,
                continent=REGION_TO_CONTINENT[region],
            )
            for entry in league.entries
            if entry and entry.puuid
        ]

    @staticmethod
    def _entryDTO_to_minified_entry(
        league: LeagueEntryDTO, region: Region
    ) -> MinifiedLeagueEntryDTO:
        return MinifiedLeagueEntryDTO(
            puuid=league.puuid,
            queueType=league.queueType,
            tier=league.tier,
            rank=league.rank,
            wins=league.wins,
            losses=league.losses,
            region=region,
            continent=REGION_TO_CONTINENT[region],
        )

    async def _fetch_and_tag(self, url: str, location: Region):
        resp = await self.fetch_json(url=url, location=location, scope="region")
        assert isinstance(resp, Mapping)
        return location, resp

    async def stream_elite_players(self, queue_bounds: EliteQueueBound):
        urls: List[Tuple[str, Region]] = []
        for queue, base_tier in queue_bounds.items():
            if base_tier is None:
                continue
            for tier in ELITE_TIER_MAPPING[base_tier]:
                template = getattr(ENDPOINTS.league, tier.value.lower())
                for region in Region:
                    urls.append((template.format(region=region, queue=queue, api_key=self._api_key), region))

        for batch in chunked(urls, self.max_in_flight):
            tasks = [asyncio.create_task(self._fetch_and_tag(url, region)) for url, region in batch]
            try:
                for fut in asyncio.as_completed(tasks):
                    try:
                        region, resp = await fut
                    except Exception as e:
                        print("fetch task failed:", e)
                        continue
                    if not resp.get("entries"):
                        continue
                    try:
                        dto = LeagueListDTO(**resp)
                    except ValidationError as err:
                        print("invalid LeagueListDTO —", err)
                        continue
                    for entry in self._listDTO_to_minified_entry(dto, region):
                        yield entry
            finally:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


    async def _page_task(self, template, r, q, t, d, pg):
        url = template.format(region=r, queue=q, tier=t, division=d, page=pg, api_key=self._api_key)
        data = await self.fetch_json(url=url, location=r, scope="region")
        if not isinstance(data, list):
            raise TypeError("Expected list JSON for sub-elite endpoints")
        return (r, q, t, d, pg), data

    async def stream_sub_elite_players(self, queue_bounds: SubEliteQueueBound):
        page_bounds = await self._discover_page_bounds(queue_bounds)
        template = str(ENDPOINTS.league.by_queue_tier_division)

        for (region, queue, tier, division), last_page in page_bounds.items():
            for batch in chunked(range(1, last_page + 1), self.max_in_flight):
                tasks = [
                    asyncio.create_task(
                        self._page_task(template, region, queue, tier, division, pg)
                    )
                    for pg in batch
                ]
                try:
                    for fut in asyncio.as_completed(tasks):
                        try:
                            _, records = await fut
                        except Exception as e:
                            print("page task failed:", e)
                            continue
                        for raw in records:
                            try:
                                dto = LeagueEntryDTO(**raw)
                            except ValidationError:
                                continue
                            yield self._entryDTO_to_minified_entry(dto, region)
                finally:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    async def _discover_page_bounds(self, queue_bounds: SubEliteQueueBound) -> PageBound:
        template = str(ENDPOINTS.league.by_queue_tier_division)

        async def probe(r, q, t, d):
            low, high = 1, self.league_page_upper_bound + 1
            while low + 1 < high:
                mid = (low + high) // 2
                url = template.format(region=r, queue=q, tier=t, division=d, page=mid, api_key=self._api_key)
                data = await self.fetch_json(url=url, location=r, scope="region")
                low, high = (mid, high) if data else (low, mid)
            return (r, q, t, d), low

        tasks = []
        for region in Region:
            for queue, opt in queue_bounds.items():
                if opt is None:
                    continue
                base_tier, base_div = opt
                for tier in TIER_MAPPING[base_tier]:
                    for division in DIVISION_MAPPING[base_div]:
                        tasks.append(asyncio.create_task(probe(region, queue, tier, division)))

        results = await asyncio.gather(*tasks)
        return dict(results)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

async def enqueue(gen, queue: asyncio.Queue):
    try:
        async for entry in gen:
            await queue.put(entry)
    finally:
        await queue.put(SENTINEL)


async def consumer_loop(queue: asyncio.Queue, cfg: PipelineConfig):
    buffer: List[List[Any]] = []
    done_seen = 0
    producers = 2

    try:
        while True:
            item = await queue.get()
            if item is SENTINEL:
                done_seen += 1
                if done_seen == producers:
                    break
                continue

            buffer.append(item)
            if len(buffer) >= cfg.bulk_batch_size:
                league_v4["save"](cfg.out_path, buffer)
                buffer.clear()

        if buffer:
            league_v4["save"](cfg.out_path, buffer)
    finally:
        print("Consumer loop finished - data flushed to", cfg.out_path)


async def monitor_resources(stop: asyncio.Event, interval: float):
    proc = psutil.Process(os.getpid())
    while not stop.is_set():
        rss = proc.memory_info().rss / (1024 * 1024)
        cpu = proc.cpu_percent(None)
        print(f"[RES] CPU%: {cpu:5.1f} | RSS: {rss:.2f} MB")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


async def main(profile: bool = False):
    cfg = PipelineConfig()

    stop_evt = asyncio.Event()
    monitor = asyncio.create_task(monitor_resources(stop_evt, cfg.monitor_interval))

    q: asyncio.Queue = asyncio.Queue(maxsize=cfg.queue_maxsize)

    async with LeagueV4(max_in_flight=cfg.max_in_flight) as api:
        prod1 = asyncio.create_task(
            enqueue(
                api.stream_elite_players({
                    Queue.RANKED_SOLO_5x5: EliteTier.MASTER,
                    Queue.RANKED_FLEX_SR: EliteTier.CHALLENGER,
                }),
                q,
            )
        )
        prod2 = asyncio.create_task(
            enqueue(
                api.stream_sub_elite_players({
                    Queue.RANKED_SOLO_5x5: (Tier.DIAMOND, Division.I),
                    Queue.RANKED_FLEX_SR: None,
                }),
                q,
            )
        )
        cons = asyncio.create_task(consumer_loop(q, cfg))

        if profile:
            pr = cProfile.Profile(); pr.enable()

        await prod1; 
        await prod2; 
        await cons

        stop_evt.set(); 
        await monitor

        if profile:
            pr.disable(); s = io.StringIO(); p = pstats.Stats(pr, stream=s).sort_stats("cumtime"); p.print_stats(10)
            print("=== PROFILE TOP 10 ===\n", s.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Run with cProfile")
    args = parser.parse_args()

    asyncio.run(main(profile=args.profile))
