# services/riot_api_client/match_v5.py
from __future__ import annotations

import asyncio
import itertools
import os
import time
from pathlib import Path
from collections import namedtuple, defaultdict, deque
from typing import AsyncGenerator, Iterable, List, Tuple, Set, Any, Deque, Dict, Iterator

from .base import RiotAPI
from config.constants import (
    Continents,
    ENDPOINTS,
    QUEUE_TYPE_TO_QUEUE_CODE,
)
from utils import storages
from models import MatchIds

SENTINEL: object = object()
MAX_PAGE_START  = 900
MAX_PAGE_COUNT  = 100

CrawlKey   = Tuple[str, str]          # (puuid, queueType)
CrawlState = namedtuple("CrawlState", ["puuid", "queueType", "continent", "start_time", "start"])
PendingItem = Tuple[str, "CrawlKey", "CrawlState"]  # (url, key, state)
# ───────────────────────── helper ───────────────────────── #

def build_buckets_by_continent(pending: Iterable[PendingItem]) -> Dict[str, Deque[PendingItem]]:
    buckets: Dict[str, Deque[PendingItem]] = defaultdict(deque)
    for item in pending:
        _, _, state = item
        buckets[state.continent].append(item)
    return buckets

def chunks_round_robin_from_buckets(
    buckets: Dict[str, Deque[PendingItem]],
    max_in_flight: int,
) -> Iterator[List[PendingItem]]:
    order = deque([c for c, dq in buckets.items() if dq])

    while order:
        batch: List[PendingItem] = []
        while len(batch) < max_in_flight and order:
            cont = order[0]
            dq = buckets[cont]
            if dq:
                batch.append(dq.popleft())
            if dq:
                order.rotate(-1)
            else:
                order.popleft()
        if batch:
            yield batch

# ─────────────────────── MatchV5 client ─────────────────── #

class MatchV5Ids(RiotAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ------------  STEP 1: load players  ------------------ #

    @staticmethod
    async def _players() -> AsyncGenerator[Tuple[str, str, str], None]:
        """
        Yield (puuid, queueType, continent) triples from the compressed league file.
        """
        in_path = (
            r"C:\Users\Jack\Documents\GitHub\fifth-time-lucky-api"
            r"\data\database\raw\league\league_players.csv.zst"
        )

        async for player in storages["league_v4"]["default"]["load"](path=in_path, indexes=[0, 1, 7]):
            yield player

    @staticmethod
    async def _collected_players() -> Tuple[Set[Tuple[Any, ...]], str]:
        league_dir = Path(
            r"C:\Users\Jack\Documents\GitHub\fifth-time-lucky-api"
            r"\data\database\raw\match"
        )

        files = list(league_dir.glob("collected_players_*.csv.zst"))
        if not files:
            print("No collected_players_*.csv.zst found.")
            return set(), ""
        if len(files) > 1:
            print("Error: multiple collected_players_*.csv.zst files found!")
            return set(), ""

        chosen = files[0]
        date_str = chosen.name[len("collected_players_") : -len(".csv.zst")]

        try:
            collected_epoch = int(date_str)
        except ValueError:
            print(f"Invalid timestamp in filename: {date_str}")
            return set(), 0

        collected: Set[Tuple[str, str]] = set()
        async for row in storages["league_v4"]["collected"]["load"](path=chosen):
            collected.add(tuple(row))

        return collected, collected_epoch

    # ------------  STEP 2: crawl match-ids  ---------------- #

    async def _fetch_with_meta(
        self, url: str, key: CrawlKey, state: CrawlState
    ) -> Tuple[CrawlKey, CrawlState, MatchIds]:
        try:
            data = await self.fetch_json(
                url=url,
                location=state.continent,
                scope="continent",
            )
        except Exception as e:
            raise RuntimeError((key, e))
        return key, state, data


    async def get_match_ids(
        self,
    ) -> AsyncGenerator[Tuple[MatchIds, CrawlKey], None]:
        puuid_endpoint = ENDPOINTS.match.by_puuid
        active: dict[CrawlKey, CrawlState] = {}


        collected_players, date_collected = await self._collected_players()
        async for puuid, queue_type, continent in self._players():
            key = (puuid, queue_type)
            active[key] = CrawlState(
                puuid=puuid,
                queueType=queue_type,
                continent=Continents(continent),
                start_time=date_collected if key in collected_players else 0,
                start=0,
            )

        while active:
            pending: List[PendingItem] = []
            for key, state in list(active.items()):
                queue_code = QUEUE_TYPE_TO_QUEUE_CODE.get(state.queueType)
                url = self._build_puuid_url(
                    url=puuid_endpoint,
                    puuid=state.puuid,
                    continent=state.continent,
                    queueType=queue_code,
                    startTime=state.start_time,
                    start=state.start,
                )
                pending.append((url, key, state))

            buckets = build_buckets_by_continent(pending)

            for batch in chunks_round_robin_from_buckets(buckets, self.max_in_flight):
                tasks = [asyncio.create_task(self._fetch_with_meta(url, key, state))
                        for url, key, state in batch]

                for fut in asyncio.as_completed(tasks):
                    try:
                        key, state, match_ids = await fut
                    except RuntimeError as err:
                        bad_key, exc = err.args[0]
                        print(f"Fetch failed for {bad_key}: {exc}")
                        active.pop(bad_key, None)
                        continue

                    yield match_ids, key

                    if (state.start != MAX_PAGE_START and len(match_ids) == MAX_PAGE_COUNT):
                        active[key] = state._replace(start=state.start + len(match_ids))
                    else:
                        active.pop(key, None)

    # ------------  helpers  -------------------------------- #

    def _build_puuid_url(
        self,
        url: str,
        puuid: str,
        continent: Continents,
        queueType: str,
        *,
        startTime: int = 0,
        start: int = 0,
    ) -> str:
        return url.format(
            continent=continent,
            puuid=puuid,
            startTime=startTime,
            endTime=int(time.time()),
            queue=queueType,
            type="ranked",
            start=start,
            api_key=self.get_api_key(),
        )

    # ------------ context-manager wiring ------------------- #

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()



'''
We need a seperate file to save the collected date (fileName) + puuid-queueType (tuples) associated with that. We can convert the list to a hashset after loading.
'''