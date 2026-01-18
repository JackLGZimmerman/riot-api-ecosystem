from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, NamedTuple

from app.core.config.constants import (
    ENDPOINTS,
    QUEUE_TYPE_TO_QUEUE_CODE,
    REGION_TO_CONTINENT,
    Queues,
    Region,
)
from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.match_ids import (
    stream_match_ids,
)
from app.services.riot_api_client.utils import PlayerCrawlState
from app.worker.pipelines.orchestrator import (
    Collector,
    Loader,
    Orchestrator,
    Saver,
    SaveSpec,
)
from database.clickhouse.operations.matchids import (
    insert_matchids_stream_in_batches,
    insert_puuid_timestamp,
    insert_puuids_in_batches,
    load_matchid_puuid_timestamp,
    load_matchid_puuids,
)
from database.clickhouse.operations.players import load_players

MATCHID_BUFFER = 200_000


class PlayerKeyRow(NamedTuple):
    puuid: str
    queue_type: str
    region: str


def build_initial_player_states(
    players: list[PlayerKeyRow], collected_puuids, collected_puuids_ts, *, ts: int
) -> list[PlayerCrawlState]:
    end_time = ts

    template = str(ENDPOINTS["match"]["by_puuid"])

    states: list[PlayerCrawlState] = []

    for player in players:
        puuid = player.puuid
        queue_type = Queues(player.queue_type)
        continent = REGION_TO_CONTINENT[Region(player.region)]
        queue = QUEUE_TYPE_TO_QUEUE_CODE[queue_type]
        start_time = (
            collected_puuids_ts
            if puuid in collected_puuids and collected_puuids_ts > 0
            else 0
        )
        base_url = template.format(
            continent=continent,
            puuid=puuid,
            startTime=start_time,
            endTime=end_time,
            type="ranked",
            queue=queue,
            start="{start}",
            count=100,
        )

        states.append(
            PlayerCrawlState(
                puuid=puuid,
                queue_type=queue_type,
                continent=continent,
                start_time=start_time,
                next_page_start=0,
                end_time=end_time,
                base_url=base_url,
            )
        )

    return states


@dataclass
class MatchIDCollectorState:
    initial_states: list[PlayerCrawlState]
    collected_puuids: list[str]
    full_player_puuids: list[str]
    ts: int


class MatchIDOrchestrator(Orchestrator):
    def __init__(
        self,
        loader: Loader,
        collector: Collector,
        saver: Saver,
    ):
        super().__init__(loader, collector, saver)
        self.ts = int(time.time())

    async def _dedupe_async(
        self, batches: AsyncIterator[list[str]]
    ) -> AsyncIterator[list[str]]:
        seen: set[str] = set()
        async for batch in batches:
            if not batch:
                continue

            out: list[str] = []
            for mid in batch:
                if mid not in seen:
                    seen.add(mid)
                    out.append(mid)

            if out:
                yield out

    async def run(self) -> None:
        state: MatchIDCollectorState = self.loader.load(ts=self.ts)

        match_ids_stream: AsyncIterator[list[str]] = self.collector.collect(state)
        match_ids_stream = self._dedupe_async(match_ids_stream)

        async def save_matchid_puuids() -> None:
            await asyncio.to_thread(
                insert_puuids_in_batches,
                state.full_player_puuids,
            )

        async def save_matchids() -> None:
            await insert_matchids_stream_in_batches(
                match_ids_stream,
                buffer_size=MATCHID_BUFFER,
            )

        async def save_matchid_puuids_timestamp() -> None:
            await asyncio.to_thread(insert_puuid_timestamp, state.ts)

        await self.saver.save(
            SaveSpec(save=save_matchid_puuids),
            SaveSpec(save=save_matchids),
            SaveSpec(save=save_matchid_puuids_timestamp),
        )


class MatchIDLoader:
    def load(self, ts: int) -> MatchIDCollectorState:
        players: list[PlayerKeyRow] = load_players()
        collected_puuids: list[str] = load_matchid_puuids()
        collected_puuid_ts: int = load_matchid_puuid_timestamp()

        initial_states = build_initial_player_states(
            players,
            collected_puuids,
            collected_puuid_ts,
            ts=ts,
        )

        return MatchIDCollectorState(
            initial_states=initial_states,
            collected_puuids=collected_puuids,
            full_player_puuids=[p.puuid for p in players],
            ts=ts,
        )


class MatchIDCollector(Collector):
    def __init__(self, riot_api: RiotAPI):
        self.riot_api = riot_api

    async def collect(self, state: MatchIDCollectorState) -> AsyncIterator[list[str]]:
        async for match_ids in stream_match_ids(
            self.riot_api,
            initial_states=state.initial_states,
            ts=state.ts,
        ):
            if match_ids:
                yield match_ids


class MatchIDSaver:
    async def save(self, *specs: SaveSpec) -> None:
        for spec in specs:
            await spec.save()


if __name__ == "__main__":
    riot_api: RiotAPI = get_riot_api()
    loader = MatchIDLoader()
    collector = MatchIDCollector(riot_api)
    saver = MatchIDSaver()

    orchestrator = MatchIDOrchestrator(loader=loader, collector=collector, saver=saver)

    asyncio.run(orchestrator.run())
