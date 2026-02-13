from __future__ import annotations

import asyncio
from uuid import uuid4

from dataclasses import dataclass
from typing import AsyncIterator
import time
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
    OrchestrationContext,
    Saver,
)
from database.clickhouse.operations.matchids import (
    load_matchid_puuid_ts,
    load_matchid_puuids,
    insert_matchids_stream_in_batches,
    upsert_puuid_timestamp,
    insert_puuids_in_batches,
    delete_failed_puuid_timestamp,
    delete_old_puuid_timestamps,
    delete_matchid_puuids,
    delete_matchids,
)
from database.clickhouse.operations.players import PlayerKeyRow, load_players

MATCHID_BUFFER = 200_000

def build_initial_player_states(
    players: list[PlayerKeyRow],
    collected_puuids: list[str],
    collected_puuids_ts: int,
    *,
    ts: int,
) -> list[PlayerCrawlState]:
    template = str(ENDPOINTS["match"]["by_puuid"])
    states: list[PlayerCrawlState] = []

    for player in players:
        puuid = player.puuid
        queue_type = Queues(player.queue_type)
        continent = REGION_TO_CONTINENT[Region(player.region)]
        queue = QUEUE_TYPE_TO_QUEUE_CODE[queue_type]

        start_time = (
            collected_puuids_ts
            if (puuid in collected_puuids and collected_puuids_ts > 0)
            else 0
        )

        base_url = template.format(
            continent=continent,
            puuid=puuid,
            startTime=start_time,
            endTime=ts,
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
                next_page_start=0,
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
        pipeline: str,
        loader: Loader,
        collector: Collector,
        saver: Saver,
    ):
        super().__init__(pipeline=pipeline, loader=loader, collector=collector, saver=saver)

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
        ctx = OrchestrationContext(
            ts=int(time.time()), run_id=uuid4(), pipeline=self.pipeline
        )
        state: MatchIDCollectorState = self.loader.load(ctx)

        match_ids_stream: AsyncIterator[list[str]] = self.collector.collect(state, ctx)
        match_ids_stream = self._dedupe_async(match_ids_stream)

        await self.saver.save(match_ids_stream, state, ctx)


class MatchIDLoader:
    def load(self, ctx: OrchestrationContext) -> MatchIDCollectorState:
        players: list[PlayerKeyRow] = load_players()
        collected_puuids: list[str] = load_matchid_puuids()
        collected_puuid_ts: int = load_matchid_puuid_ts()

        initial_states = build_initial_player_states(
            players,
            collected_puuids,
            collected_puuid_ts,
            ts=ctx.ts,
        )

        return MatchIDCollectorState(
            initial_states=initial_states,
            collected_puuids=collected_puuids,
            full_player_puuids=[p.puuid for p in players],
            ts=ctx.ts,
        )


class MatchIDCollector(Collector):
    def __init__(self, riot_api: RiotAPI):
        self.riot_api = riot_api

    async def collect(
        self, state: MatchIDCollectorState, ctx: OrchestrationContext
    ) -> AsyncIterator[list[str]]:
        async for match_ids in stream_match_ids(
            self.riot_api,
            initial_states=state.initial_states,
        ):
            if match_ids:
                yield match_ids


class MatchIDSaver:
    async def save(
        self,
        items: AsyncIterator[list[str]],
        state: MatchIDCollectorState,
        ctx: OrchestrationContext,
    ) -> None:
        try:
            await asyncio.to_thread(
                insert_puuids_in_batches, state.full_player_puuids, ctx.run_id
            )
            await insert_matchids_stream_in_batches(items, ctx.run_id)
            await asyncio.to_thread(upsert_puuid_timestamp, state.ts, ctx.run_id)
        except Exception:
            await asyncio.to_thread(delete_failed_puuid_timestamp, ctx.run_id)
            await asyncio.to_thread(delete_matchid_puuids, ctx.run_id)
            await asyncio.to_thread(delete_matchids, ctx.run_id)
        finally:
            await asyncio.to_thread(delete_old_puuid_timestamps, ctx.run_id)


if __name__ == "__main__":
    riot_api: RiotAPI = get_riot_api()
    loader = MatchIDLoader()
    collector = MatchIDCollector(riot_api)
    saver = MatchIDSaver()

    orchestrator = MatchIDOrchestrator(
        pipeline="matchids", loader=loader, collector=collector, saver=saver
    )

    asyncio.run(orchestrator.run())
