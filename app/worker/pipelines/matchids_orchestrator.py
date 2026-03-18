from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from dataclasses import dataclass, field
from typing import AsyncIterator
import time
from tenacity import before_sleep_log, retry, stop_never, wait_exponential
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
    Orchestrator,
    OrchestrationContext,
)
from app.worker.pipelines.stop_flag import raise_if_stop_requested
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

logger = logging.getLogger(__name__)

def build_initial_player_states(
    players: list[PlayerKeyRow],
    collected_puuids: set[str],
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
    full_player_keys: list[tuple[str, str]]
    ts: int
    failed_player_keys: set[tuple[str, str]] = field(default_factory=set)


class MatchIDOrchestrator(Orchestrator):
    async def _dedupe_async(
        self, batches: AsyncIterator[list[tuple[str, str]]]
    ) -> AsyncIterator[list[tuple[str, str]]]:
        seen: set[str] = set()
        async for batch in batches:
            raise_if_stop_requested(stage="match_ids:dedupe")
            if not batch:
                continue

            out: list[tuple[str, str]] = []
            for mid, queue_type in batch:
                if mid not in seen:
                    seen.add(mid)
                    out.append((mid, queue_type))

            if out:
                yield out

    async def run(self) -> None:
        ctx = OrchestrationContext(
            ts=int(time.time()), run_id=uuid4(), pipeline=self.pipeline
        )
        state: MatchIDCollectorState = self.loader.load(ctx)

        match_ids_stream: AsyncIterator[list[tuple[str, str]]] = self.collector.collect(
            state,
            ctx,
        )
        match_ids_stream = self._dedupe_async(match_ids_stream)

        await self.saver.save(match_ids_stream, state, ctx)


class MatchIDLoader:
    def load(self, ctx: OrchestrationContext) -> MatchIDCollectorState:
        players: list[PlayerKeyRow] = load_players()
        collected_player_keys = load_matchid_puuids()
        collected_puuids: set[str] = {puuid for puuid, _ in collected_player_keys}
        collected_puuid_ts: int = load_matchid_puuid_ts()

        initial_states = build_initial_player_states(
            players,
            collected_puuids,
            collected_puuid_ts,
            ts=ctx.ts,
        )

        return MatchIDCollectorState(
            initial_states=initial_states,
            full_player_keys=[(p.puuid, p.queue_type) for p in players],
            ts=ctx.ts,
        )


class MatchIDCollector(Collector):
    def __init__(self, riot_api: RiotAPI):
        self.riot_api = riot_api

    async def collect(
        self, state: MatchIDCollectorState, ctx: OrchestrationContext
    ) -> AsyncIterator[list[tuple[str, str]]]:
        _ = ctx
        raise_if_stop_requested(stage="match_ids:start")
        async for stream_item in stream_match_ids(
            self.riot_api,
            initial_states=state.initial_states,
        ):
            raise_if_stop_requested(stage="match_ids:collect")
            player_key, match_ids, error = stream_item
            puuid, queue_type = player_key
            if error is not None:
                state.failed_player_keys.add(player_key)
                logger.warning(
                    "MatchIDCrawlFailed puuid=%s queue_type=%s error=%s",
                    puuid,
                    queue_type,
                    error,
                )
                continue
            if match_ids:
                yield [(mid, queue_type) for mid in match_ids]


class MatchIDSaver:
    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=1, min=2, max=300),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _delete_with_retry(self, op_name: str, fn, *args) -> None:
        try:
            await asyncio.to_thread(fn, *args)
        except Exception as e:
            logger.exception("Error during %s: %s", op_name, e)
            raise

    async def save(
        self,
        items: AsyncIterator[list[tuple[str, str]]],
        state: MatchIDCollectorState,
        ctx: OrchestrationContext,
    ) -> None:
        timestamp_upserted = False
        try:
            await insert_matchids_stream_in_batches(items, ctx.run_id)
            successful_player_keys = [
                player_key
                for player_key in state.full_player_keys
                if player_key not in state.failed_player_keys
            ]
            if state.failed_player_keys:
                logger.info(
                    "MatchIDExcludingFailedPlayers failed=%d successful=%d",
                    len(state.failed_player_keys),
                    len(successful_player_keys),
                )
            await asyncio.to_thread(
                insert_puuids_in_batches,
                successful_player_keys,
                ctx.run_id,
            )
            await asyncio.to_thread(
                upsert_puuid_timestamp,
                state.ts,
                ctx.run_id,
            )
            timestamp_upserted = True
        except Exception:
            await self._delete_with_retry(
                "delete_failed_puuid_timestamp", delete_failed_puuid_timestamp, ctx.run_id
            )
            await self._delete_with_retry(
                "delete_matchid_puuids", delete_matchid_puuids, ctx.run_id
            )
            await self._delete_with_retry("delete_matchids", delete_matchids, ctx.run_id)
            raise
        finally:
            if timestamp_upserted:
                await self._delete_with_retry(
                    "delete_old_puuid_timestamps", delete_old_puuid_timestamps, ctx.run_id
                )
            else:
                logger.warning(
                    "Skipping delete_old_puuid_timestamps for run_id=%s because timestamp upsert did not complete.",
                    ctx.run_id,
                )


if __name__ == "__main__":
    riot_api: RiotAPI = get_riot_api()
    loader = MatchIDLoader()
    collector = MatchIDCollector(riot_api)
    saver = MatchIDSaver()

    orchestrator = MatchIDOrchestrator(
        pipeline="matchids", loader=loader, collector=collector, saver=saver
    )

    asyncio.run(orchestrator.run())
