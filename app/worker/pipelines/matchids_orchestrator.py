from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import uuid4

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
    OrchestrationContext,
    Orchestrator,
)
from app.worker.pipelines.recovery_utils import run_sync_with_retry
from app.worker.pipelines.stop_flag import raise_if_stop_requested
from database.clickhouse.operations.matchids import (
    delete_failed_puuid_timestamp,
    delete_matchid_puuids,
    delete_matchids,
    delete_old_puuid_timestamps,
    insert_matchids_stream_in_batches,
    insert_puuids_in_batches,
    load_matchid_puuid_ts,
    load_matchid_puuids,
    upsert_puuid_timestamp,
)
from database.clickhouse.operations.players import PlayerKeyRow, load_players

logger = logging.getLogger(__name__)

DEFAULT_MATCHIDS_SEASON = 16
MATCHIDS_SEASON_ENV = "MATCHIDS_SEASON"
MATCHIDS_SEASON_START_TIMES = {
    # Riot Support patch schedule: 26.01 / Data Dragon 16.1.x, 2026-01-08 UTC.
    16: 1767830400,
}


def _season_start_ts(season: int) -> int:
    start_ts = MATCHIDS_SEASON_START_TIMES.get(season)
    if start_ts is None:
        known = ", ".join(str(s) for s in sorted(MATCHIDS_SEASON_START_TIMES))
        raise ValueError(
            f"Unsupported matchids season {season}. Known seasons: {known}"
        )
    return start_ts


def _selected_matchids_season() -> int | None:
    raw = os.getenv(MATCHIDS_SEASON_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_MATCHIDS_SEASON

    value = raw.strip().lower()
    if value in {"all", "none", "null", "0"}:
        return None

    season = int(value)
    if season <= 0:
        raise ValueError(f"{MATCHIDS_SEASON_ENV} must be a positive integer")
    return season


def build_matchids_time_window(season: int | None, *, ts: int) -> tuple[int, int]:
    if season is None:
        return 0, ts

    start_ts = _season_start_ts(season)
    end_ts = (
        _season_start_ts(season + 1)
        if season + 1 in MATCHIDS_SEASON_START_TIMES
        else ts
    )
    if start_ts > end_ts:
        raise ValueError(
            f"Matchids season {season} starts after the requested end timestamp"
        )
    return start_ts, end_ts


def build_initial_player_states(
    players: list[PlayerKeyRow],
    collected_puuids: set[str],
    collected_puuids_ts: int,
    *,
    ts: int,
    start_time_floor: int = 0,
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
        start_time = min(max(start_time, start_time_floor), ts)

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
    def __init__(self, *, season: int | None = _selected_matchids_season()) -> None:
        self.season = season

    def load(self, ctx: OrchestrationContext) -> MatchIDCollectorState:
        players: list[PlayerKeyRow] = load_players()
        collected_player_keys = load_matchid_puuids()
        collected_puuids: set[str] = {puuid for puuid, _ in collected_player_keys}
        collected_puuid_ts: int = load_matchid_puuid_ts()
        start_ts, end_ts = build_matchids_time_window(self.season, ts=ctx.ts)

        initial_states = build_initial_player_states(
            players,
            collected_puuids,
            collected_puuid_ts,
            ts=end_ts,
            start_time_floor=start_ts,
        )

        logger.info(
            "MatchIDLoaderWindow season=%s start_ts=%d end_ts=%d collected_ts=%d",
            self.season,
            start_ts,
            end_ts,
            collected_puuid_ts,
        )

        return MatchIDCollectorState(
            initial_states=initial_states,
            full_player_keys=[(p.puuid, p.queue_type) for p in players],
            ts=end_ts,
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
            await run_sync_with_retry(
                logger=logger, component="MatchID",
                op_name="delete_failed_puuid_timestamp",
                func=delete_failed_puuid_timestamp, args=(ctx.run_id,),
            )
            await run_sync_with_retry(
                logger=logger, component="MatchID",
                op_name="delete_matchid_puuids",
                func=delete_matchid_puuids, args=(ctx.run_id,),
            )
            await run_sync_with_retry(
                logger=logger, component="MatchID",
                op_name="delete_matchids",
                func=delete_matchids, args=(ctx.run_id,),
            )
            raise
        finally:
            if timestamp_upserted:
                await run_sync_with_retry(
                    logger=logger, component="MatchID",
                    op_name="delete_old_puuid_timestamps",
                    func=delete_old_puuid_timestamps, args=(ctx.run_id,),
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
