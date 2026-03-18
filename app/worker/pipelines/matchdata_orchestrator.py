from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from operator import attrgetter
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Literal,
    TypeAlias,
)
from uuid import UUID, uuid4

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.core.config.settings import settings
from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.match_data import (
    stream_non_timeline_data,
    stream_timeline_data,
)
from app.services.riot_api_client.parsers.non_timeline import (
    MatchDataNonTimelineParsingOrchestrator,
    NonTimelineTables,
    TabulatedBan,
    TabulatedFeat,
    TabulatedInfo,
    TabulatedMetadata,
    TabulatedObjective,
    TabulatedParticipantChallenges,
    TabulatedParticipantPerkIds,
    TabulatedParticipantPerkValues,
    TabulatedParticipantStats,
)
from app.services.riot_api_client.parsers.timeline import (
    BuildingKillRow,
    ChampionKillDamageInstanceRow,
    ChampionKillRow,
    ChampionSpecialKillRow,
    DragonSoulGivenRow,
    EliteMonsterKillRow,
    MatchDataTimelineParsingOrchestrator,
    ParticipantStatsRow,
    RareEventRow,
    TimelineTables,
    TurretPlateDestroyedRow,
)
from app.worker.pipelines.orchestrator import (
    Collector,
    Loader,
    OrchestrationContext,
    Orchestrator,
    Saver,
)
from app.worker.pipelines.recovery_utils import run_sync_with_retry
from app.worker.pipelines.stop_flag import raise_if_stop_requested
from database.clickhouse.operations.matchdata import (
    delete_by_run_id_and_matchids,
)
from database.clickhouse.operations.utils import delete_by_run_id, persist_data
from database.clickhouse.operations.work_state import (
    claim_pending_matchids,
    ensure_matchdata_state_schema,
    mark_matchids_finished,
    seed_from_latest_matchids,
)

logger = logging.getLogger(__name__)

# RECOVERY-SYSTEM: batch checkpointing config.
MATCHDATA_BATCH_SIZE = 10_000
MATCHDATA_FLUSH_WINDOW_CALLS = 250
MATCHDATA_MIN_FLUSH_INTERVAL_S = 60.0
MATCHDATA_MAX_FLUSH_INTERVAL_S = 5_000.0
MATCHDATA_FLUSH_INTERVAL_MULTIPLIER = 6.0


def _flush_interval_from_rate_limit() -> float:
    # Approximate "work window" from sustained API pacing:
    #  - one token every (period / calls) seconds
    #  - use a larger token window to favor size-based batching
    #  - keep safety clamps so buffers still flush periodically
    call_interval_s = float(settings.rate_limit_period) / float(
        settings.rate_limit_calls
    )
    return max(
        MATCHDATA_MIN_FLUSH_INTERVAL_S,
        min(
            MATCHDATA_MAX_FLUSH_INTERVAL_S,
            call_interval_s * MATCHDATA_FLUSH_WINDOW_CALLS,
        ),
    )


def columns_from_typed_dict(td: type) -> tuple[str, ...]:
    annotations: dict[str, Any] = {}
    for cls in reversed(td.__mro__):
        annotations.update(getattr(cls, "__annotations__", {}))
    return tuple(annotations)


BUILDING_KILL_COLS = columns_from_typed_dict(BuildingKillRow)
CHAMPION_KILL_COLS = columns_from_typed_dict(ChampionKillRow)
DRAGON_SOUL_GIVEN_COLS = columns_from_typed_dict(DragonSoulGivenRow)
ELITE_MONSTER_KILL_COLS = columns_from_typed_dict(EliteMonsterKillRow)
PARTICIPANT_STATS_COLS = columns_from_typed_dict(ParticipantStatsRow)
CHAMPION_SPECIAL_KILL_COLS = columns_from_typed_dict(ChampionSpecialKillRow)
TURRET_PLATE_DESTROYED_COLS = columns_from_typed_dict(TurretPlateDestroyedRow)
PAYLOAD_EVENT_COLS = columns_from_typed_dict(RareEventRow)
CHAMPION_KILL_DAMAGE_INSTANCE_COLS = columns_from_typed_dict(
    ChampionKillDamageInstanceRow
)

TABULATED_METADATA_COLS = columns_from_typed_dict(TabulatedMetadata)
TABULATED_INFO_COLS = columns_from_typed_dict(TabulatedInfo)
TABULATED_BAN_COLS = columns_from_typed_dict(TabulatedBan)
TABULATED_FEAT_COLS = columns_from_typed_dict(TabulatedFeat)
TABULATED_OBJECTIVE_COLS = columns_from_typed_dict(TabulatedObjective)
TABULATED_PARTICIPANT_STATS_COLS = columns_from_typed_dict(TabulatedParticipantStats)
TABULATED_PARTICIPANT_PERK_VALUES_COLS = columns_from_typed_dict(
    TabulatedParticipantPerkValues
)
TABULATED_PARTICIPANT_PERK_IDS_COLS = columns_from_typed_dict(
    TabulatedParticipantPerkIds
)
TABULATED_PARTICIPANT_CHALLENGES_COLS = columns_from_typed_dict(
    TabulatedParticipantChallenges
)

NON_TIMELINE_DELETE_TABLES: tuple[str, ...] = (
    "game_data.metadata",
    "game_data.info",
    "game_data.bans",
    "game_data.feats",
    "game_data.objectives",
    "game_data.participant_stats",
    "game_data.participant_challenges",
    "game_data.participant_perk_values",
    "game_data.participant_perk_ids",
)

TIMELINE_DELETE_TABLES: tuple[str, ...] = (
    "game_data.tl_participant_stats",
    "game_data.tl_building_kill",
    "game_data.tl_champion_kill",
    "game_data.tl_champion_special_kill",
    "game_data.tl_dragon_soul_given",
    "game_data.tl_elite_monster_kill",
    "game_data.tl_payload_event",
    "game_data.tl_turret_plate_destroyed",
    "game_data.tl_ck_victim_damage_dealt",
    "game_data.tl_ck_victim_damage_received",
)


NON_TIMELINE_INSERTS = (
    ("game_data.metadata", TABULATED_METADATA_COLS, attrgetter("metadata")),
    ("game_data.info", TABULATED_INFO_COLS, attrgetter("game_info")),
    ("game_data.bans", TABULATED_BAN_COLS, attrgetter("bans")),
    ("game_data.feats", TABULATED_FEAT_COLS, attrgetter("feats")),
    ("game_data.objectives", TABULATED_OBJECTIVE_COLS, attrgetter("objectives")),
    (
        "game_data.participant_stats",
        TABULATED_PARTICIPANT_STATS_COLS,
        attrgetter("participant_stats"),
    ),
    (
        "game_data.participant_challenges",
        TABULATED_PARTICIPANT_CHALLENGES_COLS,
        attrgetter("participant_challenges"),
    ),
    (
        "game_data.participant_perk_values",
        TABULATED_PARTICIPANT_PERK_VALUES_COLS,
        attrgetter("participant_perk_values"),
    ),
    (
        "game_data.participant_perk_ids",
        TABULATED_PARTICIPANT_PERK_IDS_COLS,
        attrgetter("participant_perk_ids"),
    ),
)

TIMELINE_INSERTS = (
    (
        "game_data.tl_participant_stats",
        PARTICIPANT_STATS_COLS,
        attrgetter("participantStats"),
    ),
    ("game_data.tl_building_kill", BUILDING_KILL_COLS, attrgetter("buildingKill")),
    ("game_data.tl_champion_kill", CHAMPION_KILL_COLS, attrgetter("championKill")),
    (
        "game_data.tl_champion_special_kill",
        CHAMPION_SPECIAL_KILL_COLS,
        attrgetter("championSpecialKill"),
    ),
    (
        "game_data.tl_dragon_soul_given",
        DRAGON_SOUL_GIVEN_COLS,
        attrgetter("dragonSoulGiven"),
    ),
    (
        "game_data.tl_elite_monster_kill",
        ELITE_MONSTER_KILL_COLS,
        attrgetter("eliteMonsterKill"),
    ),
    ("game_data.tl_payload_event", PAYLOAD_EVENT_COLS, attrgetter("payloadEvents")),
    (
        "game_data.tl_turret_plate_destroyed",
        TURRET_PLATE_DESTROYED_COLS,
        attrgetter("turretPlateDestroyed"),
    ),
    (
        "game_data.tl_ck_victim_damage_dealt",
        CHAMPION_KILL_DAMAGE_INSTANCE_COLS,
        attrgetter("championKillVictimDamageDealt"),
    ),
    (
        "game_data.tl_ck_victim_damage_received",
        CHAMPION_KILL_DAMAGE_INSTANCE_COLS,
        attrgetter("championKillVictimDamageReceived"),
    ),
)

StreamName: TypeAlias = Literal["non_timeline", "timeline"]


@dataclass(frozen=True)
class StreamItem:
    stream: StreamName
    raw: Any


@dataclass(frozen=True)
class _Done:
    stream: StreamName


QueueMsg: TypeAlias = StreamItem | _Done


@dataclass(frozen=True)
class MatchDataCollectorState:
    matchids: list[str]


class MatchDataOrchestrator(Orchestrator):
    def __init__(
        self,
        *,
        pipeline: str,
        loader: Loader,
        non_timeline_collector: Collector,
        timeline_collector: Collector,
        saver: Saver,
    ) -> None:
        super().__init__(pipeline, loader, non_timeline_collector, saver)
        self.timeline_collector = timeline_collector

    async def combine_streams(
        self,
        non_timeline: AsyncIterator[Any],
        timeline: AsyncIterator[Any],
        *,
        max_buffer: int = 3_000,
    ) -> AsyncIterator[StreamItem]:
        q: asyncio.Queue[QueueMsg] = asyncio.Queue(maxsize=max_buffer)

        async def pump(name: StreamName, it: AsyncIterable[Any]) -> None:
            try:
                async for x in it:
                    await q.put(StreamItem(stream=name, raw=x))
            finally:
                await q.put(_Done(stream=name))

        async with asyncio.TaskGroup() as tg:
            tg.create_task(pump("non_timeline", non_timeline))
            tg.create_task(pump("timeline", timeline))

            done: set[StreamName] = set()
            while len(done) < 2:
                msg = await q.get()

                if isinstance(msg, _Done):
                    done.add(msg.stream)
                    continue

                yield msg

    async def run(self) -> None:
        # RECOVERY-SYSTEM: run in small claimed batches until no pending work remains.
        ts = int(time.time())
        batch_number = 0

        while True:
            raise_if_stop_requested(stage="match_data:batch-start")
            ctx = OrchestrationContext(ts=ts, run_id=uuid4(), pipeline=self.pipeline)
            state: MatchDataCollectorState = self.loader.load(ctx)
            if not state.matchids:
                logger.info(
                    "MatchData no pending matchids remain; exiting pipeline=%s",
                    self.pipeline,
                )
                return

            batch_number += 1
            logger.info(
                "MatchData batch start pipeline=%s batch=%d run_id=%s size=%d",
                self.pipeline,
                batch_number,
                ctx.run_id,
                len(state.matchids),
            )

            non_timeline_raw = self.collector.collect(state, ctx)
            timeline_raw = self.timeline_collector.collect(state, ctx)
            items = self.combine_streams(non_timeline_raw, timeline_raw)
            await self.saver.save(items, state, ctx)

            logger.info(
                "MatchData batch complete pipeline=%s batch=%d run_id=%s",
                self.pipeline,
                batch_number,
                ctx.run_id,
            )


class MatchDataLoader(Loader):
    def __init__(
        self,
        *,
        batch_size: int = MATCHDATA_BATCH_SIZE,
    ) -> None:
        self.batch_size = batch_size
        self._initialized = False

    def load(self, ctx: OrchestrationContext) -> MatchDataCollectorState:
        _ = ctx
        ensure_matchdata_state_schema()
        if not self._initialized:
            seeded_pending = seed_from_latest_matchids()
            if seeded_pending:
                logger.info("MatchData loader seeded pending=%d", seeded_pending)
            self._initialized = True

        claimed = claim_pending_matchids(batch_size=self.batch_size)
        if not claimed:
            logger.info("MatchData loader source=none size=0")
            return MatchDataCollectorState(matchids=[])

        logger.info("MatchData loader source=state_queue size=%d", len(claimed))
        return MatchDataCollectorState(matchids=claimed)


class MatchDataStreamCollector(Collector):
    def __init__(self, riot_api: RiotAPI, *, stream: StreamName) -> None:
        self.riot_api = riot_api
        self.stream = stream

    async def collect(
        self, state: MatchDataCollectorState, ctx: OrchestrationContext
    ) -> AsyncIterator[dict[str, Any]]:
        _ = ctx
        iterator: AsyncIterator[dict[str, Any]]
        if self.stream == "non_timeline":
            iterator = stream_non_timeline_data(state.matchids, riot_api=self.riot_api)
        else:
            iterator = stream_timeline_data(state.matchids, riot_api=self.riot_api)

        raise_if_stop_requested(stage=f"match_data:{self.stream}:start")
        async for raw in iterator:
            raise_if_stop_requested(stage=f"match_data:{self.stream}:collect")
            yield raw


class MatchDataSaver(Saver):
    def __init__(
        self,
        *,
        non_timeline_parser: Any,
        timeline_parser: Any,
    ) -> None:
        self.non_timeline_parser = non_timeline_parser
        self.timeline_parser = timeline_parser

        self.batch_size = MATCHDATA_BATCH_SIZE
        self.flush_interval_s = min(
            MATCHDATA_MAX_FLUSH_INTERVAL_S,
            _flush_interval_from_rate_limit() * MATCHDATA_FLUSH_INTERVAL_MULTIPLIER,
        )
        self._table_meta: dict[str, tuple[tuple[str, ...], int]] = {
            table: (cols, self.batch_size)
            for table, cols, _ in (*NON_TIMELINE_INSERTS, *TIMELINE_INSERTS)
        }

    async def save(
        self,
        items: AsyncIterator[Any],
        state: MatchDataCollectorState,
        ctx: OrchestrationContext,
    ) -> None:
        if not state.matchids:
            return

        # RECOVERY-SYSTEM: keep successful rows, requeue only failed match IDs.
        stream_successes: dict[str, set[StreamName]] = defaultdict(set)
        buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        last_flush = time.monotonic()

        try:
            async for item_any in items:
                raise_if_stop_requested(stage="match_data:save")
                item: StreamItem = item_any
                match_id = self._extract_match_id(item.raw)

                if item.stream == "non_timeline":
                    nt: NonTimelineTables = await asyncio.to_thread(
                        self.non_timeline_parser.run, item.raw
                    )
                    if match_id != "unknown":
                        stream_successes[match_id].add("non_timeline")
                    await self._buffer_inserts(
                        NON_TIMELINE_INSERTS,
                        nt,
                        buffers,
                        ctx.run_id,
                    )

                elif item.stream == "timeline":
                    tl: TimelineTables = await asyncio.to_thread(
                        self.timeline_parser.run, item.raw
                    )
                    if match_id != "unknown":
                        stream_successes[match_id].add("timeline")
                    await self._buffer_inserts(
                        TIMELINE_INSERTS,
                        tl,
                        buffers,
                        ctx.run_id,
                    )

                else:
                    raise ValueError(f"Unknown stream: {item.stream!r}")

                now = time.monotonic()
                if (now - last_flush) >= self.flush_interval_s:
                    await self._flush_all_buffers(buffers, ctx.run_id)
                    last_flush = now

            await self._flush_all_buffers(buffers, ctx.run_id)

            successful_match_ids: list[str] = [
                match_id
                for match_id in state.matchids
                if stream_successes.get(match_id) == {"non_timeline", "timeline"}
            ]
            failed_match_ids: list[str] = [
                match_id
                for match_id in state.matchids
                if stream_successes.get(match_id) != {"non_timeline", "timeline"}
            ]

            if failed_match_ids:
                logger.warning(
                    "MatchData partial failure run_id=%s failed=%d sample=%s",
                    ctx.run_id,
                    len(failed_match_ids),
                    failed_match_ids[:20],
                )
                await self.delete_failed_matchids(ctx.run_id, failed_match_ids)

            await self.mark_finished_matchids(successful_match_ids)

            logger.info(
                "MatchData completion run_id=%s total=%d finished=%d requeued=%d",
                ctx.run_id,
                len(state.matchids),
                len(successful_match_ids),
                len(failed_match_ids),
            )

        except Exception as exc:
            await self.rollback_run(ctx.run_id)
            logger.exception(
                "MatchData batch exception run_id=%s: %s",
                ctx.run_id,
                exc,
            )
            raise

    async def _run_deletes(self, tables: tuple[str, ...], run_id: UUID) -> None:
        for table in tables:
            await run_sync_with_retry(
                logger=logger,
                component="MatchData",
                op_name=f"delete_by_run_id:{table}",
                func=delete_by_run_id,
                args=(table, run_id),
            )

    async def delete_failed_matchids(self, run_id: UUID, match_ids: list[str]) -> None:
        await self._run_deletes_for_matchids(
            NON_TIMELINE_DELETE_TABLES, run_id, match_ids
        )
        await self._run_deletes_for_matchids(TIMELINE_DELETE_TABLES, run_id, match_ids)

    async def _run_deletes_for_matchids(
        self,
        tables: tuple[str, ...],
        run_id: UUID,
        match_ids: list[str],
    ) -> None:
        for table in tables:
            await run_sync_with_retry(
                logger=logger,
                component="MatchData",
                op_name=f"delete_by_run_id_and_matchids:{table}",
                func=delete_by_run_id_and_matchids,
                args=(table, run_id, match_ids),
            )

    async def mark_finished_matchids(self, match_ids: list[str]) -> None:
        await run_sync_with_retry(
            logger=logger,
            component="MatchData",
            op_name="mark_matchids_finished",
            func=mark_matchids_finished,
            kwargs={"match_ids": match_ids},
        )

    async def rollback_run(self, run_id: UUID) -> None:
        await self._run_deletes(NON_TIMELINE_DELETE_TABLES, run_id)
        await self._run_deletes(TIMELINE_DELETE_TABLES, run_id)

    @staticmethod
    def _extract_match_id(raw: Any) -> str:
        if not isinstance(raw, dict):
            return "unknown"
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            return "unknown"
        match_id = metadata.get("matchId")
        return match_id if isinstance(match_id, str) else "unknown"

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _insert_one(
        self, table: str, cols, items, run_id: UUID, *, batch_size: int
    ) -> None:
        if not items:
            return
        try:
            await asyncio.to_thread(
                persist_data, table, cols, items, run_id, batch_size
            )
        except Exception as e:
            logger.exception(
                "Error inserting into %s run_id=%s: %s",
                table,
                run_id,
                e,
            )
            raise

    async def _buffer_inserts(
        self,
        specs,
        parsed,
        buffers: dict[str, list[dict[str, Any]]],
        run_id: UUID,
    ) -> None:
        for table, _, getter in specs:
            items = list(getter(parsed))
            if not items:
                continue
            buffers[table].extend(items)
            _, batch_size = self._table_meta[table]
            if len(buffers[table]) >= batch_size:
                await self._flush_table_buffer(table, buffers, run_id)

    async def _flush_table_buffer(
        self,
        table: str,
        buffers: dict[str, list[dict[str, Any]]],
        run_id: UUID,
    ) -> None:
        items = buffers.get(table)
        if not items:
            return
        cols, batch_size = self._table_meta[table]
        buffers[table] = []
        await self._insert_one(
            table,
            cols,
            items,
            run_id,
            batch_size=batch_size,
        )

    async def _flush_all_buffers(
        self,
        buffers: dict[str, list[dict[str, Any]]],
        run_id: UUID,
    ) -> None:
        for table in tuple(buffers.keys()):
            await self._flush_table_buffer(table, buffers, run_id)


if __name__ == "__main__":

    async def _main() -> None:
        async with get_riot_api() as riot_api:
            loader = MatchDataLoader(batch_size=MATCHDATA_BATCH_SIZE)
            non_timeline_collector = MatchDataStreamCollector(
                riot_api=riot_api,
                stream="non_timeline",
            )
            timeline_collector = MatchDataStreamCollector(
                riot_api=riot_api,
                stream="timeline",
            )

            saver = MatchDataSaver(
                non_timeline_parser=MatchDataNonTimelineParsingOrchestrator(),
                timeline_parser=MatchDataTimelineParsingOrchestrator(),
            )

            orchestrator = MatchDataOrchestrator(
                pipeline="match_data",
                loader=loader,
                non_timeline_collector=non_timeline_collector,
                timeline_collector=timeline_collector,
                saver=saver,
            )

            await orchestrator.run()

    asyncio.run(_main())
