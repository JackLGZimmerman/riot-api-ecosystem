from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from operator import attrgetter
from typing import Any, Literal
from uuid import UUID, uuid4

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.core.config.constants.generic import RETRYABLE
from app.core.config.settings import settings
from app.services.riot_api_client.base import RiotAPI
from app.services.riot_api_client.match_data import MatchFetchResult, stream_match_data
from app.services.riot_api_client.parsers.non_timeline import (
    TabulatedBan,
    TabulatedFeat,
    TabulatedInfo,
    TabulatedMetadata,
    TabulatedObjective,
    TabulatedParticipantChallenges,
    TabulatedParticipantPerkIds,
    TabulatedParticipantPerkValues,
    TabulatedParticipantStats,
    is_abort_payload,
)
from app.services.riot_api_client.parsers.timeline import (
    BuildingKillRow,
    ChampionKillDamageInstanceRow,
    ChampionKillRow,
    ChampionSpecialKillRow,
    ChampionTransformRow,
    DragonSoulGivenRow,
    EliteMonsterKillRow,
    FeatUpdateRow,
    GameEndRow,
    ItemDestroyedRow,
    ItemPurchasedRow,
    ItemSoldRow,
    ItemUndoRow,
    LevelUpRow,
    ObjectiveBountyFinishRow,
    ObjectiveBountyPrestartRow,
    ParticipantStatsRow,
    PauseEndRow,
    SkillLevelUpRow,
    TurretPlateDestroyedRow,
    WardKillRow,
    WardPlacedRow,
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
    delete_by_matchids,
)
from database.clickhouse.operations.utils import delete_by_run_id, persist_data
from database.clickhouse.operations.work_state import (
    claim_pending_matchids,
    mark_matchids_finished,
    seed_from_latest_matchids,
)

logger = logging.getLogger(__name__)

# RECOVERY-SYSTEM: batch checkpointing config.
# Keep the claim batch smaller so any rollback/delete scope is capped per run,
# while preserving larger row insert batches for throughput.
MATCHDATA_CLAIM_BATCH_SIZE = 250
MATCHDATA_INSERT_BATCH_SIZE = 10_000
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


@dataclass(frozen=True)
class TableSpec:
    table: str
    columns: tuple[str, ...]
    getter: Callable[[Any], Iterable[dict[str, Any]]]


def _table_spec(table: str, row_type: type[Any], attr: str) -> TableSpec:
    return TableSpec(
        table=table,
        columns=columns_from_typed_dict(row_type),
        getter=attrgetter(attr),
    )


NON_TIMELINE_TABLE_SPECS = (
    _table_spec("game_data.metadata", TabulatedMetadata, "metadata"),
    _table_spec("game_data.info", TabulatedInfo, "game_info"),
    _table_spec("game_data.bans", TabulatedBan, "bans"),
    _table_spec("game_data.feats", TabulatedFeat, "feats"),
    _table_spec("game_data.objectives", TabulatedObjective, "objectives"),
    _table_spec(
        "game_data.participant_stats",
        TabulatedParticipantStats,
        "participant_stats",
    ),
    _table_spec(
        "game_data.participant_challenges",
        TabulatedParticipantChallenges,
        "participant_challenges",
    ),
    _table_spec(
        "game_data.participant_perk_values",
        TabulatedParticipantPerkValues,
        "participant_perk_values",
    ),
    _table_spec(
        "game_data.participant_perk_ids",
        TabulatedParticipantPerkIds,
        "participant_perk_ids",
    ),
)

TIMELINE_TABLE_SPECS = (
    _table_spec(
        "game_data.tl_participant_stats",
        ParticipantStatsRow,
        "participantStats",
    ),
    _table_spec("game_data.tl_building_kill", BuildingKillRow, "buildingKill"),
    _table_spec("game_data.tl_champion_kill", ChampionKillRow, "championKill"),
    _table_spec(
        "game_data.tl_champion_special_kill",
        ChampionSpecialKillRow,
        "championSpecialKill",
    ),
    _table_spec(
        "game_data.tl_dragon_soul_given",
        DragonSoulGivenRow,
        "dragonSoulGiven",
    ),
    _table_spec(
        "game_data.tl_elite_monster_kill",
        EliteMonsterKillRow,
        "eliteMonsterKill",
    ),
    _table_spec("game_data.tl_ward_placed", WardPlacedRow, "wardPlaced"),
    _table_spec("game_data.tl_ward_kill", WardKillRow, "wardKill"),
    _table_spec("game_data.tl_item_purchased", ItemPurchasedRow, "itemPurchased"),
    _table_spec("game_data.tl_item_sold", ItemSoldRow, "itemSold"),
    _table_spec("game_data.tl_item_destroyed", ItemDestroyedRow, "itemDestroyed"),
    _table_spec("game_data.tl_item_undo", ItemUndoRow, "itemUndo"),
    _table_spec("game_data.tl_level_up", LevelUpRow, "levelUp"),
    _table_spec("game_data.tl_skill_level_up", SkillLevelUpRow, "skillLevelUp"),
    _table_spec("game_data.tl_pause_end", PauseEndRow, "pauseEnd"),
    _table_spec("game_data.tl_game_end", GameEndRow, "gameEnd"),
    _table_spec(
        "game_data.tl_objective_bounty_prestart",
        ObjectiveBountyPrestartRow,
        "objectiveBountyPrestart",
    ),
    _table_spec(
        "game_data.tl_objective_bounty_finish",
        ObjectiveBountyFinishRow,
        "objectiveBountyFinish",
    ),
    _table_spec("game_data.tl_feat_update", FeatUpdateRow, "featUpdate"),
    _table_spec(
        "game_data.tl_champion_transform",
        ChampionTransformRow,
        "championTransform",
    ),
    _table_spec(
        "game_data.tl_turret_plate_destroyed",
        TurretPlateDestroyedRow,
        "turretPlateDestroyed",
    ),
    _table_spec(
        "game_data.tl_ck_victim_damage_dealt",
        ChampionKillDamageInstanceRow,
        "championKillVictimDamageDealt",
    ),
    _table_spec(
        "game_data.tl_ck_victim_damage_received",
        ChampionKillDamageInstanceRow,
        "championKillVictimDamageReceived",
    ),
)

ALL_TABLE_SPECS = (*NON_TIMELINE_TABLE_SPECS, *TIMELINE_TABLE_SPECS)
ALL_DELETE_TABLES = tuple(spec.table for spec in ALL_TABLE_SPECS)

type StreamName = Literal["non_timeline", "timeline"]


@dataclass(frozen=True)
class StreamItem:
    stream: StreamName
    raw: Any


@dataclass(frozen=True)
class _Done:
    stream: StreamName


type QueueMsg = StreamItem | _Done


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
        batch_size: int = MATCHDATA_CLAIM_BATCH_SIZE,
    ) -> None:
        self.batch_size = batch_size
        self._initialized = False

    def load(self, ctx: OrchestrationContext) -> MatchDataCollectorState:
        _ = ctx
        if not self._initialized:
            seeded_pending = seed_from_latest_matchids()
            if seeded_pending:
                logger.info("MatchData loader seeded pending=%d", seeded_pending)
            self._initialized = True

        claimed = claim_pending_matchids(batch_size=self.batch_size)
        logger.info(
            "MatchData loader source=%s size=%d",
            "state_queue" if claimed else "none",
            len(claimed),
        )
        return MatchDataCollectorState(matchids=claimed)


class MatchDataStreamCollector(Collector):
    def __init__(self, riot_api: RiotAPI, *, stream: StreamName) -> None:
        self.riot_api = riot_api
        self.stream = stream

    async def collect(
        self, state: MatchDataCollectorState, ctx: OrchestrationContext
    ) -> AsyncIterator[MatchFetchResult]:
        _ = ctx
        endpoint_type = (
            "by_match_id" if self.stream == "non_timeline" else "timeline_by_match_id"
        )
        iterator = stream_match_data(
            state.matchids,
            endpoint_type=endpoint_type,
            riot_api=self.riot_api,
        )

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

        self.batch_size = MATCHDATA_INSERT_BATCH_SIZE
        self.flush_interval_s = min(
            MATCHDATA_MAX_FLUSH_INTERVAL_S,
            _flush_interval_from_rate_limit() * MATCHDATA_FLUSH_INTERVAL_MULTIPLIER,
        )
        self._table_columns: dict[str, tuple[str, ...]] = {
            spec.table: spec.columns for spec in ALL_TABLE_SPECS
        }

    async def save(
        self,
        items: AsyncIterator[Any],
        state: MatchDataCollectorState,
        ctx: OrchestrationContext,
    ) -> None:
        if not state.matchids:
            return

        stream_successes: dict[str, set[StreamName]] = defaultdict(set)
        stream_terminals: dict[str, set[StreamName]] = defaultdict(set)
        aborted_match_ids: set[str] = set()
        buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        last_flush = time.monotonic()

        parsers: dict[StreamName, tuple[Any, tuple[TableSpec, ...]]] = {
            "non_timeline": (self.non_timeline_parser, NON_TIMELINE_TABLE_SPECS),
            "timeline": (self.timeline_parser, TIMELINE_TABLE_SPECS),
        }

        try:
            async for item in items:
                raise_if_stop_requested(stage="match_data:save")
                fetch: MatchFetchResult = item.raw
                stream: StreamName = item.stream
                match_id = fetch.match_id

                if fetch.data is None:
                    if fetch.status is not None and fetch.status not in RETRYABLE:
                        stream_terminals[match_id].add(stream)
                        logger.warning(
                            "MatchDataTerminal match_id=%s stream=%s status=%s",
                            match_id,
                            stream,
                            fetch.status,
                        )
                    continue

                if stream == "non_timeline" and is_abort_payload(fetch.data):
                    aborted_match_ids.add(match_id)
                    logger.info("MatchDataAbort match_id=%s; retiring.", match_id)
                    continue

                parser, specs = parsers[stream]
                parsed = await asyncio.to_thread(parser.run, fetch.data)
                self._attach_match_id(parsed, match_id)
                stream_successes[match_id].add(stream)
                await self._buffer_inserts(specs, parsed, buffers, ctx.run_id)

                now = time.monotonic()
                if (now - last_flush) >= self.flush_interval_s:
                    await self._flush_all_buffers(buffers, ctx.run_id)
                    last_flush = now

            await self._flush_all_buffers(buffers, ctx.run_id)

            both: set[StreamName] = {"non_timeline", "timeline"}
            finished: list[str] = []
            retired: list[str] = list(aborted_match_ids)
            requeued: list[str] = []
            for mid in state.matchids:
                if mid in aborted_match_ids:
                    continue
                successes = stream_successes.get(mid, set())
                terminals = stream_terminals.get(mid, set())
                if successes | terminals != both:
                    requeued.append(mid)
                elif successes:
                    finished.append(mid)
                else:
                    retired.append(mid)

            if requeued:
                logger.warning(
                    "MatchData retryable failure run_id=%s count=%d sample=%s",
                    ctx.run_id,
                    len(requeued),
                    requeued[:20],
                )
                await self.delete_failed_matchids(requeued)

            if retired:
                logger.warning(
                    "MatchData retired run_id=%s count=%d aborts=%d sample=%s",
                    ctx.run_id,
                    len(retired),
                    len(aborted_match_ids),
                    retired[:20],
                )
                await self.delete_failed_matchids(retired)
                await self.delete_source_matchids(retired)

            await self.mark_finished_matchids([*finished, *retired])

            logger.info(
                "MatchData done run_id=%s total=%d ok=%d retired=%d requeued=%d",
                ctx.run_id,
                len(state.matchids),
                len(finished),
                len(retired),
                len(requeued),
            )

        except Exception as exc:
            await self.delete_failed_matchids(state.matchids)
            await self.rollback_run(ctx.run_id)
            logger.exception(
                "MatchData batch exception run_id=%s: %s",
                ctx.run_id,
                exc,
            )
            raise

    async def _delete_tables(
        self,
        tables: tuple[str, ...],
        func: Callable[..., Any],
        *func_args: Any,
    ) -> None:
        for table in tables:
            await run_sync_with_retry(
                logger=logger,
                component="MatchData",
                op_name=f"{func.__name__}:{table}",
                func=func,
                args=(table, *func_args),
            )

    async def delete_failed_matchids(self, match_ids: list[str]) -> None:
        await self._delete_tables(ALL_DELETE_TABLES, delete_by_matchids, match_ids)

    async def mark_finished_matchids(self, match_ids: list[str]) -> None:
        await run_sync_with_retry(
            logger=logger,
            component="MatchData",
            op_name="mark_matchids_finished",
            func=mark_matchids_finished,
            kwargs={"match_ids": match_ids},
        )

    async def delete_source_matchids(self, match_ids: list[str]) -> None:
        await run_sync_with_retry(
            logger=logger,
            component="MatchData",
            op_name="delete_source_matchids",
            func=delete_by_matchids,
            args=("game_data.matchids", match_ids),
        )

    async def rollback_run(self, run_id: UUID) -> None:
        await self._delete_tables(ALL_DELETE_TABLES, delete_by_run_id, run_id)

    @staticmethod
    def _attach_match_id(parsed: Any, match_id: str) -> None:
        for rows in vars(parsed).values():
            for row in rows:
                row["matchId"] = match_id

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _insert_one(
        self,
        table: str,
        cols: tuple[str, ...],
        items: list[dict[str, Any]],
        run_id: UUID,
    ) -> None:
        if not items:
            return
        try:
            await asyncio.to_thread(
                persist_data, table, cols, items, run_id, self.batch_size
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
        specs: tuple[TableSpec, ...],
        parsed: Any,
        buffers: dict[str, list[dict[str, Any]]],
        run_id: UUID,
    ) -> None:
        for spec in specs:
            items = list(spec.getter(parsed))
            if not items:
                continue
            buffers[spec.table].extend(items)
            if len(buffers[spec.table]) >= self.batch_size:
                await self._flush_table_buffer(spec.table, buffers, run_id)

    async def _flush_table_buffer(
        self,
        table: str,
        buffers: dict[str, list[dict[str, Any]]],
        run_id: UUID,
    ) -> None:
        items = buffers.get(table)
        if not items:
            return
        cols = self._table_columns[table]
        buffers[table] = []
        await self._insert_one(table, cols, items, run_id)

    async def _flush_all_buffers(
        self,
        buffers: dict[str, list[dict[str, Any]]],
        run_id: UUID,
    ) -> None:
        for table in tuple(buffers.keys()):
            await self._flush_table_buffer(table, buffers, run_id)
