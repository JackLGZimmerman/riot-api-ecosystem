from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from operator import attrgetter
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
)
from uuid import UUID, uuid4

from tenacity import before_sleep_log, retry, stop_never, wait_exponential

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
from database.clickhouse.operations.matchdata import (
    delete_by_run_id,
    delete_match_ids,
    insert_match_ids,
    persist_data,
)
from database.clickhouse.operations.matchids import load_matchids

logger = logging.getLogger(__name__)


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

TRaw = TypeVar("TRaw", contravariant=True)
TParsed = TypeVar("TParsed", covariant=True)

StreamName: TypeAlias = Literal["non_timeline", "timeline"]


@dataclass(frozen=True)
class StreamItem:
    stream: StreamName
    raw: Any


@dataclass(frozen=True)
class _Done:
    stream: StreamName


QueueMsg: TypeAlias = StreamItem | _Done


class Parser(Protocol[TRaw, TParsed]):
    def run(self, raw: TRaw) -> TParsed: ...


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
        self.pipeline = "match_data"
        self.non_timeline_collector = non_timeline_collector
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
        ctx = OrchestrationContext(
            ts=int(time.time()), run_id=uuid4(), pipeline=self.pipeline
        )

        state: MatchDataCollectorState = self.loader.load(ctx)

        non_timeline_raw = self.non_timeline_collector.collect(state, ctx)
        timeline_raw = self.timeline_collector.collect(state, ctx)

        items = self.combine_streams(non_timeline_raw, timeline_raw)
        await self.saver.save(items, state, ctx)


class MatchDataLoader(Loader):
    def __init__(self, max_workers: int = 16) -> None:
        self.max_workers = max_workers

    def load(self, ctx: OrchestrationContext) -> MatchDataCollectorState:
        matchids: list[str] = load_matchids()
        return MatchDataCollectorState(matchids=matchids)


class MatchDataNonTimelineCollector(Collector):
    def __init__(self, riot_api: RiotAPI) -> None:
        self.riot_api = riot_api

    async def collect(
        self, state: MatchDataCollectorState, ctx: OrchestrationContext
    ) -> AsyncIterator[dict[str, Any]]:
        async for raw in stream_non_timeline_data(
            state.matchids, riot_api=self.riot_api
        ):
            yield raw


class MatchDataTimelineCollector(Collector):
    def __init__(self, riot_api: RiotAPI) -> None:
        self.riot_api = riot_api

    async def collect(
        self, state: MatchDataCollectorState, ctx: OrchestrationContext
    ) -> AsyncIterator[dict[str, Any]]:
        async for raw in stream_timeline_data(state.matchids, riot_api=self.riot_api):
            yield raw


class MatchDataSaver(Saver):
    def __init__(self, *, non_timeline_parser: Parser, timeline_parser: Parser) -> None:
        self.non_timeline_parser = non_timeline_parser
        self.timeline_parser = timeline_parser

        self.nt_small = 5_000
        self.nt_medium = 20_000
        self.tl_events = 80_000
        self.tl_damage = 150_000

    async def save(
        self,
        items: AsyncIterator[Any],
        state: MatchDataCollectorState,
        ctx: OrchestrationContext,
    ) -> None:
        try:
            async for item_any in items:
                item: StreamItem = item_any

                if item.stream == "non_timeline":
                    nt: NonTimelineTables = await self._parse_non_timeline(item.raw)
                    await self._persist_non_timeline(nt, ctx.run_id)

                elif item.stream == "timeline":
                    tl: TimelineTables = await self._parse_timeline(item.raw)
                    await self._persist_timeline(tl, ctx.run_id)

                else:
                    raise ValueError(f"Unknown stream: {item.stream!r}")

        except Exception:
            await self.delete_failed_non_timeline(ctx.run_id)
            await self.delete_failed_timeline(ctx.run_id)
            await self._delete_match_ids(ctx.run_id)
            raise

        finally:
            await asyncio.to_thread(insert_match_ids, state.matchids, ctx.run_id)

    async def _parse_non_timeline(self, raw_data) -> NonTimelineTables:
        return await asyncio.to_thread(self.non_timeline_parser.run, raw_data)

    async def _parse_timeline(self, raw_data) -> TimelineTables:
        return await asyncio.to_thread(self.timeline_parser.run, raw_data)

    async def delete_failed_non_timeline(self, run_id: UUID) -> None:
        await self._run_deletes(NON_TIMELINE_DELETE_TABLES, run_id)

    async def delete_failed_timeline(self, run_id: UUID) -> None:
        await self._run_deletes(TIMELINE_DELETE_TABLES, run_id)

    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=1, min=1, max=300),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _delete_one(self, table: str, run_id: UUID) -> None:
        try:
            await asyncio.to_thread(delete_by_run_id, table, run_id)
        except Exception as e:
            logger.exception("Error deleting from %s run_id=%s: %s", table, run_id, e)
            raise

    async def _run_deletes(self, tables: tuple[str, ...], run_id: UUID) -> None:
        for table in tables:
            await self._delete_one(table, run_id)

    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _delete_match_ids(self, run_id: UUID) -> None:
        try:
            await asyncio.to_thread(delete_match_ids, run_id)
        except Exception as e:
            logger.exception(
                "Error deleting from game_data.matchdata_matchids run_id=%s: %s",
                run_id,
                e,
            )
            raise

    async def _persist_non_timeline(self, t: NonTimelineTables, run_id: UUID) -> None:
        await self._run_inserts(
            NON_TIMELINE_INSERTS, t, run_id, batch_size=self.nt_small
        )

    async def _persist_timeline(self, t: TimelineTables, run_id: UUID) -> None:
        await self._run_inserts(TIMELINE_INSERTS, t, run_id, batch_size=self.nt_medium)

    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _insert_one(
        self, table: str, cols, items, run_id: UUID, batch_size: int
    ) -> None:
        if not items:
            return
        try:
            await asyncio.to_thread(
                persist_data, table, cols, items, run_id, batch_size
            )
        except Exception as e:
            logger.exception("Error inserting into %s run_id=%s: %s", table, run_id, e)
            raise

    async def _run_inserts(self, specs, t, run_id: UUID, *, batch_size: int) -> None:
        for table, cols, getter in specs:
            items = getter(t)
            await self._insert_one(table, cols, items, run_id, batch_size)


if __name__ == "__main__":
    riot_api: RiotAPI = get_riot_api()

    loader = MatchDataLoader()
    non_timeline_collector = MatchDataNonTimelineCollector(riot_api=riot_api)
    timeline_collector = MatchDataTimelineCollector(riot_api=riot_api)

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

    asyncio.run(orchestrator.run())
