from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    AsyncIterable,
    Protocol,
    TypeVar,
    Literal,
    TypeAlias,
    Iterable,
    Awaitable,
    Generic,
    Callable,
)
import time
from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.match_data import (
    stream_non_timeline_data,
    stream_timeline_data,
)
from app.services.riot_api_client.parsers.non_timeline import (
    MatchDataNonTimelineParsingOrchestrator,
    NonTimelineTables,
)
from app.services.riot_api_client.parsers.timeline import (
    MatchDataTimelineParsingOrchestrator,
    TimelineTables,
)
from app.worker.pipelines.orchestrator import (
    Collector,
    Loader,
    Orchestrator,
    OrchestrationContext,
    Saver,
)
from database.clickhouse.operations.matchids import load_matchids
from uuid import uuid4, UUID

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
    def parse(self, raw: TRaw) -> TParsed: ...


T = TypeVar("T")


class SizeSink(Generic[T]):
    def __init__(
        self,
        *,
        name: str,
        insert_batch: Callable[[list[T]], Awaitable[None]],
        batch_rows: int,
    ) -> None:
        self.name = name
        self._insert_batch = insert_batch
        self._batch_rows = batch_rows
        self._buf: list[T] = []

    async def add_many(self, rows: Iterable[T]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return

        self._buf.extend(rows_list)

        if len(self._buf) >= self._batch_rows:
            await self.flush()

    async def flush(self) -> None:
        if not self._buf:
            return
        batch = self._buf
        self._buf = []
        await self._insert_batch(batch)

    async def flush_all(self) -> None:
        await self.flush()


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
        max_buffer: int = 50_000,
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
    def __init__(self, *, non_timeline_parser, timeline_parser) -> None:
        self.non_timeline_parser = non_timeline_parser
        self.timeline_parser = timeline_parser

        self.nt_small = 5_000
        self.nt_medium = 20_000
        self.tl_events = 80_000
        self.tl_damage = 150_000


    def _make_sinks(self, run_id: UUID) -> dict[str, SizeSink[Any]]:
        async def guarded(fn, rows):
            await asyncio.to_thread(fn, rows, run_id)

        return {
            "metadata": SizeSink(
                name="metadata",
                insert_batch=lambda rows: guarded(insert_metadata_batch, rows),
                batch_rows=self.nt_small,
            ),
            "game_info": SizeSink(
                name="game_info",
                insert_batch=lambda rows: guarded(insert_game_info_batch, rows),
                batch_rows=self.nt_small,
            ),
            "bans": SizeSink(
                name="bans",
                insert_batch=lambda rows: guarded(insert_bans_batch, rows),
                batch_rows=self.nt_medium,
            ),
            "feats": SizeSink(
                name="feats",
                insert_batch=lambda rows: guarded(insert_feats_batch, rows),
                batch_rows=self.nt_medium,
            ),
            "objectives": SizeSink(
                name="objectives",
                insert_batch=lambda rows: guarded(insert_objectives_batch, rows),
                batch_rows=self.nt_medium,
            ),
            "p_stats": SizeSink(
                name="p_stats",
                insert_batch=lambda rows: guarded(insert_p_stats_batch, rows),
                batch_rows=self.nt_medium,
            ),
            "p_challenges": SizeSink(
                name="p_challenges",
                insert_batch=lambda rows: guarded(insert_p_challenges_batch, rows),
                batch_rows=self.nt_medium,
            ),
            "p_perks": SizeSink(
                name="p_perks",
                insert_batch=lambda rows: guarded(insert_p_perks_batch, rows),
                batch_rows=self.nt_medium,
            ),
            # -------------------------
            # timeline sinks
            # -------------------------
            "tl_p_stats": SizeSink(
                name="tl_p_stats",
                insert_batch=lambda rows: guarded(insert_tl_p_stats_batch, rows),
                batch_rows=self.tl_events,
            ),
            "building_kill": SizeSink(
                name="building_kill",
                insert_batch=lambda rows: guarded(insert_building_kill_batch, rows),
                batch_rows=self.tl_events,
            ),
            "champion_kill": SizeSink(
                name="champion_kill",
                insert_batch=lambda rows: guarded(insert_champion_kill_batch, rows),
                batch_rows=self.tl_events,
            ),
            "champion_special_kill": SizeSink(
                name="champion_special_kill",
                insert_batch=lambda rows: guarded(
                    insert_champion_special_kill_batch, rows
                ),
                batch_rows=self.tl_events,
            ),
            "dragon_soul_given": SizeSink(
                name="dragon_soul_given",
                insert_batch=lambda rows: guarded(insert_dragon_soul_given_batch, rows),
                batch_rows=self.tl_events,
            ),
            "elite_monster_kill": SizeSink(
                name="elite_monster_kill",
                insert_batch=lambda rows: guarded(
                    insert_elite_monster_kill_batch, rows
                ),
                batch_rows=self.tl_events,
            ),
            "game_end": SizeSink(
                name="game_end",
                insert_batch=lambda rows: guarded(insert_game_end_batch, rows),
                batch_rows=self.tl_events,
            ),
            "item_destroyed": SizeSink(
                name="item_destroyed",
                insert_batch=lambda rows: guarded(insert_item_destroyed_batch, rows),
                batch_rows=self.tl_events,
            ),
            "item_purchased": SizeSink(
                name="item_purchased",
                insert_batch=lambda rows: guarded(insert_item_purchased_batch, rows),
                batch_rows=self.tl_events,
            ),
            "item_sold": SizeSink(
                name="item_sold",
                insert_batch=lambda rows: guarded(insert_item_sold_batch, rows),
                batch_rows=self.tl_events,
            ),
            "item_undo": SizeSink(
                name="item_undo",
                insert_batch=lambda rows: guarded(insert_item_undo_batch, rows),
                batch_rows=self.tl_events,
            ),
            "level_up": SizeSink(
                name="level_up",
                insert_batch=lambda rows: guarded(insert_level_up_batch, rows),
                batch_rows=self.tl_events,
            ),
            "pause_end": SizeSink(
                name="pause_end",
                insert_batch=lambda rows: guarded(insert_pause_end_batch, rows),
                batch_rows=self.tl_events,
            ),
            "skill_level_up": SizeSink(
                name="skill_level_up",
                insert_batch=lambda rows: guarded(insert_skill_level_up_batch, rows),
                batch_rows=self.tl_events,
            ),
            "turret_plate_destroyed": SizeSink(
                name="turret_plate_destroyed",
                insert_batch=lambda rows: guarded(
                    insert_turret_plate_destroyed_batch, rows
                ),
                batch_rows=self.tl_events,
            ),
            "ward_kill": SizeSink(
                name="ward_kill",
                insert_batch=lambda rows: guarded(insert_ward_kill_batch, rows),
                batch_rows=self.tl_events,
            ),
            "ward_placed": SizeSink(
                name="ward_placed",
                insert_batch=lambda rows: guarded(insert_ward_placed_batch, rows),
                batch_rows=self.tl_events,
            ),
            "ck_damage_dealt": SizeSink(
                name="ck_damage_dealt",
                insert_batch=lambda rows: guarded(
                    insert_ck_victim_damage_dealt_batch, rows
                ),
                batch_rows=self.tl_damage,
            ),
            "ck_damage_received": SizeSink(
                name="ck_damage_received",
                insert_batch=lambda rows: guarded(
                    insert_ck_victim_damage_received_batch, rows
                ),
                batch_rows=self.tl_damage,
            ),
        }

    async def save(
        self,
        items: AsyncIterator[Any],
        state: MatchDataCollectorState,
        ctx: OrchestrationContext,
    ) -> None:
        sinks = self._make_sinks(ctx.run_id)

        try:
            async for item_any in items:
                item: StreamItem = item_any

                if item.stream == "non_timeline":
                    nt: NonTimelineTables = await asyncio.to_thread(
                        self.non_timeline_parser.run, item.raw
                    )
                    await self._persist_non_timeline(nt, sinks)

                elif item.stream == "timeline":
                    tl: TimelineTables = await asyncio.to_thread(
                        self.timeline_parser.run, item.raw
                    )
                    await self._persist_timeline(tl, sinks)

                else:
                    raise ValueError(f"Unknown stream: {item.stream!r}")

            # flush remaining buffers at end
            await asyncio.gather(*(s.flush_all() for s in sinks.values()))

        except Exception:
            # no background writers exist, so rollback immediately
            await asyncio.to_thread(self.delete_failed_non_timeline, ctx.run_id)
            await asyncio.to_thread(self.delete_failed_timeline, ctx.run_id)
            raise

        finally:
            pass  # retention cleanup if needed

    async def delete_failed_non_timeline(self, run_id: UUID) -> None: ...

    async def delete_failed_timeline(self, run_id: UUID) -> None: ...

    async def _persist_non_timeline(
        self, t: NonTimelineTables, ctx: OrchestrationContext
    ) -> None:
        # If these are independent inserts, you *can* parallelize within this bundle:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(persist_metadata(t.metadata, ctx.run_id))
            tg.create_task(persist_game_info(t.game_info, ctx.run_id))

            tg.create_task(persist_bans(t.bans, ctx.run_id))
            tg.create_task(persist_feats(t.feats, ctx.run_id))
            tg.create_task(persist_objectives(t.objectives, ctx.run_id))
            tg.create_task(persist_p_stats(t.p_stats, ctx.run_id))
            tg.create_task(persist_p_challenges(t.p_challenges, ctx.run_id))
            tg.create_task(persist_p_perks(t.p_perks, ctx.run_id))

    async def _persist_timeline(
        self, t: TimelineTables, ctx: OrchestrationContext
    ) -> None:
        # If these are independent inserts, you *can* parallelize within this bundle:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(persist_tl_p_stats(t.participantStats, ctx.run_id))

            tg.create_task(persist_building_kill(t.buildingKill, ctx.run_id))
            tg.create_task(persist_champion_kill(t.championKill, ctx.run_id))
            tg.create_task(
                persist_champion_special_kill(t.championSpecialKill, ctx.run_id)
            )
            tg.create_task(persist_dragon_soul_given(t.dragonSoulGiven, ctx.run_id))
            tg.create_task(persist_elite_monster_kill(t.eliteMonsterKill, ctx.run_id))
            tg.create_task(persist_game_end(t.gameEnd, ctx.run_id))

            tg.create_task(persist_item_destroyed(t.itemDestroyed, ctx.run_id))
            tg.create_task(persist_item_purchased(t.itemPurchased, ctx.run_id))
            tg.create_task(persist_item_sold(t.itemSold, ctx.run_id))
            tg.create_task(persist_item_undo(t.itemUndo, ctx.run_id))

            tg.create_task(persist_level_up(t.levelUp, ctx.run_id))
            tg.create_task(persist_pause_end(t.pauseEnd, ctx.run_id))
            tg.create_task(persist_skill_level_up(t.skillLevelUp, ctx.run_id))

            tg.create_task(
                persist_turret_plate_destroyed(t.turretPlateDestroyed, ctx.run_id)
            )
            tg.create_task(persist_ward_kill(t.wardKill, ctx.run_id))
            tg.create_task(persist_ward_placed(t.wardPlaced, ctx.run_id))

            tg.create_task(
                persist_ck_victim_damage_dealt(
                    t.championKillVictimDamageDealt, ctx.run_id
                )
            )
            tg.create_task(
                persist_ck_victim_damage_received(
                    t.championKillVictimDamageReceived, ctx.run_id
                )
            )


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
