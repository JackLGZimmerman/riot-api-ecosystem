from __future__ import annotations

import asyncio
import io
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable

import zstandard as zstd

from app.core.config.constants import (
    ENDPOINTS,
    QUEUE_TYPE_TO_QUEUE_CODE,
    REGION_TO_CONTINENT,
)
from app.core.config.constants.paths import (
    MATCH_IDS_DATA_DIR,
    PLAYER_INFO,
    PUUIDS_FOR_MATCH_IDS,
    PUUIDS_FOR_MATCH_IDS_CHECKPOINT,
)
from app.infrastructure.files.utils import atomic_outputs
from app.models import MinifiedLeagueEntryDTO
from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.match_ids import (
    stream_match_ids,
)
from app.services.riot_api_client.utils import PlayerCrawlState
from app.worker.pipelines.orchestrator import (
    Collector,
    Loader,
    OrchestrationContext,
    Orchestrator,
    Saver,
    SaveSpec,
)

orchestration_ctx: ContextVar[OrchestrationContext] = ContextVar("orchestration_ctx")

MATCHID_BUFFER = 200_000


@dataclass(frozen=True)
class MatchCollectorInput:
    puuid_ts: int
    collected_puuids: frozenset[str]
    players: list[MinifiedLeagueEntryDTO]


@dataclass(frozen=True)
class MatchIDCollectorState:
    initial_states: list[PlayerCrawlState]
    ts: int
    collected_puuids: frozenset[str]
    current_player_puuids: frozenset[str]


def build_initial_player_states(
    state: MatchCollectorInput, *, ts: int
) -> list[PlayerCrawlState]:
    end_time = ts

    template = str(ENDPOINTS["match"]["by_puuid"])

    states: list[PlayerCrawlState] = []

    for player in state.players:
        puuid = player.puuid
        continent = REGION_TO_CONTINENT[player.region]
        queue = QUEUE_TYPE_TO_QUEUE_CODE[player.queueType]
        start_time = (
            state.puuid_ts
            if puuid in state.collected_puuids and state.puuid_ts > 0
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
                queue_type=player.queueType,
                continent=continent,
                start_time=start_time,
                next_page_start=0,
                end_time=end_time,
                base_url=base_url,
            )
        )

    return states


def read_checkpoint_ts(path: Path) -> int:
    if not path.exists():
        return -1
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("ts", -1))


def write_checkpoint_ts(path: Path, ts: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ts": ts}, separators=(",", ":")), encoding="utf-8")


def read_players_jsonl_zst(path: Path) -> list[MinifiedLeagueEntryDTO]:
    if not path.exists():
        return []

    dctx = zstd.ZstdDecompressor()
    out: list[MinifiedLeagueEntryDTO] = []

    with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
        with io.TextIOWrapper(reader, encoding="utf-8", newline="") as text:
            for line in text:
                line = line.strip()
                if not line:
                    continue
                out.append(MinifiedLeagueEntryDTO.model_validate(json.loads(line)))
    return out


def read_puuids_txt_zst(path: Path) -> set[str]:
    if not path.exists():
        return set()

    puuids: set[str] = set()
    dctx = zstd.ZstdDecompressor()

    with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text:
            puuid = line.strip()
            if puuid:
                puuids.add(puuid)

    return puuids


def write_match_ids_txt_zst(file_path: Path, match_ids: Iterable[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    cctx = zstd.ZstdCompressor()
    with file_path.open("wb") as fb, cctx.stream_writer(fb) as zstream:
        with io.TextIOWrapper(zstream, encoding="utf-8", newline="") as text:
            for match_id in match_ids:
                text.write(match_id + "\n")


def write_puuids_txt_zst(out_path: Path, puuids: Iterable[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wctx = zstd.ZstdCompressor()
    with out_path.open("wb") as fb, wctx.stream_writer(fb) as stream:
        with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
            for puuid in puuids:
                text.write(puuid + "\n")


async def buffered_write_shards(
    items: AsyncIterator[list[str]],
    *,
    out_dir: Path,
    shard_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    buf: list[str] = []
    cursor = 0

    async for batch in items:
        if not batch:
            continue

        buf.extend(batch)

        while (len(buf) - cursor) >= shard_size:
            chunk = buf[cursor : cursor + shard_size]
            cursor += shard_size

            shard_path = out_dir / f"part-{shard_idx:06d}.txt.zst"
            await asyncio.to_thread(write_match_ids_txt_zst, shard_path, chunk)
            shard_idx += 1

            if cursor > 0 and cursor >= (len(buf) // 2):
                buf = buf[cursor:]
                cursor = 0

    if (len(buf) - cursor) > 0:
        shard_path = out_dir / f"part-{shard_idx:06d}.txt.zst"
        await asyncio.to_thread(write_match_ids_txt_zst, shard_path, buf[cursor:])


class MatchIDLoader:
    def __init__(
        self,
        *,
        all_player_details_path: Path,
        collected_puuids_path: Path,
        collected_puuids_checkpoint_path: Path,
        read_players_jsonl_zst: Callable[[Path], list[MinifiedLeagueEntryDTO]],
    ):
        self.all_player_details_path = all_player_details_path
        self.collected_puuids_path = collected_puuids_path
        self.collected_puuids_checkpoint_path = collected_puuids_checkpoint_path
        self._read_players_jsonl_zst = read_players_jsonl_zst

    def load(self) -> MatchIDCollectorState:
        ctx = orchestration_ctx.get()

        puuid_ts = read_checkpoint_ts(self.collected_puuids_checkpoint_path)
        prev_puuids = read_puuids_txt_zst(self.collected_puuids_path)
        players = self._read_players_jsonl_zst(self.all_player_details_path)

        current_puuids = frozenset(p.puuid for p in players)

        input_state = MatchCollectorInput(
            puuid_ts=puuid_ts,
            collected_puuids=frozenset(prev_puuids),
            players=players,
        )

        initial_states = build_initial_player_states(
            input_state,
            ts=ctx.orchestration_start_time,
        )

        return MatchIDCollectorState(
            initial_states=initial_states,
            ts=ctx.orchestration_start_time,
            collected_puuids=frozenset(prev_puuids),
            current_player_puuids=current_puuids,
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
        finals = [spec.out_path for spec in specs]

        async with atomic_outputs(*finals) as tmp_paths:
            for spec, tmp in zip(specs, tmp_paths):
                await spec.save(tmp)


class MatchIDOrchestrator(Orchestrator):
    def __init__(
        self,
        loader: Loader,
        collector: Collector,
        saver: Saver,
        *,
        collected_match_ids_dir: Path,
        collected_puuids_path: Path,
        checkpoint_path: Path,
    ):
        super().__init__(loader, collector, saver)
        self.collected_match_ids_dir = collected_match_ids_dir
        self.collected_puuids_path = collected_puuids_path
        self.checkpoint_path = checkpoint_path

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
        token = orchestration_ctx.set(
            OrchestrationContext(orchestration_start_time=int(time.time()))
        )
        try:
            state: MatchIDCollectorState = self.loader.load()
            match_ids_stream = self.collector.collect(state)
            match_ids_stream = self._dedupe_async(match_ids_stream)

            self.collected_puuids_path.parent.mkdir(parents=True, exist_ok=True)
            self.collected_match_ids_dir.mkdir(parents=True, exist_ok=True)

            async def save_match_ids(tmp_path: Path) -> None:
                tmp_path.mkdir(parents=True, exist_ok=True)

                await buffered_write_shards(
                    match_ids_stream,
                    out_dir=tmp_path,
                    shard_size=MATCHID_BUFFER,
                )

            async def save_puuids(tmp_path: Path) -> None:
                await asyncio.to_thread(
                    write_puuids_txt_zst, tmp_path, state.current_player_puuids
                )

            async def save_checkpoint(tmp_path: Path) -> None:
                await asyncio.to_thread(write_checkpoint_ts, tmp_path, state.ts)

            await self.saver.save(
                SaveSpec(out_path=self.collected_match_ids_dir, save=save_match_ids),
                SaveSpec(out_path=self.collected_puuids_path, save=save_puuids),
                SaveSpec(out_path=self.checkpoint_path, save=save_checkpoint),
            )
        finally:
            orchestration_ctx.reset(token)


if __name__ == "__main__":
    all_player_details_path = PLAYER_INFO
    collected_puuids_path = PUUIDS_FOR_MATCH_IDS
    collected_match_ids_dir = MATCH_IDS_DATA_DIR
    riot_api: RiotAPI = get_riot_api()
    loader = MatchIDLoader(
        all_player_details_path=PLAYER_INFO,
        collected_puuids_path=PUUIDS_FOR_MATCH_IDS,
        collected_puuids_checkpoint_path=PUUIDS_FOR_MATCH_IDS_CHECKPOINT,
        read_players_jsonl_zst=read_players_jsonl_zst,
    )

    collector = MatchIDCollector(riot_api)
    saver = MatchIDSaver()

    orchestrator = MatchIDOrchestrator(
        loader=loader,
        collector=collector,
        saver=saver,
        collected_match_ids_dir=collected_match_ids_dir,
        collected_puuids_path=collected_puuids_path,
        checkpoint_path=PUUIDS_FOR_MATCH_IDS_CHECKPOINT,
    )

    asyncio.run(orchestrator.run())
