from __future__ import annotations

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import zstandard as zstd

from app.core.config.constants.paths import (
    MATCH_DATA_MATCH_IDS_DIR,
)
from app.infrastructure.files.utils import atomic_outputs
from app.services.riot_api_client.base import RiotAPI, get_riot_api
from app.services.riot_api_client.match_data import (
    stream_non_timeline_data,
    stream_timeline_data,
)
from app.worker.pipelines.orchestrator import (
    Collector,
    Loader,
    Orchestrator,
    Saver,
    SaveSpec,
)


@dataclass(frozen=True)
class MatchDataCollectorState:
    matchids: list[str]


class MatchDataOrchestrator(Orchestrator):
    def __init__(
        self,
        *,
        loader: Loader,
        non_timeline_collector: Collector,
        timeline_collector: Collector,
        saver: Saver,
        out_matchids: Path,
        out_non_timeline: Path,
        out_timeline: Path,
    ) -> None:
        super().__init__(loader, non_timeline_collector, saver)
        self.non_timeline_collector = non_timeline_collector
        self.timeline_collector = timeline_collector
        self.out_matchids = out_matchids
        self.out_non_timeline = out_non_timeline
        self.out_timeline = out_timeline

    async def run(self) -> None:
        state: MatchDataCollectorState = self.loader.load()

        non_timeline_stream: AsyncIterator[dict[str, Any]] = (
            self.non_timeline_collector.collect(state)
        )
        timeline_stream: AsyncIterator[dict[str, Any]] = (
            self.timeline_collector.collect(state)
        )

        async def save_matchids(path: Path) -> None:
            await _write_matchids_txt_zst(path, state.matchids)

        async def save_non_timeline(path: Path) -> None:
            await _write_non_timeline_jsonl_zst(path, non_timeline_stream)

        async def save_timeline(path: Path) -> None:
            await _write_timeline_jsonl_zst(path, timeline_stream)

        await self.saver.save(
            SaveSpec(out_path=self.out_matchids, save=save_matchids),
            SaveSpec(out_path=self.out_non_timeline, save=save_non_timeline),
            SaveSpec(out_path=self.out_timeline, save=save_timeline),
        )


class MatchDataLoader(Loader):
    def __init__(self, match_ids_dir: Path, *, max_workers: int = 16) -> None:
        self.match_ids_dir = match_ids_dir
        self.max_workers = max_workers

    def load(self) -> MatchDataCollectorState:
        paths = list(self.match_ids_dir.glob("part-*.txt.zst"))
        matchids = list(iter_lines_from_many_zst(paths, max_workers=self.max_workers))
        return MatchDataCollectorState(matchids=matchids)


class MatchDataNonTimelineCollector(Collector):
    def __init__(self, riot_api: RiotAPI) -> None:
        self.riot_api = riot_api

    async def collect(self, state: MatchDataCollectorState) -> AsyncIterator[Any]:
        async for non_timeline in stream_non_timeline_data(
            state, riot_api=self.riot_api
        ):
            yield non_timeline


class MatchDataNonTimelineParser:
    def parse(self):
        pass


class MatchDataTimelineParser:
    def parse(self):
        pass
    

"""

"""


class MatchDataTimelineCollector(Collector):
    def __init__(self, riot_api: RiotAPI) -> None:
        self.riot_api = riot_api

    async def collect(self, state: MatchDataCollectorState) -> AsyncIterator[Any]:
        async for non_timeline in stream_timeline_data(state, riot_api=self.riot_api):
            yield non_timeline


def _read_matchids_txt_zst(path: Path) -> list[str]:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8")
            return [line.rstrip("\n") for line in text]


def iter_lines_from_many_zst(paths: list[Path], *, max_workers: int = 16):
    if not paths:
        return iter(())

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_read_matchids_txt_zst, p) for p in paths]
        for fut in as_completed(futures):
            yield from fut.result()


async def _write_matchids_txt_zst(path: Path, matchids: list[str]) -> None:
    cctx = zstd.ZstdCompressor(level=3)
    with path.open("wb") as fh:
        with cctx.stream_writer(fh) as writer:
            text = io.TextIOWrapper(writer, encoding="utf-8")
            for mid in matchids:
                text.write(mid)
                text.write("\n")
            text.flush()


def _write_non_timeline_jsonl_zst(
    path: Path,
    data: AsyncIterator[dict[str, Any]],
) -> None:
    pass


def _write_timeline_jsonl_zst(
    path: Path,
    data: AsyncIterator[dict[str, Any]],
) -> None:
    pass


class MatchDataSaver(Saver):
    async def save(self, *specs: SaveSpec) -> None:
        finals = [spec.out_path for spec in specs]

        async with atomic_outputs(*finals) as tmp_paths:
            for spec, tmp in zip(specs, tmp_paths):
                await spec.save(tmp)


if __name__ == "__main__":
    riot_api: RiotAPI = get_riot_api()

    loader = MatchDataLoader(match_ids_dir=MATCH_DATA_MATCH_IDS_DIR)

    non_timeline_collector = MatchDataNonTimelineCollector(riot_api=riot_api)
    timeline_collector = MatchDataTimelineCollector(riot_api=riot_api)

    saver = MatchDataSaver()

    orchestrator = MatchDataOrchestrator(
        loader=loader,
        non_timeline_collector=non_timeline_collector,
        timeline_collector=timeline_collector,
        saver=saver,
        out_matchids=Path("matchids.txt.zst"),
        out_non_timeline=Path("non_timeline.jsonl.zst"),
        out_timeline=Path("timeline.jsonl.zst"),
    )

    asyncio.run(orchestrator.run())
