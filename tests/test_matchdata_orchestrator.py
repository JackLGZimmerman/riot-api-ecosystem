from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID

from app.services.riot_api_client.match_data import MatchFetchResult
from app.worker.pipelines.matchdata_orchestrator import (
    MatchDataCollectorState,
    MatchDataSaver,
    StreamItem,
)
from app.worker.pipelines.orchestrator import OrchestrationContext


class FakeParser:
    def run(self, data):
        return SimpleNamespace(rows=[{}])


class RecordingSaver(MatchDataSaver):
    def __init__(self) -> None:
        super().__init__(
            non_timeline_parser=FakeParser(),
            timeline_parser=FakeParser(),
        )
        self.deleted: list[list[str]] = []
        self.finished: list[list[str]] = []
        self.source_deleted: list[list[str]] = []

    async def delete_failed_matchids(self, match_ids: list[str]) -> None:
        self.deleted.append(list(match_ids))

    async def delete_source_matchids(self, match_ids: list[str]) -> None:
        self.source_deleted.append(list(match_ids))

    async def mark_finished_matchids(self, match_ids: list[str]) -> None:
        self.finished.append(list(match_ids))

    async def _buffer_inserts(self, specs, parsed, buffers, run_id) -> None:
        return None

    async def _flush_all_buffers(self, buffers, run_id) -> None:
        return None


def _ctx() -> OrchestrationContext:
    return OrchestrationContext(
        ts=123,
        run_id=UUID("11111111-1111-1111-1111-111111111111"),
        pipeline="match_data",
    )


async def _items(*items):
    for item in items:
        yield item


def test_matchdata_requeues_single_stream_success() -> None:
    saver = RecordingSaver()
    state = MatchDataCollectorState(matchids=["NA1_1"])

    asyncio.run(
        saver.save(
            _items(
                StreamItem(
                    "non_timeline",
                    MatchFetchResult("NA1_1", {"metadata": {}}, 200),
                )
            ),
            state,
            _ctx(),
        )
    )

    assert saver.deleted == [["NA1_1"], ["NA1_1"]]
    assert saver.finished == [[]]
    assert saver.source_deleted == []


def test_matchdata_retires_success_plus_terminal_without_keeping_partial_rows() -> None:
    saver = RecordingSaver()
    state = MatchDataCollectorState(matchids=["NA1_1"])

    asyncio.run(
        saver.save(
            _items(
                StreamItem(
                    "non_timeline",
                    MatchFetchResult("NA1_1", {"metadata": {}}, 200),
                ),
                StreamItem(
                    "timeline",
                    MatchFetchResult("NA1_1", None, 404),
                ),
            ),
            state,
            _ctx(),
        )
    )

    assert saver.deleted == [["NA1_1"], ["NA1_1"]]
    assert saver.source_deleted == [["NA1_1"]]
    assert saver.finished == [["NA1_1"]]
