from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.services.riot_api_client.match_data import MatchFetchResult
from app.worker.pipelines.matchdata_orchestrator import (
    MatchDataCollectorState,
    MatchDataSaver,
    NON_TIMELINE_TABLE_SPECS,
    StreamItem,
    TIMELINE_TABLE_SPECS,
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
        self.stream_deleted: list[tuple[list[str], list[str]]] = []

    async def delete_failed_matchids(self, match_ids: list[str]) -> None:
        self.deleted.append(list(match_ids))

    async def delete_stream_matchids(self, specs, match_ids: list[str]) -> None:
        self.stream_deleted.append(([spec.table for spec in specs], list(match_ids)))

    async def delete_unanchored_residue(self, state: MatchDataCollectorState) -> None:
        return None

    async def delete_source_matchids(self, match_ids: list[str]) -> None:
        self.source_deleted.append(list(match_ids))

    async def mark_finished_matchids(self, match_ids: list[str]) -> None:
        self.finished.append(list(match_ids))

    async def _buffer_inserts(self, specs, parsed, buffers, run_id) -> None:
        return None

    async def _flush_all_buffers(self, buffers, run_id) -> None:
        return None


class FailingSaver(RecordingSaver):
    async def _buffer_inserts(self, specs, parsed, buffers, run_id) -> None:
        raise RuntimeError("insert failed")


class ResidueRecordingSaver(RecordingSaver):
    async def delete_unanchored_residue(self, state: MatchDataCollectorState) -> None:
        await MatchDataSaver.delete_unanchored_residue(self, state)


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

    assert saver.deleted == []
    assert saver.finished == []
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

    assert saver.deleted == [["NA1_1"]]
    assert saver.source_deleted == [["NA1_1"]]
    assert saver.finished == [["NA1_1"]]


def test_matchdata_finishes_when_missing_stream_arrives_for_anchored_match() -> None:
    saver = RecordingSaver()
    state = MatchDataCollectorState(
        matchids=["NA1_1"],
        non_timeline_matchids=[],
        timeline_matchids=["NA1_1"],
    )

    asyncio.run(
        saver.save(
            _items(
                StreamItem(
                    "timeline",
                    MatchFetchResult("NA1_1", {"frames": []}, 200),
                )
            ),
            state,
            _ctx(),
        )
    )

    assert saver.deleted == []
    assert saver.source_deleted == []
    assert saver.finished == [["NA1_1"]]


def test_matchdata_stream_anchors_flush_last() -> None:
    assert NON_TIMELINE_TABLE_SPECS[-1].table == "game_data.info"
    assert TIMELINE_TABLE_SPECS[-1].table == "game_data.tl_game_end"


def test_matchdata_exception_deletes_partial_rows() -> None:
    saver = FailingSaver()
    state = MatchDataCollectorState(matchids=["NA1_1"])

    with pytest.raises(RuntimeError, match="insert failed"):
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

    assert saver.deleted == [["NA1_1"]]
    assert saver.finished == []


def test_matchdata_deletes_unanchored_residue(monkeypatch) -> None:
    def fake_load_table_matchids(table, match_ids):
        return {"NA1_1"} if table == "game_data.metadata" else set()

    monkeypatch.setattr(
        "app.worker.pipelines.matchdata_orchestrator.load_table_matchids",
        fake_load_table_matchids,
    )
    saver = ResidueRecordingSaver()
    state = MatchDataCollectorState(
        matchids=["NA1_1"],
        non_timeline_matchids=["NA1_1"],
        timeline_matchids=["NA1_1"],
    )

    asyncio.run(saver.delete_unanchored_residue(state))

    assert saver.stream_deleted == [
        ([spec.table for spec in NON_TIMELINE_TABLE_SPECS], ["NA1_1"])
    ]
