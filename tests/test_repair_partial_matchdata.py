from __future__ import annotations

from uuid import UUID

import pytest

from app.worker.pipelines.matchdata_orchestrator import ALL_DELETE_TABLES
from scripts import repair_partial_matchdata as repair


def test_repair_source_tables_cover_delete_tables_and_lineage() -> None:
    assert repair.SOURCE_TABLES == (
        *ALL_DELETE_TABLES,
        "game_data.matchids",
        repair.QUEUE_TABLE,
    )


class _QueueClient:
    inserted: list[tuple[UUID, str]]

    def __init__(self, existing: list[str]) -> None:
        self.existing = existing
        self.inserted = []

    def query(self, query: str, parameters: dict[str, object] | None = None):
        assert "game_data.matchdata_matchids" in query
        return type("Result", (), {"result_rows": [(matchid,) for matchid in self.existing]})

    def insert(
        self,
        table: str,
        data: list[tuple[UUID, str]],
        column_names: tuple[str, str],
    ) -> None:
        assert table == repair.QUEUE_TABLE
        assert column_names == ("run_id", "matchid")
        self.inserted.extend(data)


def test_queue_missing_matchids_is_idempotent() -> None:
    client = _QueueClient(existing=["EUW1_1"])

    missing = repair._queue_missing_matchids(
        client,
        matchids=["EUW1_1", "NA1_1", "NA1_1"],
        apply=True,
    )

    assert missing == 1
    assert [matchid for _run_id, matchid in client.inserted] == ["NA1_1"]


class _BackupClient:
    commands: list[str]

    def __init__(self, counts: list[int]) -> None:
        self.counts = counts
        self.commands = []

    def query(self, query: str, parameters: dict[str, object] | None = None):
        assert parameters == {"matchids": ["EUW1_1"]}
        count = self.counts.pop(0)
        return type("Result", (), {"result_rows": [(count,)]})

    def command(self, query: str, parameters: dict[str, object] | None = None) -> None:
        self.commands.append(query)


def test_backup_source_table_refuses_count_mismatch() -> None:
    client = _BackupClient(counts=[2, 1])

    with pytest.raises(RuntimeError, match="Backup row count mismatch"):
        repair._backup_source_table(
            client,
            source_table="game_data.info",
            backup_prefix="partial_test",
            matchids=["EUW1_1"],
            apply=True,
        )

    assert "CREATE DATABASE IF NOT EXISTS game_data_repair" in client.commands[0]
    assert "CREATE TABLE game_data_repair.partial_test_info" in client.commands[1]
    assert "INSERT INTO game_data_repair.partial_test_info" in client.commands[2]
