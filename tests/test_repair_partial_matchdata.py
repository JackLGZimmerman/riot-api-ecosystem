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


def test_metadata_repair_tables_are_limited_to_small_sources() -> None:
    assert repair.METADATA_REPAIR_TABLES == (
        repair.METADATA_TABLE,
        repair.INFO_TABLE,
        repair.TIMELINE_END_TABLE,
        repair.PARTICIPANT_STATS_TABLE,
        repair.QUEUE_TABLE,
    )


def test_classify_presence_splits_metadata_only_and_stream_partial() -> None:
    classes = repair._classify_presence(
        ["EUW1_1", "NA1_1", "KR_1", "BR1_1", "EUW1_1"],
        info_matchids={"EUW1_1", "NA1_1", "KR_1", "BR1_1"},
        timeline_matchids={"EUW1_1", "KR_1", "BR1_1"},
        metadata_matchids={"KR_1"},
        valid_participant_matchids={"EUW1_1"},
    )

    assert classes.metadata_only == ("EUW1_1",)
    assert classes.stream_partial == ("NA1_1",)
    assert classes.already_complete == ("KR_1",)
    assert classes.metadata_blocked == ("BR1_1",)
    assert classes.no_action == ()


def test_backup_table_name_rejects_unsafe_prefix() -> None:
    with pytest.raises(ValueError, match="Unsafe backup prefix"):
        repair._backup_table_name("game_data.info", "bad-prefix")


def test_empty_scoped_repair_run_does_not_fall_back_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = repair.parse_args(
        [
            "--repair-run-id",
            "11111111-1111-1111-1111-111111111111",
            "--allow-full-requeue",
        ]
    )
    monkeypatch.setattr(
        repair, "_load_queue_matchids_for_run", lambda _client, _run_id: []
    )
    monkeypatch.setattr(
        repair,
        "_load_auto_repair_matchids",
        lambda _client: pytest.fail("explicit empty scope must not auto-repair"),
    )

    assert repair._candidate_matchids(object(), args) == []


class _QueueClient:
    inserted: list[tuple[UUID, str]]

    def __init__(self, existing: list[str]) -> None:
        self.existing = existing
        self.inserted = []

    def query(self, query: str, parameters: dict[str, object] | None = None):
        assert "game_data.matchdata_matchids" in query
        return type(
            "Result", (), {"result_rows": [(matchid,) for matchid in self.existing]}
        )

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


class _RepairClient:
    commands: list[tuple[str, dict[str, object] | None]]
    inserted: list[tuple[str, list[tuple[UUID, str]], tuple[str, str]]]

    def __init__(
        self,
        *,
        info: set[str],
        timeline: set[str],
        metadata: set[str] | None = None,
        valid_participants: set[str] | None = None,
        incomplete_participants: set[str] | None = None,
        duplicate_participants: set[str] | None = None,
        bad_dataversions: list[int] | None = None,
        metadata_after_counts: dict[str, int] | None = None,
    ) -> None:
        self.info = info
        self.timeline = timeline
        self.metadata = metadata or set()
        self.valid_participants = valid_participants or set()
        self.incomplete_participants = incomplete_participants or set()
        self.duplicate_participants = duplicate_participants or set()
        self.bad_dataversions = bad_dataversions or []
        self.metadata_after_counts = metadata_after_counts
        self.commands = []
        self.inserted = []

    def query(self, query: str, parameters: dict[str, object] | None = None):
        sql = " ".join(query.split())
        matchids = list((parameters or {}).get("matchids", []))

        if "dataversion != 2" in sql:
            return _rows([(len(self.bad_dataversions), self.bad_dataversions)])
        if "unique_participant_ids" in sql:
            rows = []
            for matchid in matchids:
                if matchid in self.valid_participants:
                    rows.append((matchid, 10, 10))
                elif matchid in self.incomplete_participants:
                    rows.append((matchid, 9, 9))
                elif matchid in self.duplicate_participants:
                    rows.append((matchid, 10, 9))
            return _rows(rows)
        if "SELECT matchid, count()" in sql and repair.METADATA_TABLE in sql:
            counts = self.metadata_after_counts
            if counts is None:
                counts = {matchid: 1 for matchid in matchids}
            return _rows([(matchid, count) for matchid, count in counts.items()])
        if "SELECT DISTINCT matchid" in sql:
            if repair.INFO_TABLE in sql:
                return _rows(
                    [(matchid,) for matchid in matchids if matchid in self.info]
                )
            if repair.TIMELINE_END_TABLE in sql:
                return _rows(
                    [(matchid,) for matchid in matchids if matchid in self.timeline]
                )
            if repair.METADATA_TABLE in sql:
                return _rows(
                    [(matchid,) for matchid in matchids if matchid in self.metadata]
                )
            if repair.QUEUE_TABLE in sql:
                return _rows([])
        if "SELECT count()" in sql:
            return _rows([(self._backup_count(sql, matchids),)])
        raise AssertionError(f"Unexpected query: {sql}")

    def command(self, query: str, parameters: dict[str, object] | None = None) -> None:
        self.commands.append((" ".join(query.split()), parameters))

    def insert(
        self,
        table: str,
        data: list[tuple[UUID, str]],
        column_names: tuple[str, str],
    ) -> None:
        self.inserted.append((table, data, column_names))

    def _backup_count(self, sql: str, matchids: list[str]) -> int:
        if repair.METADATA_TABLE in sql or "_metadata" in sql:
            return len([matchid for matchid in matchids if matchid in self.metadata])
        if repair.PARTICIPANT_STATS_TABLE in sql or "_participant_stats" in sql:
            return 10 * len([m for m in matchids if m in self.valid_participants])
        if repair.INFO_TABLE in sql or "_info" in sql:
            return len([matchid for matchid in matchids if matchid in self.info])
        if repair.TIMELINE_END_TABLE in sql or "_tl_game_end" in sql:
            return len([matchid for matchid in matchids if matchid in self.timeline])
        if repair.QUEUE_TABLE in sql or "_matchdata_matchids" in sql:
            return len(matchids)
        return 0


def _rows(rows: list[tuple[object, ...]]):
    return type("Result", (), {"result_rows": rows})


def test_metadata_only_apply_backs_up_before_insert_and_clears_queue() -> None:
    client = _RepairClient(
        info={"EUW1_1"},
        timeline={"EUW1_1"},
        valid_participants={"EUW1_1"},
    )

    backups, repair_run_id = repair._apply_metadata_only_repair(
        client,
        matchids=["EUW1_1"],
        backup_prefix="metadata_test",
        apply=True,
    )

    assert repair_run_id is not None
    assert [backup.source_table for backup in backups] == list(
        repair.METADATA_REPAIR_TABLES
    )
    commands = [command for command, _parameters in client.commands]
    metadata_insert = next(
        i
        for i, command in enumerate(commands)
        if command.startswith("INSERT INTO game_data.metadata")
    )
    backup_creates = [
        i
        for i, command in enumerate(commands)
        if command.startswith("CREATE TABLE game_data_repair.metadata_test_")
    ]
    queue_delete = next(
        i
        for i, command in enumerate(commands)
        if command.startswith("ALTER TABLE game_data.matchdata_matchids")
    )
    assert len(backup_creates) == len(repair.METADATA_REPAIR_TABLES)
    assert max(backup_creates) < metadata_insert < queue_delete


def test_metadata_only_apply_refuses_non_2_dataversion() -> None:
    client = _RepairClient(
        info={"EUW1_1"},
        timeline={"EUW1_1"},
        valid_participants={"EUW1_1"},
        bad_dataversions=[1],
    )

    with pytest.raises(RuntimeError, match="dataversion=2"):
        repair._apply_metadata_only_repair(
            client,
            matchids=["EUW1_1"],
            backup_prefix="metadata_test",
            apply=True,
        )

    assert client.commands == []


def test_metadata_only_apply_refuses_duplicate_missing_or_incomplete_participants() -> (
    None
):
    duplicate_client = _RepairClient(
        info={"EUW1_1"},
        timeline={"EUW1_1"},
        duplicate_participants={"EUW1_1"},
    )
    missing_client = _RepairClient(info={"NA1_1"}, timeline={"NA1_1"})
    incomplete_client = _RepairClient(
        info={"KR_1"},
        timeline={"KR_1"},
        incomplete_participants={"KR_1"},
    )

    for client, matchid in (
        (duplicate_client, "EUW1_1"),
        (missing_client, "NA1_1"),
        (incomplete_client, "KR_1"),
    ):
        with pytest.raises(RuntimeError, match="target validation failed"):
            repair._apply_metadata_only_repair(
                client,
                matchids=[matchid],
                backup_prefix="metadata_test",
                apply=True,
            )
        assert client.commands == []


def test_insert_missing_metadata_is_guarded_to_missing_rows() -> None:
    client = _RepairClient(
        info={"EUW1_1"},
        timeline={"EUW1_1"},
        valid_participants={"EUW1_1"},
    )

    repair._insert_missing_metadata(
        client,
        matchids=["EUW1_1"],
        repair_run_id=UUID("11111111-1111-1111-1111-111111111111"),
    )

    command, parameters = client.commands[0]
    assert command.startswith("INSERT INTO game_data.metadata")
    assert "toUInt8(2) AS dataversion" in command
    assert "groupArray((participantid, puuid))" in command
    assert "matchid NOT IN" in command
    assert parameters == {
        "run_id": "11111111-1111-1111-1111-111111111111",
        "matchids": ["EUW1_1"],
    }


def test_metadata_only_apply_keeps_queue_if_post_validation_fails() -> None:
    client = _RepairClient(
        info={"EUW1_1"},
        timeline={"EUW1_1"},
        valid_participants={"EUW1_1"},
        metadata_after_counts={"EUW1_1": 0},
    )

    with pytest.raises(RuntimeError, match="post-validation failed"):
        repair._apply_metadata_only_repair(
            client,
            matchids=["EUW1_1"],
            backup_prefix="metadata_test",
            apply=True,
        )

    commands = [command for command, _parameters in client.commands]
    assert any(
        command.startswith("INSERT INTO game_data.metadata") for command in commands
    )
    assert not any(
        command.startswith("ALTER TABLE game_data.matchdata_matchids")
        for command in commands
    )


def test_apply_does_not_full_requeue_stream_partial_without_allow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RepairClient(info={"NA1_1"}, timeline=set())
    monkeypatch.setattr(repair, "get_client", lambda: client)

    with pytest.raises(SystemExit, match="without --allow-full-requeue"):
        repair.main(["--apply", "--matchid", "NA1_1"])

    assert client.commands == []
    assert client.inserted == []


def test_dry_run_outputs_class_counts_and_estimated_action(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _RepairClient(
        info={"EUW1_1", "NA1_1"},
        timeline={"EUW1_1"},
        valid_participants={"EUW1_1"},
    )
    monkeypatch.setattr(repair, "get_client", lambda: client)

    repair.main(["--matchid", "EUW1_1", "--matchid", "NA1_1"])

    output = capsys.readouterr().out
    assert "DRY RUN: 2 matchdata repair candidates" in output
    assert "metadata_only=1" in output
    assert "stream_partial=1" in output
    assert "Estimated action:" in output
    assert "metadata_only=insert_metadata_and_clear_queue" in output
    assert "stream_partial=full_requeue_requires_allow_full_requeue" in output
