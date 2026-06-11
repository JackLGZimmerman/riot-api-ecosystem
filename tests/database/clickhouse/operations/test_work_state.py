from __future__ import annotations

from uuid import UUID

from database.clickhouse.operations import utils as ops_utils
from database.clickhouse.operations import work_state


def _patch_client(monkeypatch, client) -> None:
    # work_state issues queries/commands via its own get_client binding; the
    # shared record_timestamp helper inserts via operations.utils.get_client.
    monkeypatch.setattr(work_state, "get_client", lambda: client)
    monkeypatch.setattr(ops_utils, "get_client", lambda: client)


class FakeResult:
    def __init__(self, rows):
        self.result_rows = rows


class FakeClient:
    def __init__(self, query_results):
        self._query_results = iter(query_results)
        self.queries = []
        self.commands = []
        self.inserts = []

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters))
        return FakeResult(next(self._query_results))

    def command(self, sql, parameters=None):
        self.commands.append((sql, parameters))

    def insert(self, table, data, column_names):
        self.inserts.append((table, data, column_names))


def test_seed_from_latest_matchids_uses_insert_select(monkeypatch):
    run_id = UUID("11111111-1111-1111-1111-111111111111")
    client = FakeClient(
        [
            [(run_id,)],
            [(None,)],
            [(2,)],
        ]
    )
    _patch_client(monkeypatch, client)
    monkeypatch.setattr(work_state.time, "time", lambda: 123)

    assert work_state.seed_from_latest_matchids() == 2

    assert len(client.commands) == 1
    command_sql, command_params = client.commands[0]
    assert (
        "INSERT INTO game_data.matchdata_matchids (run_id, matchid)" in command_sql
    )
    assert "SELECT DISTINCT" in command_sql
    assert "INNER JOIN timeline_matchids USING (matchid)" in command_sql
    assert command_params == {"run_id": run_id}

    count_sql, count_params = client.queries[2]
    assert "SELECT count()" in count_sql
    assert "FROM (" in count_sql
    assert count_params == {"run_id": run_id}

    assert client.inserts == [
        (
            "game_data.data_timestamps",
            [(work_state.MATCHDATA_SEEDED_RUN_NAME, run_id, 123)],
            ("name", "run_id", "stored_at"),
        )
    ]


def test_seed_from_latest_matchids_skips_already_seeded_run(monkeypatch):
    run_id = UUID("22222222-2222-2222-2222-222222222222")
    client = FakeClient(
        [
            [(run_id,)],
            [(run_id,)],
        ]
    )
    _patch_client(monkeypatch, client)

    assert work_state.seed_from_latest_matchids() == 0
    assert client.commands == []
    assert client.inserts == []


def test_seed_from_latest_matchids_marks_empty_seed(monkeypatch):
    run_id = UUID("33333333-3333-3333-3333-333333333333")
    client = FakeClient(
        [
            [(run_id,)],
            [(None,)],
            [(0,)],
        ]
    )
    _patch_client(monkeypatch, client)
    monkeypatch.setattr(work_state.time, "time", lambda: 456)

    assert work_state.seed_from_latest_matchids() == 0
    assert client.commands == []
    assert client.inserts == [
        (
            "game_data.data_timestamps",
            [(work_state.MATCHDATA_SEEDED_RUN_NAME, run_id, 456)],
            ("name", "run_id", "stored_at"),
        )
    ]


def test_claim_pending_matchids_balances_by_continent(monkeypatch):
    client = FakeClient(
        [
            [("NA1_1",), ("LA1_1",), ("EUW1_1",)],
        ]
    )
    _patch_client(monkeypatch, client)

    assert work_state.claim_pending_matchids(batch_size=250) == [
        "NA1_1",
        "LA1_1",
        "EUW1_1",
    ]

    claim_sql, claim_params = client.queries[0]
    assert "AS continent" in claim_sql
    assert "cityHash64('matchdata_claim', matchid) AS shuffle_key" in claim_sql
    assert "LIMIT %(limit)s BY continent" in claim_sql
    assert "PARTITION BY continent" in claim_sql
    assert "continent_order" in claim_sql
    assert "LIMIT %(limit)s BY region" not in claim_sql
    assert "region_order" not in claim_sql
    assert claim_params == {"limit": 250}
