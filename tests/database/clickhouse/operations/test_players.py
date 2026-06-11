from __future__ import annotations

from database.clickhouse.operations import players


class FakeResult:
    def __init__(self, rows):
        self.result_rows = rows


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters))
        return FakeResult(self.rows)


def test_load_players_reads_latest_published_snapshot(monkeypatch) -> None:
    client = FakeClient(
        [
            (b"puuid-a\x00", "RANKED_SOLO_5x5", "na1"),
            ("puuid-b", "RANKED_SOLO_5x5", "kr"),
        ]
    )
    monkeypatch.setattr(players, "get_client", lambda: client)

    rows = players.load_players()

    assert rows == [
        players.PlayerKeyRow("puuid-a", "RANKED_SOLO_5x5", "na1"),
        players.PlayerKeyRow("puuid-b", "RANKED_SOLO_5x5", "kr"),
    ]
    sql, params = client.queries[0]
    assert "players_snapshot_ts" not in sql
    assert "WHERE name = %(timestamp_name)s" in sql
    assert "WHERE run_id = (SELECT run_id FROM latest)" in sql
    assert params == {"timestamp_name": players.PLAYERS_SNAPSHOT_TIMESTAMP_NAME}
