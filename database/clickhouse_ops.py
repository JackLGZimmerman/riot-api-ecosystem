from __future__ import annotations

from typing import Any, Dict, Iterable

from database.clickhouse import get_client


def insert_events(rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return

    cols = ["event_time", "game_id", "puuid", "event_type", "payload"]
    data = [[r.get(c) for c in cols] for r in rows]

    get_client().insert(
        table="events",
        data=data,
        column_names=cols,
    )


def query(sql: str, params: dict | None = None):
    return get_client().query(sql, parameters=params or {})
