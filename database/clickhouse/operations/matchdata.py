from itertools import islice
from typing import Iterable, Sequence
from uuid import UUID

from database.clickhouse.client import get_client


def _batched(iterable: Iterable, batch_size: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            return
        yield batch


def insert_rows_in_batches(
    table: str,
    columns: Sequence[str],
    rows: Iterable[tuple],
    batch_size: int,
) -> None:
    client = get_client()
    cols = tuple(columns)

    for batch in _batched(rows, batch_size):
        client.insert(table, batch, cols)


def persist_data(
    table: str,
    columns: Sequence[str],
    items: Iterable[dict],
    run_id: UUID,
    batch_size: int,
) -> None:
    source_cols = tuple(columns)
    db_cols = tuple(c.lower() for c in source_cols)
    rows = ((run_id, *(item[c] for c in source_cols)) for item in items)

    insert_rows_in_batches(
        table,
        ("run_id", *db_cols),
        rows,
        batch_size=batch_size,
    )


def delete_by_run_id(table: str, run_id: UUID) -> None:
    command = f"ALTER TABLE {table} DELETE WHERE run_id = %(run_id)s"
    get_client().command(command, parameters={"run_id": str(run_id)})


def insert_match_ids(
    match_ids: Iterable[str],
    run_id: UUID,
    batch_size: int = 50_000,
) -> None:
    rows = ((run_id, matchid) for matchid in match_ids)

    insert_rows_in_batches(
        table="game_data.matchdata_matchids",
        columns=("run_id", "matchid"),
        rows=rows,
        batch_size=batch_size,
    )


def delete_match_ids(run_id: UUID) -> None:
    delete_by_run_id("game_data.matchdata_matchids", run_id)


def load_pending_match_ids() -> list[str]:
    query = """
        SELECT m.matchid
        FROM game_data.matchids AS m
        WHERE m.matchid NOT IN (
            SELECT matchid
            FROM game_data.matchdata_matchids
        )
    """
    rows = get_client().query(query).result_rows
    return [row[0] for row in rows]
