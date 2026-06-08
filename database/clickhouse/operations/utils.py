import logging
from itertools import islice
from collections.abc import Iterable, Sequence
from uuid import UUID

from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)


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
        logger.debug("Insert batch table=%s rows=%d", table, len(batch))
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


def _as_text(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8").rstrip("\x00")
    return str(value)


def dedupe_matchids(values: Iterable[object]) -> list[str]:
    """Order-preserving dedupe of match ids: normalise via `_as_text`
    (bytes-aware) and drop empty values."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _as_text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


DATA_TIMESTAMPS_TABLE = "game_data.data_timestamps"
# Canonical column order for game_data.data_timestamps writes (matches the
# schema column order in 0001_data_timestamps_schema.sql).
_DATA_TIMESTAMPS_COLUMNS = ("name", "run_id", "stored_at")


def record_timestamp(name: str, run_id: UUID, stored_at: int) -> None:
    """Insert one (name, run_id, stored_at) row using a single canonical column
    order, so every caller writes identical rows."""
    get_client().insert(
        table=DATA_TIMESTAMPS_TABLE,
        data=[(name, run_id, stored_at)],
        column_names=_DATA_TIMESTAMPS_COLUMNS,
    )


def delete_timestamp_for_run(name: str, run_id: UUID) -> None:
    """Delete the data_timestamps row for one (name, run_id)."""
    get_client().command(
        f"""
        ALTER TABLE {DATA_TIMESTAMPS_TABLE}
        DELETE
        WHERE name = %(name)s
          AND run_id = %(run_id)s
        """,
        parameters={"name": name, "run_id": run_id},
    )


def delete_timestamps_except_run(name: str, run_id: UUID) -> None:
    """Delete all data_timestamps rows for `name` except the given run_id."""
    get_client().command(
        f"""
        ALTER TABLE {DATA_TIMESTAMPS_TABLE}
        DELETE
        WHERE name = %(name)s
          AND run_id != %(run_id)s
        """,
        parameters={"name": name, "run_id": run_id},
    )
