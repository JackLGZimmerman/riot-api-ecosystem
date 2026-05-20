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
