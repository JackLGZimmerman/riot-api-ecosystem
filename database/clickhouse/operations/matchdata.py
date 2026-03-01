from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
import time
from typing import Iterable, Mapping, Sequence
from uuid import UUID

from database.clickhouse.client import get_client

MATCHDATA_STATE_TABLE = "game_data.matchdata_state"

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"

MATCHDATA_STATE_COLUMNS = (
    "matchid",
    "status",
    "retry_count",
    "error_message",
    "run_id",
    "updated_at",
    "state_version",
)


@dataclass(frozen=True)
class ClaimedMatchID:
    matchid: str
    retry_count: int


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


def ensure_match_state_table() -> None:
    query = """
        CREATE TABLE IF NOT EXISTS game_data.matchdata_state
        (
            matchid String CODEC (ZSTD(3)),
            status Enum8(
                'pending' = 1,
                'processing' = 2,
                'finished' = 3,
                'failed' = 4
            ),
            retry_count UInt16,
            error_message String CODEC (ZSTD(3)),
            run_id Nullable(UUID),
            updated_at DateTime64(3, 'UTC'),
            state_version UInt64,
            INDEX idx_status status TYPE set(4) GRANULARITY 1
        )
        ENGINE = ReplacingMergeTree(state_version)
        ORDER BY (matchid)
    """
    get_client().command(query)


def _insert_match_state_rows(
    rows: list[tuple[str, str, int, str, UUID | None]],
) -> None:
    if not rows:
        return

    now = datetime.now(timezone.utc)
    version_seed = time.time_ns()
    data = [
        (
            matchid,
            status,
            int(retry_count),
            error_message,
            run_id,
            now,
            version_seed + idx,
        )
        for idx, (matchid, status, retry_count, error_message, run_id) in enumerate(rows)
    ]

    get_client().insert(
        table=MATCHDATA_STATE_TABLE,
        data=data,
        column_names=MATCHDATA_STATE_COLUMNS,
    )


def _unique_matchids(match_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(match_ids))


def initialize_match_state_from_matchids() -> None:
    ensure_match_state_table()

    query = """
        INSERT INTO game_data.matchdata_state
            (
                matchid,
                status,
                retry_count,
                error_message,
                run_id,
                updated_at,
                state_version
            )
        SELECT
            src.matchid,
            'pending' AS status,
            toUInt16(0) AS retry_count,
            '' AS error_message,
            CAST(NULL, 'Nullable(UUID)') AS run_id,
            now64(3) AS updated_at,
            toUInt64(toUnixTimestamp64Nano(now64(9)) + rowNumberInAllBlocks()) AS state_version
        FROM (
            SELECT DISTINCT matchid
            FROM game_data.matchids
        ) AS src
        LEFT JOIN (
            SELECT DISTINCT matchid
            FROM game_data.matchdata_state
        ) AS existing USING (matchid)
        WHERE existing.matchid IS NULL
    """
    get_client().command(query)


def claim_pending_match_ids(limit: int, *, run_id: UUID) -> list[ClaimedMatchID]:
    if limit <= 0:
        return []

    query = """
        SELECT
            matchid,
            toUInt16(argMax(retry_count, state_version)) AS retry_count
        FROM game_data.matchdata_state
        GROUP BY matchid
        HAVING toString(argMax(status, state_version)) = %(status)s
        ORDER BY matchid
        LIMIT %(limit)s
    """
    rows = get_client().query(
        query,
        parameters={"status": STATUS_PENDING, "limit": limit},
    ).result_rows

    claimed: list[ClaimedMatchID] = [
        ClaimedMatchID(matchid=row[0], retry_count=int(row[1])) for row in rows
    ]
    if not claimed:
        return []

    _insert_match_state_rows(
        [
            (m.matchid, STATUS_PROCESSING, m.retry_count, "", run_id)
            for m in claimed
        ]
    )
    return claimed


def recover_stale_processing_match_ids(
    *,
    stale_after_minutes: int,
    max_retries: int,
) -> tuple[list[str], list[str], set[UUID]]:
    if stale_after_minutes <= 0:
        return [], [], set()

    query = """
        SELECT
            matchid,
            toUInt16(argMax(retry_count, state_version)) AS retry_count,
            argMax(run_id, state_version) AS run_id
        FROM game_data.matchdata_state
        GROUP BY matchid
        HAVING toString(argMax(status, state_version)) = %(status)s
           AND argMax(updated_at, state_version) < now64(3) - toIntervalMinute(%(minutes)s)
        ORDER BY matchid
    """
    rows = get_client().query(
        query,
        parameters={"status": STATUS_PROCESSING, "minutes": stale_after_minutes},
    ).result_rows
    if not rows:
        return [], [], set()

    requeued: list[str] = []
    permanently_failed: list[str] = []
    state_rows: list[tuple[str, str, int, str, UUID | None]] = []
    stale_run_ids: set[UUID] = set()
    stale_error = (
        f"processing claim timed out after {stale_after_minutes} minutes; recovered"
    )

    for matchid, retry_count_raw, stale_run_id_raw in rows:
        stale_run_id = _coerce_uuid(stale_run_id_raw)
        if stale_run_id is not None:
            stale_run_ids.add(stale_run_id)

        retry_count = int(retry_count_raw)
        next_retry_count = retry_count + 1
        if next_retry_count >= max_retries:
            status = STATUS_FAILED
            permanently_failed.append(matchid)
        else:
            status = STATUS_PENDING
            requeued.append(matchid)

        state_rows.append((matchid, status, next_retry_count, stale_error, stale_run_id))

    _insert_match_state_rows(state_rows)
    return requeued, permanently_failed, stale_run_ids


def mark_match_ids_finished(
    match_ids: Iterable[str],
    retry_counts: Mapping[str, int],
    *,
    run_id: UUID,
) -> None:
    unique_ids = _unique_matchids(match_ids)
    rows = [
        (matchid, STATUS_FINISHED, int(retry_counts.get(matchid, 0)), "", run_id)
        for matchid in unique_ids
    ]
    _insert_match_state_rows(rows)


def mark_match_ids_after_attempt(
    match_ids: Iterable[str],
    retry_counts: Mapping[str, int],
    *,
    max_retries: int,
    error_message: str,
    run_id: UUID,
) -> tuple[list[str], list[str]]:
    unique_ids = _unique_matchids(match_ids)
    if not unique_ids:
        return [], []

    requeued: list[str] = []
    permanently_failed: list[str] = []
    rows: list[tuple[str, str, int, str, UUID | None]] = []
    normalized_error = error_message[:1000]

    for matchid in unique_ids:
        next_retry_count = int(retry_counts.get(matchid, 0)) + 1
        if next_retry_count >= max_retries:
            status = STATUS_FAILED
            permanently_failed.append(matchid)
        else:
            status = STATUS_PENDING
            requeued.append(matchid)

        rows.append((matchid, status, next_retry_count, normalized_error, run_id))

    _insert_match_state_rows(rows)
    return requeued, permanently_failed


def _coerce_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value

    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None
