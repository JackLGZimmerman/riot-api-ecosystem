import logging
import time
from collections.abc import Iterable

from database.clickhouse.client import get_client
from database.clickhouse.operations.utils import dedupe_matchids

logger = logging.getLogger(__name__)

MUTATION_POLL_INTERVAL_S = 5.0
MUTATION_PROGRESS_LOG_INTERVAL_S = 60.0


def _split_table_name(table: str) -> tuple[str, str]:
    parts = table.split(".", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Expected fully-qualified ClickHouse table name: {table}")
    return parts[0], parts[1]


def _latest_mutation_sequence(client, *, database: str, table: str) -> int:
    rows = client.query(
        """
        SELECT maxOrNull(toUInt64OrZero(extract(mutation_id, 'mutation_(\\d+)')))
        FROM system.mutations
        WHERE database = %(database)s
          AND table = %(table)s
        """,
        parameters={"database": database, "table": table},
    ).result_rows
    value = rows[0][0] if rows else None
    return int(value or 0)


def _has_matching_rows(client, *, table: str, match_ids: list[str]) -> bool:
    rows = client.query(
        f"""
        SELECT 1
        FROM {table}
        WHERE has(%(match_ids)s, matchid)
        LIMIT 1
        """,
        parameters={"match_ids": match_ids},
    ).result_rows
    return bool(rows)


def _wait_for_mutations_after(
    client,
    *,
    database: str,
    table: str,
    after_sequence: int,
) -> None:
    last_log = 0.0

    while True:
        rows = client.query(
            """
            SELECT
                count() AS total,
                countIf(is_done) AS done,
                ifNull(sum(parts_to_do), 0) AS parts_to_do,
                groupArrayIf(latest_fail_reason, latest_fail_reason != '') AS failures
            FROM system.mutations
            WHERE database = %(database)s
              AND table = %(table)s
              AND toUInt64OrZero(extract(mutation_id, 'mutation_(\\d+)')) > %(after_sequence)s
            """,
            parameters={
                "database": database,
                "table": table,
                "after_sequence": after_sequence,
            },
        ).result_rows

        total, done, parts_to_do, failures = rows[0]
        if failures:
            raise RuntimeError(
                f"ClickHouse mutation failed for {database}.{table}: {failures[0]}"
            )
        if total and done == total:
            return

        now = time.monotonic()
        if now - last_log >= MUTATION_PROGRESS_LOG_INTERVAL_S:
            logger.info(
                "Waiting for ClickHouse mutation database=%s table=%s total=%s done=%s parts_to_do=%s",
                database,
                table,
                total,
                done,
                parts_to_do,
            )
            last_log = now

        time.sleep(MUTATION_POLL_INTERVAL_S)


def delete_by_matchids(
    table: str,
    match_ids: Iterable[str],
) -> None:
    ids = dedupe_matchids(match_ids)
    if not ids:
        return

    database, table_name = _split_table_name(table)
    client = get_client()
    if not _has_matching_rows(client, table=table, match_ids=ids):
        logger.debug(
            "No matchdata rows to delete table=%s matchids=%d",
            table,
            len(ids),
        )
        return

    after_sequence = _latest_mutation_sequence(
        client,
        database=database,
        table=table_name,
    )

    client.command(
        f"""
        ALTER TABLE {table}
        DELETE
        WHERE has(%(match_ids)s, matchid)
        SETTINGS mutations_sync = 0
        """,
        parameters={"match_ids": ids},
    )
    _wait_for_mutations_after(
        client,
        database=database,
        table=table_name,
        after_sequence=after_sequence,
    )
