#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.worker.pipelines.matchdata_orchestrator import ALL_DELETE_TABLES
from database.clickhouse.client import get_client
from database.clickhouse.operations.utils import dedupe_matchids

REPAIR_DATABASE = "game_data_repair"
QUEUE_TABLE = "game_data.matchdata_matchids"
SOURCE_TABLES = (*ALL_DELETE_TABLES, "game_data.matchids", QUEUE_TABLE)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _assert_qualified_table(table: str) -> tuple[str, str]:
    database, _, name = table.partition(".")
    if not database or not name:
        raise ValueError(f"Expected qualified table name, got {table!r}")
    if not IDENTIFIER_RE.fullmatch(database) or not IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"Unsafe table name: {table!r}")
    return database, name


def _backup_table_name(source_table: str, backup_prefix: str) -> str:
    _database, name = _assert_qualified_table(source_table)
    return f"{REPAIR_DATABASE}.{backup_prefix}_{name}"


def _load_partial_matchids(client) -> list[str]:
    rows = client.query(
        """
        SELECT matchid
        FROM
        (
            SELECT
                matchid,
                max(source = 'info') AS has_info,
                max(source = 'timeline') AS has_timeline
            FROM
            (
                SELECT DISTINCT matchid, 'info' AS source
                FROM game_data.info
                WHERE matchid != ''
                UNION ALL
                SELECT DISTINCT matchid, 'timeline' AS source
                FROM game_data.tl_game_end
                WHERE matchid != ''
            )
            GROUP BY matchid
        )
        WHERE has_info != has_timeline
        ORDER BY matchid
        """
    ).result_rows
    return dedupe_matchids(row[0] for row in rows)


def _count_table_matchids(client, table: str, matchids: list[str]) -> int:
    rows = client.query(
        f"""
        SELECT count()
        FROM {table}
        WHERE has(%(matchids)s, matchid)
        """,
        parameters={"matchids": matchids},
    ).result_rows
    return int(rows[0][0])


def _backup_source_table(
    client,
    *,
    source_table: str,
    backup_prefix: str,
    matchids: list[str],
    apply: bool,
) -> tuple[str, int]:
    backup_table = _backup_table_name(source_table, backup_prefix)
    source_count = _count_table_matchids(client, source_table, matchids)
    if not apply:
        return backup_table, source_count

    client.command(f"CREATE DATABASE IF NOT EXISTS {REPAIR_DATABASE}")
    client.command(
        f"""
        CREATE TABLE {backup_table}
        AS {source_table}
        ENGINE = MergeTree
        ORDER BY tuple()
        """
    )
    client.command(
        f"""
        INSERT INTO {backup_table}
        SELECT *
        FROM {source_table}
        WHERE has(%(matchids)s, matchid)
        """,
        parameters={"matchids": matchids},
    )
    backup_count = _count_table_matchids(client, backup_table, matchids)
    if backup_count != source_count:
        raise RuntimeError(
            f"Backup row count mismatch for {source_table}: "
            f"source={source_count} backup={backup_count}"
        )
    return backup_table, source_count


def _queue_missing_matchids(client, *, matchids: list[str], apply: bool) -> int:
    ids = dedupe_matchids(matchids)
    existing_rows = client.query(
        f"""
        SELECT DISTINCT matchid
        FROM {QUEUE_TABLE}
        WHERE has(%(matchids)s, matchid)
        """,
        parameters={"matchids": ids},
    ).result_rows
    existing = {row[0] for row in existing_rows}
    missing = [matchid for matchid in ids if matchid not in existing]
    if apply and missing:
        repair_run_id = uuid4()
        client.insert(
            QUEUE_TABLE,
            [(repair_run_id, matchid) for matchid in missing],
            column_names=("run_id", "matchid"),
        )
        print(f"Queued {len(missing)} partial matchids with run_id={repair_run_id}")
    return len(missing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backup and requeue partial matchdata rows without deleting data."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create backup tables and insert missing partial matchids into the queue.",
    )
    parser.add_argument(
        "--backup-prefix",
        default=f"partial_matchdata_{datetime.now(UTC):%Y%m%d_%H%M%S}",
        help="Prefix for backup tables created under game_data_repair.",
    )
    parser.add_argument(
        "--matchid",
        action="append",
        default=[],
        help="Repair only this matchid. Can be provided multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = get_client()
    matchids = dedupe_matchids(args.matchid) or _load_partial_matchids(client)
    if not matchids:
        print("No partial matchdata rows found.")
        return

    action = "APPLY" if args.apply else "DRY RUN"
    print(f"{action}: {len(matchids)} partial matchids")
    print(f"Backup prefix: {args.backup_prefix}")

    for source_table in SOURCE_TABLES:
        backup_table, row_count = _backup_source_table(
            client,
            source_table=source_table,
            backup_prefix=args.backup_prefix,
            matchids=matchids,
            apply=args.apply,
        )
        print(f"{source_table} -> {backup_table}: {row_count} rows")

    missing = _queue_missing_matchids(client, matchids=matchids, apply=args.apply)
    if args.apply:
        print(f"Repair complete; {missing} matchids were newly queued.")
    else:
        print(f"Dry run complete; {missing} matchids would be queued.")


if __name__ == "__main__":
    main()
