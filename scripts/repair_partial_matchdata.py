#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.worker.pipelines.matchdata_orchestrator import ALL_DELETE_TABLES
from database.clickhouse.client import get_client
from database.clickhouse.operations.utils import dedupe_matchids

REPAIR_DATABASE = "game_data_repair"
QUEUE_TABLE = "game_data.matchdata_matchids"
METADATA_TABLE = "game_data.metadata"
INFO_TABLE = "game_data.info"
TIMELINE_END_TABLE = "game_data.tl_game_end"
PARTICIPANT_STATS_TABLE = "game_data.participant_stats"
SOURCE_TABLES = (*ALL_DELETE_TABLES, "game_data.matchids", QUEUE_TABLE)
METADATA_REPAIR_TABLES = (
    METADATA_TABLE,
    INFO_TABLE,
    TIMELINE_END_TABLE,
    PARTICIPANT_STATS_TABLE,
    QUEUE_TABLE,
)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MATCHID_RE = re.compile(r"\b[A-Z0-9]+_[0-9]+\b")


@dataclass(frozen=True)
class BackupSummary:
    source_table: str
    backup_table: str
    row_count: int


@dataclass(frozen=True)
class RepairClasses:
    metadata_only: tuple[str, ...]
    stream_partial: tuple[str, ...]
    metadata_blocked: tuple[str, ...]
    already_complete: tuple[str, ...]
    no_action: tuple[str, ...]

    @property
    def total(self) -> int:
        return (
            len(self.metadata_only)
            + len(self.stream_partial)
            + len(self.metadata_blocked)
            + len(self.already_complete)
            + len(self.no_action)
        )


def _assert_qualified_table(table: str) -> tuple[str, str]:
    database, _, name = table.partition(".")
    if not database or not name:
        raise ValueError(f"Expected qualified table name, got {table!r}")
    if not IDENTIFIER_RE.fullmatch(database) or not IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"Unsafe table name: {table!r}")
    return database, name


def _backup_table_name(source_table: str, backup_prefix: str) -> str:
    _database, name = _assert_qualified_table(source_table)
    if not IDENTIFIER_RE.fullmatch(backup_prefix):
        raise ValueError(f"Unsafe backup prefix: {backup_prefix!r}")
    return f"{REPAIR_DATABASE}.{backup_prefix}_{name}"


def _load_stream_partial_matchids(client) -> list[str]:
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


def _load_partial_matchids(client) -> list[str]:
    return _load_stream_partial_matchids(client)


def _load_metadata_gap_matchids(client) -> list[str]:
    rows = client.query(
        f"""
        SELECT info_matchids.matchid
        FROM
        (
            SELECT DISTINCT matchid
            FROM {INFO_TABLE}
            WHERE matchid != ''
        ) AS info_matchids
        INNER JOIN
        (
            SELECT DISTINCT matchid
            FROM {TIMELINE_END_TABLE}
            WHERE matchid != ''
        ) AS timeline_matchids USING (matchid)
        WHERE info_matchids.matchid NOT IN
        (
            SELECT DISTINCT matchid
            FROM {METADATA_TABLE}
            WHERE matchid != ''
        )
        ORDER BY info_matchids.matchid
        """
    ).result_rows
    return dedupe_matchids(row[0] for row in rows)


def _load_auto_repair_matchids(client) -> list[str]:
    return dedupe_matchids(
        [*_load_metadata_gap_matchids(client), *_load_stream_partial_matchids(client)]
    )


def _load_queue_matchids_for_run(client, run_id: str) -> list[str]:
    rows = client.query(
        f"""
        SELECT DISTINCT matchid
        FROM {QUEUE_TABLE}
        WHERE run_id = toUUID(%(run_id)s)
        ORDER BY matchid
        """,
        parameters={"run_id": str(UUID(run_id))},
    ).result_rows
    return dedupe_matchids(row[0] for row in rows)


def _load_active_mutation_matchids(client) -> list[str]:
    rows = client.query(
        """
        SELECT command
        FROM system.mutations
        WHERE database = 'game_data'
          AND is_done = 0
        ORDER BY create_time DESC, mutation_id DESC
        LIMIT 1
        """
    ).result_rows
    if not rows:
        return []
    return dedupe_matchids(MATCHID_RE.findall(str(rows[0][0])))


def _load_distinct_matchids(client, table: str, matchids: list[str]) -> set[str]:
    if not matchids:
        return set()
    _assert_qualified_table(table)
    rows = client.query(
        f"""
        SELECT DISTINCT matchid
        FROM {table}
        WHERE has(%(matchids)s, matchid)
        """,
        parameters={"matchids": matchids},
    ).result_rows
    return {row[0] for row in rows}


def _load_participant_valid_matchids(client, matchids: list[str]) -> set[str]:
    if not matchids:
        return set()
    rows = client.query(
        f"""
        SELECT
            matchid,
            count() AS participant_rows,
            uniqExact(participantid) AS unique_participant_ids
        FROM {PARTICIPANT_STATS_TABLE}
        WHERE has(%(matchids)s, matchid)
        GROUP BY matchid
        """,
        parameters={"matchids": matchids},
    ).result_rows
    return {row[0] for row in rows if int(row[1]) == 10 and int(row[1]) == int(row[2])}


def _classify_presence(
    matchids: Iterable[str],
    *,
    info_matchids: set[str],
    timeline_matchids: set[str],
    metadata_matchids: set[str],
    valid_participant_matchids: set[str],
) -> RepairClasses:
    metadata_only: list[str] = []
    stream_partial: list[str] = []
    metadata_blocked: list[str] = []
    already_complete: list[str] = []
    no_action: list[str] = []

    for matchid in dedupe_matchids(matchids):
        has_info = matchid in info_matchids
        has_timeline = matchid in timeline_matchids
        has_metadata = matchid in metadata_matchids

        if has_info != has_timeline:
            stream_partial.append(matchid)
        elif has_info and has_timeline and has_metadata:
            already_complete.append(matchid)
        elif (
            has_info
            and has_timeline
            and not has_metadata
            and matchid in valid_participant_matchids
        ):
            metadata_only.append(matchid)
        elif has_info and has_timeline and not has_metadata:
            metadata_blocked.append(matchid)
        else:
            no_action.append(matchid)

    return RepairClasses(
        metadata_only=tuple(metadata_only),
        stream_partial=tuple(stream_partial),
        metadata_blocked=tuple(metadata_blocked),
        already_complete=tuple(already_complete),
        no_action=tuple(no_action),
    )


def _classify_matchids(client, matchids: list[str]) -> RepairClasses:
    ids = dedupe_matchids(matchids)
    return _classify_presence(
        ids,
        info_matchids=_load_distinct_matchids(client, INFO_TABLE, ids),
        timeline_matchids=_load_distinct_matchids(client, TIMELINE_END_TABLE, ids),
        metadata_matchids=_load_distinct_matchids(client, METADATA_TABLE, ids),
        valid_participant_matchids=_load_participant_valid_matchids(client, ids),
    )


def _count_table_matchids(client, table: str, matchids: list[str]) -> int:
    _assert_qualified_table(table)
    rows = client.query(
        f"""
        SELECT count()
        FROM {table}
        WHERE has(%(matchids)s, matchid)
        """,
        parameters={"matchids": matchids},
    ).result_rows
    return int(rows[0][0])


def _backup_tables(
    client,
    *,
    source_tables: tuple[str, ...],
    backup_prefix: str,
    matchids: list[str],
    apply: bool,
) -> list[BackupSummary]:
    summaries: list[BackupSummary] = []
    for source_table in source_tables:
        backup_table, row_count = _backup_source_table(
            client,
            source_table=source_table,
            backup_prefix=backup_prefix,
            matchids=matchids,
            apply=apply,
        )
        summaries.append(
            BackupSummary(
                source_table=source_table,
                backup_table=backup_table,
                row_count=row_count,
            )
        )
    return summaries


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


def _validate_metadata_dataversion(client) -> None:
    rows = client.query(
        f"""
        SELECT
            count() AS bad_rows,
            groupArrayDistinct(dataversion) AS bad_dataversions
        FROM {METADATA_TABLE}
        WHERE dataversion != 2
        """
    ).result_rows
    bad_rows = int(rows[0][0]) if rows else 0
    if bad_rows:
        raise RuntimeError(
            "Metadata-only repair assumes existing metadata dataversion=2; "
            f"found non-2 dataversions: {rows[0][1]}"
        )


def _require_metadata_only_targets(client, matchids: list[str]) -> None:
    classes = _classify_matchids(client, matchids)
    if classes.metadata_only == tuple(matchids):
        return
    raise RuntimeError(
        "Metadata-only repair target validation failed: "
        f"metadata_only={len(classes.metadata_only)} "
        f"stream_partial={len(classes.stream_partial)} "
        f"metadata_blocked={len(classes.metadata_blocked)} "
        f"already_complete={len(classes.already_complete)} "
        f"no_action={len(classes.no_action)}"
    )


def _insert_missing_metadata(
    client, *, matchids: list[str], repair_run_id: UUID
) -> None:
    client.command(
        f"""
        INSERT INTO {METADATA_TABLE} (run_id, matchid, dataversion, participants)
        SELECT
            toUUID(%(run_id)s) AS run_id,
            matchid,
            toUInt8(2) AS dataversion,
            arrayMap(
                item -> tupleElement(item, 2),
                arraySort(
                    item -> tupleElement(item, 1),
                    groupArray((participantid, puuid))
                )
            ) AS participants
        FROM {PARTICIPANT_STATS_TABLE}
        WHERE has(%(matchids)s, matchid)
          AND matchid NOT IN
          (
              SELECT DISTINCT matchid
              FROM {METADATA_TABLE}
              WHERE has(%(matchids)s, matchid)
          )
        GROUP BY matchid
        """,
        parameters={"run_id": str(repair_run_id), "matchids": matchids},
    )


def _load_metadata_counts(client, matchids: list[str]) -> dict[str, int]:
    rows = client.query(
        f"""
        SELECT matchid, count()
        FROM {METADATA_TABLE}
        WHERE has(%(matchids)s, matchid)
        GROUP BY matchid
        """,
        parameters={"matchids": matchids},
    ).result_rows
    return {row[0]: int(row[1]) for row in rows}


def _validate_metadata_after_insert(client, matchids: list[str]) -> None:
    counts = _load_metadata_counts(client, matchids)
    bad = [matchid for matchid in matchids if counts.get(matchid, 0) != 1]
    if bad:
        raise RuntimeError(
            "Metadata-only repair post-validation failed; expected exactly one "
            f"metadata row for every target, bad_count={len(bad)} sample={bad[:5]}"
        )


def _delete_queue_rows(client, matchids: list[str]) -> None:
    if not matchids:
        return
    client.command(
        f"""
        ALTER TABLE {QUEUE_TABLE}
        DELETE
        WHERE has(%(matchids)s, matchid)
        SETTINGS mutations_sync = 2
        """,
        parameters={"matchids": matchids},
    )


def _apply_metadata_only_repair(
    client,
    *,
    matchids: list[str],
    backup_prefix: str,
    apply: bool,
) -> tuple[list[BackupSummary], UUID | None]:
    ids = dedupe_matchids(matchids)
    _validate_metadata_dataversion(client)
    _require_metadata_only_targets(client, ids)
    backups = _backup_tables(
        client,
        source_tables=METADATA_REPAIR_TABLES,
        backup_prefix=backup_prefix,
        matchids=ids,
        apply=apply,
    )
    if not apply:
        return backups, None

    repair_run_id = uuid4()
    _insert_missing_metadata(client, matchids=ids, repair_run_id=repair_run_id)
    _validate_metadata_after_insert(client, ids)
    _delete_queue_rows(client, ids)
    return backups, repair_run_id


def _apply_full_requeue(
    client,
    *,
    matchids: list[str],
    backup_prefix: str,
    apply: bool,
) -> tuple[list[BackupSummary], int]:
    ids = dedupe_matchids(matchids)
    backups = _backup_tables(
        client,
        source_tables=SOURCE_TABLES,
        backup_prefix=backup_prefix,
        matchids=ids,
        apply=apply,
    )
    missing = _queue_missing_matchids(client, matchids=ids, apply=apply)
    return backups, missing


def _print_backups(backups: list[BackupSummary]) -> None:
    for backup in backups:
        print(
            f"{backup.source_table} -> {backup.backup_table}: {backup.row_count} rows"
        )


def _print_repair_summary(classes: RepairClasses) -> None:
    print(
        "Repair classes: "
        f"metadata_only={len(classes.metadata_only)} "
        f"stream_partial={len(classes.stream_partial)} "
        f"metadata_blocked={len(classes.metadata_blocked)} "
        f"already_complete={len(classes.already_complete)} "
        f"no_action={len(classes.no_action)}"
    )
    print(
        "Estimated action: "
        "metadata_only=insert_metadata_and_clear_queue; "
        "stream_partial=full_requeue_requires_allow_full_requeue"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair matchdata gaps. Metadata-only gaps are insert-only; true "
            "stream partials require explicit full-requeue approval."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create backups and apply the selected repair actions.",
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
    parser.add_argument(
        "--repair-run-id",
        help="Use matchids currently queued with this repair run_id as candidates.",
    )
    parser.add_argument(
        "--allow-full-requeue",
        action="store_true",
        help="Allow stream-partial matchids to use the old backup-and-requeue path.",
    )
    parser.add_argument(
        "--exclude-active-mutation-matchids",
        action="store_true",
        help="Exclude matchids parsed from the latest active game_data mutation command.",
    )
    return parser.parse_args(argv)


def _candidate_matchids(client, args: argparse.Namespace) -> list[str]:
    matchids = list(args.matchid)
    if args.repair_run_id:
        matchids.extend(_load_queue_matchids_for_run(client, args.repair_run_id))
    ids = dedupe_matchids(matchids)
    if args.matchid or args.repair_run_id:
        return ids
    return _load_auto_repair_matchids(client)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    client = get_client()
    matchids = _candidate_matchids(client, args)

    if args.exclude_active_mutation_matchids:
        active_matchids = set(_load_active_mutation_matchids(client))
        before = len(matchids)
        matchids = [matchid for matchid in matchids if matchid not in active_matchids]
        print(f"Excluded active mutation matchids: {before - len(matchids)}")

    if not matchids:
        print("No matchdata repair candidates found.")
        return

    action = "APPLY" if args.apply else "DRY RUN"
    classes = _classify_matchids(client, matchids)
    print(f"{action}: {classes.total} matchdata repair candidates")
    print(f"Backup prefix: {args.backup_prefix}")
    _print_repair_summary(classes)

    if args.apply and classes.metadata_blocked:
        raise RuntimeError(
            "Refusing metadata-only repair because participant_stats cannot "
            f"reconstruct participants for {len(classes.metadata_blocked)} matchids."
        )

    metadata_backup_prefix = (
        args.backup_prefix
        if not classes.stream_partial
        else f"{args.backup_prefix}_metadata_only"
    )
    full_backup_prefix = (
        args.backup_prefix
        if not classes.metadata_only
        else f"{args.backup_prefix}_stream_partial"
    )

    if classes.metadata_only:
        backups, repair_run_id = _apply_metadata_only_repair(
            client,
            matchids=list(classes.metadata_only),
            backup_prefix=metadata_backup_prefix,
            apply=args.apply,
        )
        _print_backups(backups)
        if args.apply:
            print(
                "Metadata-only repair complete; "
                f"inserted={len(classes.metadata_only)} run_id={repair_run_id}"
            )

    if classes.stream_partial:
        if args.allow_full_requeue:
            backups, missing = _apply_full_requeue(
                client,
                matchids=list(classes.stream_partial),
                backup_prefix=full_backup_prefix,
                apply=args.apply,
            )
            _print_backups(backups)
            if args.apply:
                print(
                    "Stream-partial repair complete; "
                    f"{missing} matchids were newly queued."
                )
            else:
                print(
                    "Dry run complete; "
                    f"{missing} stream-partial matchids would be queued."
                )
        else:
            print(
                "Skipped stream-partial full requeue; pass --allow-full-requeue "
                f"to process {len(classes.stream_partial)} matchids."
            )
            if args.apply and not classes.metadata_only:
                raise SystemExit(
                    "Refusing to full-requeue stream-partial matchids without "
                    "--allow-full-requeue."
                )

    if not classes.metadata_only and not classes.stream_partial:
        print("No applicable repair classes found.")
    elif not args.apply and classes.metadata_only:
        print(
            "Dry run complete; "
            f"{len(classes.metadata_only)} metadata rows would be inserted and "
            "their queue rows would be cleared after validation."
        )


if __name__ == "__main__":
    main()
