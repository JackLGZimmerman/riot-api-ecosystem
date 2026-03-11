from __future__ import annotations

import logging
from typing import Iterable
from uuid import UUID

from database.clickhouse.client import get_client
from database.clickhouse.operations.matchids import PUUID_DATA_TIMESTAMP_NAME

MATCHDATA_STATE_TABLE = "game_data.matchdata_matchids"

logger = logging.getLogger("app.services.riot_api_client.rate_limiter")


# RECOVERY-SYSTEM: basic matchdata status queue helpers.
def ensure_matchdata_state_schema() -> None:
    client = get_client()
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {MATCHDATA_STATE_TABLE}
        (
            run_id UUID,
            matchid String,
            status LowCardinality(String) DEFAULT 'pending',
            last_error String DEFAULT ''
        )
        ENGINE = MergeTree
        ORDER BY (status, matchid, run_id)
        """
    )
    client.command(
        f"""
        ALTER TABLE {MATCHDATA_STATE_TABLE}
        ADD COLUMN IF NOT EXISTS status LowCardinality(String) DEFAULT 'pending'
        """
    )
    client.command(
        f"""
        ALTER TABLE {MATCHDATA_STATE_TABLE}
        ADD COLUMN IF NOT EXISTS last_error String DEFAULT ''
        """
    )


def seed_from_latest_matchids() -> int:
    latest_run_rows = get_client().query(
        """
        SELECT argMax(run_id, stored_at)
        FROM game_data.data_timestamps
        WHERE name = %(name)s
        """,
        parameters={"name": PUUID_DATA_TIMESTAMP_NAME},
    ).result_rows
    if not latest_run_rows or latest_run_rows[0][0] is None:
        return 0

    latest_run_id = latest_run_rows[0][0]
    rows = get_client().query(
        f"""
        SELECT DISTINCT
            m.run_id,
            toString(m.matchid) AS matchid
        FROM game_data.matchids AS m
        WHERE m.run_id = %(run_id)s
          AND toString(m.matchid) NOT IN
          (
              SELECT DISTINCT matchid
              FROM {MATCHDATA_STATE_TABLE}
          )
        """,
        parameters={"run_id": latest_run_id},
    ).result_rows
    if not rows:
        return 0

    data: list[tuple[UUID, str, str]] = [
        (
            run_id,
            _as_text(matchid),
            "pending",
        )
        for run_id, matchid in rows
    ]
    get_client().insert(
        table=MATCHDATA_STATE_TABLE,
        data=data,
        column_names=("run_id", "matchid", "status"),
    )
    logger.debug(
        "Seeded matchdata state pending=%d latest_run_id=%s",
        len(data),
        latest_run_id,
    )
    return len(data)


def claim_pending_matchids(*, batch_size: int) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    rows = get_client().query(
        f"""
        SELECT matchid
        FROM {MATCHDATA_STATE_TABLE}
        WHERE status = 'pending'
        ORDER BY matchid
        LIMIT %(limit)s
        """,
        parameters={"limit": batch_size},
    ).result_rows
    claimed = _dedupe(_as_text(row[0]) for row in rows)
    logger.debug("Claimed matchdata pending rows=%d", len(claimed))
    return claimed


def mark_matchids_finished(match_ids: Iterable[str]) -> None:
    ids = _dedupe(match_ids)
    if not ids:
        return
    get_client().command(
        f"""
        ALTER TABLE {MATCHDATA_STATE_TABLE}
        UPDATE
            status = 'finished',
            last_error = ''
        WHERE has(%(match_ids)s, matchid)
        SETTINGS mutations_sync = 2
        """,
        parameters={"match_ids": ids},
    )
    logger.debug("Marked matchdata finished rows=%d", len(ids))


def mark_matchids_pending(match_ids: Iterable[str], *, error: str) -> None:
    ids = _dedupe(match_ids)
    if not ids:
        return
    get_client().command(
        f"""
        ALTER TABLE {MATCHDATA_STATE_TABLE}
        UPDATE
            status = 'pending',
            last_error = %(error)s
        WHERE has(%(match_ids)s, matchid)
        SETTINGS mutations_sync = 2
        """,
        parameters={
            "match_ids": ids,
            "error": error,
        },
    )
    logger.debug("Marked matchdata pending rows=%d", len(ids))


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _as_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _as_text(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8").rstrip("\x00")
    return str(value)
