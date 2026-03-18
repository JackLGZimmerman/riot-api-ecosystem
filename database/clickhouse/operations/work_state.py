from __future__ import annotations

import logging
import time
from typing import Iterable
from uuid import UUID

from database.clickhouse.client import get_client
from database.clickhouse.operations.matchids import PUUID_DATA_TIMESTAMP_NAME

MATCHDATA_STATE_TABLE = "game_data.matchdata_matchids"
MATCHDATA_SEEDED_RUN_NAME = "matchdata_seeded_matchids_run"
CONTINENTS: tuple[str, ...] = ("americas", "europe", "asia", "sea")

logger = logging.getLogger("app.services.riot_api_client.rate_limiter")


# RECOVERY-SYSTEM: basic matchdata queue helpers.
def ensure_matchdata_state_schema() -> None:
    client = get_client()
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {MATCHDATA_STATE_TABLE}
        (
            run_id UUID,
            matchid String,
            continent LowCardinality(String) ALIAS multiIf(
                lower(splitByChar('_', matchid)[1]) IN ('br1', 'la1', 'la2', 'na1'), 'americas',
                lower(splitByChar('_', matchid)[1]) IN ('euw1', 'eun1', 'ru', 'tr1', 'me1'), 'europe',
                lower(splitByChar('_', matchid)[1]) IN ('jp1', 'kr'), 'asia',
                lower(splitByChar('_', matchid)[1]) IN ('ph2', 'th2', 'tw2', 'oc1', 'vn2', 'sg2'), 'sea',
                'unknown'
            ),
            shuffle_key UInt64 ALIAS cityHash64(matchid)
        )
        ENGINE = MergeTree
        ORDER BY (matchid, run_id)
        """
    )
    client.command(
        f"""
        ALTER TABLE {MATCHDATA_STATE_TABLE}
        DROP COLUMN IF EXISTS status
        """
    )
    client.command(
        f"""
        ALTER TABLE {MATCHDATA_STATE_TABLE}
        DROP COLUMN IF EXISTS last_error
        """
    )
    client.command(
        f"""
        ALTER TABLE {MATCHDATA_STATE_TABLE}
        ADD COLUMN IF NOT EXISTS continent LowCardinality(String) ALIAS multiIf(
            lower(splitByChar('_', matchid)[1]) IN ('br1', 'la1', 'la2', 'na1'), 'americas',
            lower(splitByChar('_', matchid)[1]) IN ('euw1', 'eun1', 'ru', 'tr1', 'me1'), 'europe',
            lower(splitByChar('_', matchid)[1]) IN ('jp1', 'kr'), 'asia',
            lower(splitByChar('_', matchid)[1]) IN ('ph2', 'th2', 'tw2', 'oc1', 'vn2', 'sg2'), 'sea',
            'unknown'
        )
        """
    )
    client.command(
        f"""
        ALTER TABLE {MATCHDATA_STATE_TABLE}
        ADD COLUMN IF NOT EXISTS shuffle_key UInt64 ALIAS cityHash64(matchid)
        """
    )


def seed_from_latest_matchids() -> int:
    client = get_client()
    latest_run_rows = client.query(
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
    seeded_run_rows = client.query(
        """
        SELECT argMax(run_id, stored_at)
        FROM game_data.data_timestamps
        WHERE name = %(name)s
        """,
        parameters={"name": MATCHDATA_SEEDED_RUN_NAME},
    ).result_rows
    if seeded_run_rows and seeded_run_rows[0][0] == latest_run_id:
        logger.debug("Matchdata seed skipped latest_run_id=%s (already seeded)", latest_run_id)
        return 0

    rows = client.query(
        f"""
        SELECT DISTINCT
            m.run_id AS run_id,
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
        client.insert(
            table="game_data.data_timestamps",
            data=[(latest_run_id, MATCHDATA_SEEDED_RUN_NAME, int(time.time()))],
            column_names=("run_id", "name", "stored_at"),
        )
        return 0

    data: list[tuple[UUID, str]] = [
        (
            run_id,
            _as_text(matchid),
        )
        for run_id, matchid in rows
    ]
    client.insert(
        table=MATCHDATA_STATE_TABLE,
        data=data,
        column_names=("run_id", "matchid"),
    )
    client.insert(
        table="game_data.data_timestamps",
        data=[(latest_run_id, MATCHDATA_SEEDED_RUN_NAME, int(time.time()))],
        column_names=("run_id", "name", "stored_at"),
    )
    logger.debug("Seeded matchdata queue rows=%d latest_run_id=%s", len(data), latest_run_id)
    return len(data)


def claim_pending_matchids(*, batch_size: int) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    rows = get_client().query(
        f"""
        WITH
        limited AS
        (
            SELECT
                matchid,
                continent,
                shuffle_key
            FROM {MATCHDATA_STATE_TABLE}
            ORDER BY continent, shuffle_key, matchid
            LIMIT %(limit)s BY continent
        ),
        ranked AS
        (
            SELECT
                matchid,
                shuffle_key,
                row_number() OVER (
                    PARTITION BY continent
                    ORDER BY shuffle_key, matchid
                ) AS row_n,
                transform(
                    continent,
                    ['americas', 'europe', 'asia', 'sea'],
                    [1, 2, 3, 4],
                    5
                ) AS continent_order
            FROM limited
        )
        SELECT matchid
        FROM ranked
        ORDER BY row_n, continent_order, shuffle_key, matchid
        LIMIT %(limit)s
        """,
        parameters={"limit": batch_size},
    ).result_rows
    claimed = _dedupe(_as_text(row[0]) for row in rows)

    per_continent_counts: dict[str, int] = {continent: 0 for continent in CONTINENTS}
    unknown_count = 0
    for matchid in claimed:
        continent = _continent_for_matchid(matchid)
        if continent in per_continent_counts:
            per_continent_counts[continent] += 1
        else:
            unknown_count += 1

    logger.debug(
        "Claimed matchdata queue rows=%d counts=%s unknown=%d",
        len(claimed),
        per_continent_counts,
        unknown_count,
    )
    return claimed


def mark_matchids_finished(match_ids: Iterable[str]) -> None:
    ids = _dedupe(match_ids)
    if not ids:
        return
    get_client().command(
        f"""
        ALTER TABLE {MATCHDATA_STATE_TABLE}
        DELETE
        WHERE has(%(match_ids)s, matchid)
        SETTINGS mutations_sync = 2
        """,
        parameters={"match_ids": ids},
    )
    logger.debug("Removed matchdata queue rows=%d", len(ids))


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


def _continent_for_matchid(matchid: str) -> str:
    shard = matchid.split("_", 1)[0].lower()
    if shard in {"br1", "la1", "la2", "na1"}:
        return "americas"
    if shard in {"euw1", "eun1", "ru", "tr1", "me1"}:
        return "europe"
    if shard in {"jp1", "kr"}:
        return "asia"
    if shard in {"ph2", "th2", "tw2", "oc1", "vn2", "sg2"}:
        return "sea"
    return "unknown"
