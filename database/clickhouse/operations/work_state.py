from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from uuid import UUID

from app.core.config.constants import CONTINENT_TO_REGIONS, Continent
from database.clickhouse.client import get_client
from database.clickhouse.operations.matchids import PUUID_DATA_TIMESTAMP_NAME
from database.clickhouse.operations.utils import _as_text

MATCHDATA_STATE_TABLE = "game_data.matchdata_matchids"
MATCHDATA_SEEDED_RUN_NAME = "matchdata_seeded_matchids_run"
CONTINENTS: tuple[str, ...] = tuple(continent.value for continent in Continent)
CONTINENT_SHARDS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (
        continent.value,
        tuple(region.value for region in CONTINENT_TO_REGIONS[continent]),
    )
    for continent in Continent
)
SHARD_TO_CONTINENT = {
    shard: continent for continent, shards in CONTINENT_SHARDS for shard in shards
}

logger = logging.getLogger(__name__)


def _sql_strings(values: Iterable[str], *, brackets: str = "()") -> str:
    left, right = brackets
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"{left}{quoted}{right}"


def _continent_alias_sql(matchid_column: str = "matchid") -> str:
    shard_expr = f"lower(splitByChar('_', {matchid_column})[1])"
    cases = ",\n                ".join(
        f"{shard_expr} IN {_sql_strings(shards)}, '{continent}'"
        for continent, shards in CONTINENT_SHARDS
    )
    return (
        f"multiIf(\n                {cases},\n                'unknown'\n            )"
    )


def _load_latest_run_id(*, client, name: str) -> UUID | None:
    rows = client.query(
        """
        SELECT argMax(run_id, stored_at)
        FROM game_data.data_timestamps
        WHERE name = %(name)s
        """,
        parameters={"name": name},
    ).result_rows
    return None if not rows or rows[0][0] is None else rows[0][0]


def _mark_seeded(client, run_id: UUID) -> None:
    client.insert(
        table="game_data.data_timestamps",
        data=[(run_id, MATCHDATA_SEEDED_RUN_NAME, int(time.time()))],
        column_names=("run_id", "name", "stored_at"),
    )


# RECOVERY-SYSTEM: basic matchdata queue helpers.
def ensure_matchdata_state_schema() -> None:
    client = get_client()
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {MATCHDATA_STATE_TABLE}
        (
            run_id UUID,
            matchid String,
            continent LowCardinality(String) ALIAS {_continent_alias_sql()},
            shuffle_key UInt64 ALIAS cityHash64(matchid)
        )
        ENGINE = MergeTree
        ORDER BY (matchid, run_id)
        """
    )
    client.command(
        f"""
        ALTER TABLE {MATCHDATA_STATE_TABLE}
        ADD COLUMN IF NOT EXISTS continent LowCardinality(String) ALIAS {_continent_alias_sql()}
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
    latest_run_id = _load_latest_run_id(client=client, name=PUUID_DATA_TIMESTAMP_NAME)
    if latest_run_id is None:
        return 0

    if (
        _load_latest_run_id(client=client, name=MATCHDATA_SEEDED_RUN_NAME)
        == latest_run_id
    ):
        logger.debug(
            "Matchdata seed skipped latest_run_id=%s (already seeded)", latest_run_id
        )
        return 0

    rows = client.query(
        f"""
        WITH completed_matchids AS
        (
            SELECT DISTINCT matchid
            FROM game_data.info
            WHERE matchid != ''
              AND endofgameresult LIKE 'Abort%%'
            UNION DISTINCT
            SELECT DISTINCT matchid
            FROM game_data.tl_payload_event
            WHERE matchid != ''
              AND type = 'GAME_END'
        )
        SELECT DISTINCT
            m.run_id AS run_id,
            toString(m.matchid) AS matchid
        FROM game_data.matchids AS m
        WHERE m.run_id = %(run_id)s
          AND toString(m.matchid) NOT IN completed_matchids
          AND toString(m.matchid) NOT IN
          (
              SELECT DISTINCT matchid
              FROM {MATCHDATA_STATE_TABLE}
          )
        """,
        parameters={"run_id": latest_run_id},
    ).result_rows
    if not rows:
        _mark_seeded(client, latest_run_id)
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
    _mark_seeded(client, latest_run_id)
    logger.debug(
        "Seeded matchdata queue rows=%d latest_run_id=%s", len(data), latest_run_id
    )
    return len(data)


def claim_pending_matchids(*, batch_size: int) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    rows = (
        get_client()
        .query(
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
                    {_sql_strings(CONTINENTS, brackets="[]")},
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
        )
        .result_rows
    )
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


def _continent_for_matchid(matchid: str) -> str:
    shard = matchid.split("_", 1)[0].lower()
    return SHARD_TO_CONTINENT.get(shard, "unknown")
