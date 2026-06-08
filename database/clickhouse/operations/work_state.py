from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from uuid import UUID

from app.core.config.constants import CONTINENT_TO_REGIONS, Continent
from database.clickhouse.client import get_client
from database.clickhouse.operations.matchids import PUUID_DATA_TIMESTAMP_NAME
from database.clickhouse.operations.utils import dedupe_matchids, record_timestamp

MATCHDATA_STATE_TABLE = "game_data.matchdata_matchids"
MATCHDATA_SEEDED_RUN_NAME = "matchdata_seeded_matchids_run"
CONTINENTS: tuple[str, ...] = tuple(c.value for c in Continent)
CONTINENT_SHARDS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (c.value, tuple(r.value for r in CONTINENT_TO_REGIONS[c])) for c in Continent
)
SHARD_TO_CONTINENT: dict[str, str] = {
    shard: continent for continent, shards in CONTINENT_SHARDS for shard in shards
}

logger = logging.getLogger(__name__)


def _sql_strings(values: Iterable[str], *, brackets: str = "()") -> str:
    left, right = brackets
    return f"{left}{', '.join(f'\'{v}\'' for v in values)}{right}"


def _continent_expr(matchid_column: str = "matchid") -> str:
    shard = f"lower(splitByChar('_', {matchid_column})[1])"
    cases = ",\n            ".join(
        f"{shard} IN {_sql_strings(shards)}, '{continent}'"
        for continent, shards in CONTINENT_SHARDS
    )
    return f"multiIf(\n            {cases},\n            'unknown'\n        )"


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


def _mark_seeded(run_id: UUID) -> None:
    record_timestamp(MATCHDATA_SEEDED_RUN_NAME, run_id, int(time.time()))


def _seed_candidates_select() -> str:
    return f"""
        WITH completed_matchids AS
        (
            SELECT DISTINCT matchid FROM game_data.info WHERE matchid != ''
            UNION DISTINCT
            SELECT DISTINCT matchid FROM game_data.tl_game_end WHERE matchid != ''
        )
        SELECT DISTINCT
            m.run_id AS run_id,
            toString(m.matchid) AS matchid
        FROM game_data.matchids AS m
        WHERE m.run_id = %(run_id)s
          AND toString(m.matchid) NOT IN completed_matchids
          AND toString(m.matchid) NOT IN (
              SELECT DISTINCT matchid FROM {MATCHDATA_STATE_TABLE}
          )
    """


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

    candidates_select = _seed_candidates_select()
    rows = client.query(
        f"""
        SELECT count()
        FROM ({candidates_select})
        """,
        parameters={"run_id": latest_run_id},
    ).result_rows
    pending = int(rows[0][0]) if rows else 0
    if pending == 0:
        _mark_seeded(latest_run_id)
        return 0

    client.command(
        f"""
        INSERT INTO {MATCHDATA_STATE_TABLE} (run_id, matchid)
        {candidates_select}
        """,
        parameters={"run_id": latest_run_id},
    )
    _mark_seeded(latest_run_id)
    logger.debug(
        "Seeded matchdata queue rows=%d latest_run_id=%s", pending, latest_run_id
    )
    return pending


def claim_pending_matchids(*, batch_size: int) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    rows = (
        get_client()
        .query(
            f"""
            WITH limited AS (
                SELECT
                    matchid,
                    {_continent_expr()} AS continent,
                    cityHash64(matchid) AS shuffle_key
                FROM {MATCHDATA_STATE_TABLE}
                ORDER BY continent, shuffle_key, matchid
                LIMIT %(limit)s BY continent
            ),
            ranked AS (
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
    claimed = dedupe_matchids(row[0] for row in rows)

    counts: dict[str, int] = {c: 0 for c in CONTINENTS}
    unknown = 0
    for matchid in claimed:
        shard = matchid.split("_", 1)[0].lower()
        c = SHARD_TO_CONTINENT.get(shard)
        if c is None:
            unknown += 1
        else:
            counts[c] += 1

    logger.debug(
        "Claimed matchdata queue rows=%d counts=%s unknown=%d",
        len(claimed),
        counts,
        unknown,
    )
    return claimed


def mark_matchids_finished(match_ids: Iterable[str]) -> None:
    ids = dedupe_matchids(match_ids)
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
