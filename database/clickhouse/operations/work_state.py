from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from uuid import UUID

from app.core.config.constants import CONTINENT_TO_REGIONS, Continent, Region
from database.clickhouse.client import get_client
from database.clickhouse.operations.matchids import PUUID_DATA_TIMESTAMP_NAME
from database.clickhouse.operations.utils import dedupe_matchids, record_timestamp

MATCHDATA_STATE_TABLE = "game_data.matchdata_matchids"
MATCHDATA_AVAILABLE_SEEDED_NAME = "matchdata_seeded_available_matchids_run"
CONTINENTS: tuple[str, ...] = tuple(c.value for c in Continent)
REGIONS: tuple[str, ...] = tuple(r.value for r in Region)
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


def _seed_candidates_select() -> str:
    return f"""
        WITH source_matchids AS
        (
            SELECT
                any(run_id) AS run_id,
                toString(matchid) AS matchid
            FROM game_data.matchids
            WHERE matchid != ''
            GROUP BY matchid
        ),
        info_matchids AS
        (
            SELECT DISTINCT matchid FROM game_data.info WHERE matchid != ''
        ),
        timeline_matchids AS
        (
            SELECT DISTINCT matchid FROM game_data.tl_game_end WHERE matchid != ''
        ),
        completed_matchids AS
        (
            SELECT info_matchids.matchid
            FROM info_matchids
            INNER JOIN timeline_matchids USING (matchid)
        )
        SELECT DISTINCT
            source_matchids.run_id AS run_id,
            source_matchids.matchid AS matchid
        FROM source_matchids
        WHERE source_matchids.matchid NOT IN completed_matchids
          AND source_matchids.matchid NOT IN (
              SELECT DISTINCT matchid FROM {MATCHDATA_STATE_TABLE}
          )
    """


def seed_from_matchids() -> int:
    client = get_client()
    latest_run_id = _load_latest_run_id(client=client, name=PUUID_DATA_TIMESTAMP_NAME)
    if latest_run_id is None:
        return 0

    if (
        _load_latest_run_id(client=client, name=MATCHDATA_AVAILABLE_SEEDED_NAME)
        == latest_run_id
    ):
        logger.debug(
            "Matchdata seed skipped latest_run_id=%s (available inventory already seeded)",
            latest_run_id,
        )
        return 0

    candidates_select = _seed_candidates_select()
    rows = client.query(
        f"""
        SELECT count()
        FROM ({candidates_select})
        """
    ).result_rows
    pending = int(rows[0][0]) if rows else 0
    if pending == 0:
        record_timestamp(
            MATCHDATA_AVAILABLE_SEEDED_NAME, latest_run_id, int(time.time())
        )
        return 0

    client.command(
        f"""
        INSERT INTO {MATCHDATA_STATE_TABLE} (run_id, matchid)
        {candidates_select}
        """
    )
    record_timestamp(MATCHDATA_AVAILABLE_SEEDED_NAME, latest_run_id, int(time.time()))
    logger.debug(
        "Seeded matchdata queue rows=%d latest_run_id=%s source=available_matchids",
        pending,
        latest_run_id,
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
                    cityHash64('matchdata_claim', matchid) AS shuffle_key
                FROM {MATCHDATA_STATE_TABLE}
                ORDER BY continent, shuffle_key, matchid
                LIMIT %(limit)s BY continent
            ),
            ranked AS (
                SELECT
                    matchid,
                    continent,
                    shuffle_key,
                    row_number() OVER (
                        PARTITION BY continent
                        ORDER BY shuffle_key, matchid
                    ) AS row_n,
                    transform(
                        continent,
                        {_sql_strings(CONTINENTS, brackets="[]")},
                        arrayEnumerate({_sql_strings(CONTINENTS, brackets="[]")}),
                        length({_sql_strings(CONTINENTS, brackets="[]")}) + 1
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
        continent = SHARD_TO_CONTINENT.get(shard)
        if continent is None:
            unknown += 1
        else:
            counts[continent] += 1
    exhausted = [continent for continent, count in counts.items() if count == 0]

    logger.debug(
        "Claimed matchdata queue rows=%d continent_counts=%s exhausted_continents=%s unknown=%d",
        len(claimed),
        counts,
        exhausted,
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
