"""Materialise classification sufficient-statistic tables in ClickHouse.

One-time server-side builds; the loaders ([load.py], [temporal.py]) then read the
prepared tables and run only the shared smoother + standardisation. Storing raw
sufficient statistics (SUM, COUNT, SUM_timeplayed) makes every prior level an
exact SQL GROUP BY rollup. Heavy scans use shard -> stage -> GROUP BY combine to
bound peak memory.

Tables (one row per ``split, championid, teamposition, build`` unless noted):

* ``classification_identity_base`` — participant-stats sums + ``matchups`` + ``sum_w_timeplayed``.
* ``classification_final_base`` — final-snapshot sums (denominator = ``matchups``).
* ``classification_context_base`` — team-share / matchup sums and their counts.
* ``temporal_identity_bins`` — per (…, bucket): ``frames`` + a SUM per metric.
* ``classification_base_meta`` — 1-row ``catalogue_hash`` guard.
"""

from __future__ import annotations

import logging

from app.classification.embeddings.config import (
    FINAL_PARTICIPANT_STATS_TABLE,
    ITEM_VALUE_TOTALS_TABLE,
    ML_GAME_SPLIT_TABLE,
    PARTICIPANT_STATS_TABLE,
)
from app.classification.embeddings.context_features import (
    MATCHUP_FEATURE_NAMES,
    TEAM_FEATURE_NAMES,
    matchup_query,
    team_share_query,
)
from app.classification.embeddings.registry import (
    FINAL_SNAPSHOT_AVG_METRICS,
    LARGEST_AVG_METRICS,
    PER_MINUTE_METRICS,
    RATE_METRICS,
    catalogue_hash,
)
from app.classification.embeddings import temporal as T
from app.core.utils.common import sql_literal
from app.core.utils.smoothing import build_group_sql
from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)

DB = "game_data_filtered"
IDENTITY_BASE = f"{DB}.classification_identity_base"
FINAL_BASE = f"{DB}.classification_final_base"
CONTEXT_BASE = f"{DB}.classification_context_base"
META_TABLE = f"{DB}.classification_base_meta"
_FINAL_STAGE = f"{DB}.classification_final_stage"
_TEAM_STAGE = f"{DB}.classification_context_team_stage"
_MATCHUP_STAGE = f"{DB}.classification_context_matchup_stage"
_STAT_STAGE = f"{DB}.temporal_stat_stage"
_EV_STAGE = f"{DB}.temporal_ev_stage"

SPLITS: tuple[str, ...] = ("train", "test")
K_SHARDS = 8  # match-hash shards for the heavy tl_participant_stats / team scans

# Participant-stats metrics whose raw SUM is materialised (rate + largest-avg use
# matchups as denominator; per-minute use sum_w_timeplayed).
IDENTITY_SUM_METRICS: tuple[str, ...] = (
    *RATE_METRICS,
    *LARGEST_AVG_METRICS,
    *PER_MINUTE_METRICS,
)
_NULLABLE_ZERO = {"damagedealttoepicmonsters"}

_HEAVY = (
    "SETTINGS max_threads = 4,"
    " max_bytes_before_external_group_by = 2000000000,"
    " join_algorithm = 'grace_hash'"
)
_KEYS = ("split", "championid", "teamposition", "build")
_KEY_DDL = (
    "split LowCardinality(String),\n"
    "    championid Int32,\n"
    "    teamposition LowCardinality(String),\n"
    "    build LowCardinality(String)"
)


def _cols(names: tuple[str, ...], prefix: str, ctype: str = "Float64") -> str:
    return ",\n    ".join(f"{prefix}{n} {ctype}" for n in names)


def _identity_ddl() -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {IDENTITY_BASE}
(
    {_KEY_DDL},
    build_group LowCardinality(String),
    matchups UInt64,
    sum_w_timeplayed Float64,
    {_cols(IDENTITY_SUM_METRICS, "sum_")}
)
ENGINE = MergeTree ORDER BY (split, championid, teamposition, build)
"""


def _final_ddl(table: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {table}
(
    {_KEY_DDL},
    {_cols(FINAL_SNAPSHOT_AVG_METRICS, "sum_final_")}
)
ENGINE = MergeTree ORDER BY (split, championid, teamposition, build)
"""


def _context_stage_ddl(table: str, feature_names: tuple[str, ...]) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {table}
(
    {_KEY_DDL},
    cnt UInt64,
    {_cols(feature_names, "sum_")}
)
ENGINE = MergeTree ORDER BY (split, championid, teamposition, build)
"""


def _context_base_ddl() -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {CONTEXT_BASE}
(
    {_KEY_DDL},
    cnt_team UInt64,
    {_cols(TEAM_FEATURE_NAMES, "sum_")},
    cnt_matchup UInt64,
    {_cols(MATCHUP_FEATURE_NAMES, "sum_")}
)
ENGINE = MergeTree ORDER BY (split, championid, teamposition, build)
"""


def _temporal_stat_ddl(table: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {table}
(
    {_KEY_DDL},
    bucket UInt8,
    frames UInt64,
    {_cols(T.TEMPORAL_METRICS, "sum_")}
)
ENGINE = MergeTree ORDER BY (split, championid, teamposition, build, bucket)
"""


def _temporal_ev_ddl(table: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {table}
(
    {_KEY_DDL},
    bucket UInt8,
    {_cols(T.EVENT_METRICS, "ev_")}
)
ENGINE = MergeTree ORDER BY (split, championid, teamposition, build, bucket)
"""


def _bins_ddl() -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {T.BINS_TABLE}
(
    {_KEY_DDL},
    bucket UInt8,
    frames UInt64,
    {_cols(T.TEMPORAL_METRICS, "sum_")},
    {_cols(T.EVENT_METRICS, "ev_")}
)
ENGINE = MergeTree ORDER BY (split, championid, teamposition, build, bucket)
"""


def _meta_ddl() -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {META_TABLE}
(catalogue_hash String, built_at DateTime DEFAULT now())
ENGINE = MergeTree ORDER BY tuple()
"""


def _range_cond(col: str, shard: int, bounds: tuple[str, ...]) -> str:
    """Shard ``shard`` as a ``matchid`` range (``bounds`` = K-1 interior cuts). A
    range on the leading ORDER BY key prunes granules; ``cityHash64 % K`` cannot,
    forcing a full 1.1B-row scan (the temporal-build OOM)."""
    parts = []
    if shard > 0:
        parts.append(f"{col} >= {sql_literal(bounds[shard - 1])}")
    if shard < len(bounds):
        parts.append(f"{col} < {sql_literal(bounds[shard])}")
    return " AND ".join(parts) if parts else "1"


def _matchid_bounds(client, k: int) -> tuple[str, ...]:
    """K-1 matchid cut points splitting the timeline into K equal contiguous ranges."""
    rows = client.query(
        f"""
SELECT arr[idx] FROM (
    SELECT groupArray(m) AS arr FROM (
        SELECT DISTINCT matchid AS m FROM {FINAL_PARTICIPANT_STATS_TABLE} ORDER BY m
    )
) ARRAY JOIN arrayMap(i -> toUInt32(intDiv(length(arr) * i, {k})), range(1, {k})) AS idx
"""
    ).result_rows
    return tuple(str(r[0]) for r in rows)


def _shard_pred(col: str, shard: int | None, bounds: tuple[str, ...] | None = None) -> str:
    if shard is None:
        return ""
    if bounds is None:
        return f"\n        AND cityHash64({col}) % {K_SHARDS} = {shard}"
    return f"\n        AND {_range_cond(col, shard, bounds)}"


def _identity_cte(
    split: str,
    *,
    stat_columns: tuple[str, ...],
    shard: int | None = None,
    bounds: tuple[str, ...] | None = None,
) -> str:
    """participant-grain rows (matchid, participantid, identity keys + stats)."""
    extra = "".join(f",\n        ps.{c} AS {c}" for c in stat_columns)
    return f"""
participant_context AS (
    SELECT
        ps.matchid AS matchid,
        ps.participantid AS participantid,
        s.split AS split,
        assumeNotNull(ps.championid) AS championid,
        toString(ps.teamposition) AS teamposition,
        toString(ivt.highest_value_label) AS build{extra}
    FROM {PARTICIPANT_STATS_TABLE} AS ps
    INNER JOIN {ML_GAME_SPLIT_TABLE} AS s ON ps.matchid = s.matchid
    INNER JOIN {ITEM_VALUE_TOTALS_TABLE} AS ivt
        ON ps.matchid = ivt.matchid AND ps.participantid = ivt.participantid
    WHERE s.split = {sql_literal(split)}
        AND isNotNull(ps.championid)
        AND toString(ps.teamposition) != 'UNKNOWN'{_shard_pred("ps.matchid", shard, bounds)}
)
"""


def _identity_insert(split: str) -> str:
    aggs = []
    for m in IDENTITY_SUM_METRICS:
        value = f"coalesce(pc.{m}, 0)" if m in _NULLABLE_ZERO else f"pc.{m}"
        aggs.append(f"toFloat64(sum({value})) AS sum_{m}")
    return f"""
INSERT INTO {IDENTITY_BASE}
WITH {_identity_cte(split, stat_columns=("timeplayed", *IDENTITY_SUM_METRICS))}
SELECT
    pc.split, pc.championid, pc.teamposition, pc.build,
    {build_group_sql("pc.build")},
    toUInt64(count()) AS matchups,
    toFloat64(sum(pc.timeplayed)) AS sum_w_timeplayed,
    {",\n    ".join(aggs)}
FROM participant_context AS pc
GROUP BY split, championid, teamposition, build, build_group
{_HEAVY}
"""


def _final_insert(split: str, shard: int) -> str:
    tuple_cols = ",\n                ".join(FINAL_SNAPSHOT_AVG_METRICS)
    sums = ",\n    ".join(
        f"toFloat64(sum(tupleElement(fs.final_stats, {i}))) AS sum_final_{m}"
        for i, m in enumerate(FINAL_SNAPSHOT_AVG_METRICS, start=1)
    )
    return f"""
INSERT INTO {_FINAL_STAGE}
WITH final_snapshot AS (
    SELECT matchid, participantid,
        argMax(tuple(
                {tuple_cols}
        ), frame_timestamp) AS final_stats
    FROM {FINAL_PARTICIPANT_STATS_TABLE}
    WHERE matchid IN (
        SELECT matchid FROM {ML_GAME_SPLIT_TABLE} WHERE split = {sql_literal(split)}
    ){_shard_pred("matchid", shard)}
    GROUP BY matchid, participantid
),
{_identity_cte(split, stat_columns=(), shard=shard)}
SELECT
    pc.split, pc.championid, pc.teamposition, pc.build,
    {sums}
FROM participant_context AS pc
LEFT JOIN final_snapshot AS fs
    ON pc.matchid = fs.matchid AND pc.participantid = fs.participantid
GROUP BY split, championid, teamposition, build
{_HEAVY}
"""


def _final_combine(split: str) -> str:
    sums = ",\n    ".join(
        f"sum(sum_final_{m}) AS sum_final_{m}" for m in FINAL_SNAPSHOT_AVG_METRICS
    )
    return f"""
INSERT INTO {FINAL_BASE}
SELECT split, championid, teamposition, build,
    {sums}
FROM {_FINAL_STAGE}
WHERE split = {sql_literal(split)}
GROUP BY split, championid, teamposition, build
"""


def _context_stage_insert(stage: str, query: str, feature_names: tuple[str, ...], split: str) -> str:
    cols = ", ".join(("split", *_KEYS[1:], "cnt", *(f"sum_{n}" for n in feature_names)))
    return f"""
INSERT INTO {stage} ({cols})
SELECT {sql_literal(split)} AS split, q.*
FROM (
{query}
) AS q
"""


def _context_combine(split: str) -> str:
    team_sums = ",\n        ".join(
        f"sum(sum_{n}) AS sum_{n}" for n in TEAM_FEATURE_NAMES
    )
    matchup_sums = ",\n        ".join(
        f"sum(sum_{n}) AS sum_{n}" for n in MATCHUP_FEATURE_NAMES
    )
    team_out = ",\n    ".join(f"t.sum_{n}" for n in TEAM_FEATURE_NAMES)
    matchup_out = ",\n    ".join(f"ifNull(m.sum_{n}, 0)" for n in MATCHUP_FEATURE_NAMES)
    lit = sql_literal(split)
    return f"""
INSERT INTO {CONTEXT_BASE}
SELECT
    t.split, t.championid, t.teamposition, t.build,
    t.cnt_team,
    {team_out},
    ifNull(m.cnt_matchup, 0) AS cnt_matchup,
    {matchup_out}
FROM (
    SELECT split, championid, teamposition, build,
        sum(cnt) AS cnt_team,
        {team_sums}
    FROM {_TEAM_STAGE} WHERE split = {lit}
    GROUP BY split, championid, teamposition, build
) AS t
LEFT JOIN (
    SELECT split, championid, teamposition, build,
        sum(cnt) AS cnt_matchup,
        {matchup_sums}
    FROM {_MATCHUP_STAGE} WHERE split = {lit}
    GROUP BY split, championid, teamposition, build
) AS m USING (split, championid, teamposition, build)
"""


def _temporal_stat_insert(split: str, shard: int, bounds: tuple[str, ...]) -> str:
    sums = ",\n        ".join(f"sum(t.{m}) AS sum_{m}" for m in T.TEMPORAL_METRICS)
    return f"""
INSERT INTO {_STAT_STAGE}
WITH {_identity_cte(split, stat_columns=(), shard=shard, bounds=bounds)}
SELECT
    pc.split, pc.championid, pc.teamposition, pc.build,
    {T._bucket("t.frame_timestamp")} AS bucket,
    count() AS frames,
    {sums}
FROM {FINAL_PARTICIPANT_STATS_TABLE} AS t
INNER JOIN participant_context AS pc
    ON t.matchid = pc.matchid AND t.participantid = pc.participantid
WHERE {_range_cond("t.matchid", shard, bounds)}
GROUP BY split, championid, teamposition, build, bucket
{_HEAVY}
"""


def _temporal_ev_insert(split: str, shard: int, bounds: tuple[str, ...]) -> str:
    ck = T._bucket("timestamp")
    rc = _range_cond("matchid", shard, bounds)
    return f"""
INSERT INTO {_EV_STAGE}
WITH {_identity_cte(split, stat_columns=(), shard=shard, bounds=bounds)}
SELECT
    pc.split, pc.championid, pc.teamposition, pc.build,
    e.bucket AS bucket,
    sum(e.kills) AS ev_kills,
    sum(e.assists) AS ev_assists,
    sum(e.deaths) AS ev_deaths,
    sum(e.plate_top) AS ev_plate_top,
    sum(e.plate_mid) AS ev_plate_mid,
    sum(e.plate_bot) AS ev_plate_bot
FROM (
    SELECT matchid, toUInt8(killerid) AS participantid, {ck} AS bucket,
        toUInt8(1) AS kills, toUInt8(0) AS assists, toUInt8(0) AS deaths,
        toUInt8(0) AS plate_top, toUInt8(0) AS plate_mid, toUInt8(0) AS plate_bot
    FROM {T.CHAMPION_KILL_TABLE} WHERE killerid > 0 AND {rc}
    UNION ALL
    SELECT matchid, toUInt8(victimid), {ck},
        toUInt8(0), toUInt8(0), toUInt8(1),
        toUInt8(0), toUInt8(0), toUInt8(0)
    FROM {T.CHAMPION_KILL_TABLE} WHERE victimid > 0 AND {rc}
    UNION ALL
    SELECT matchid, toUInt8(arrayJoin(assistingparticipantids)), {ck},
        toUInt8(0), toUInt8(1), toUInt8(0),
        toUInt8(0), toUInt8(0), toUInt8(0)
    FROM {T.CHAMPION_KILL_TABLE} WHERE {rc}
    UNION ALL
    SELECT matchid, toUInt8(killerid), {ck},
        toUInt8(0), toUInt8(0), toUInt8(0),
        toUInt8(lanetype = 'TOP_LANE'),
        toUInt8(lanetype = 'MID_LANE'),
        toUInt8(lanetype = 'BOT_LANE')
    FROM {T.TURRET_PLATE_TABLE} WHERE killerid > 0 AND {rc}
) AS e
INNER JOIN participant_context AS pc
    ON e.matchid = pc.matchid AND e.participantid = pc.participantid
GROUP BY split, championid, teamposition, build, bucket
{_HEAVY}
"""


def _temporal_combine(split: str) -> str:
    stat_sums = ",\n        ".join(f"sum(sum_{m}) AS sum_{m}" for m in T.TEMPORAL_METRICS)
    stat_out = ",\n    ".join(f"s.sum_{m}" for m in T.TEMPORAL_METRICS)
    ev_sums = ",\n        ".join(f"sum(ev_{m}) AS ev_{m}" for m in T.EVENT_METRICS)
    ev_out = ",\n    ".join(f"ifNull(e.ev_{m}, 0)" for m in T.EVENT_METRICS)
    lit = sql_literal(split)
    return f"""
INSERT INTO {T.BINS_TABLE}
SELECT
    s.split, s.championid, s.teamposition, s.build, s.bucket,
    s.frames,
    {stat_out},
    {ev_out}
FROM (
    SELECT split, championid, teamposition, build, bucket,
        sum(frames) AS frames,
        {stat_sums}
    FROM {_STAT_STAGE} WHERE split = {lit}
    GROUP BY split, championid, teamposition, build, bucket
) AS s
LEFT JOIN (
    SELECT split, championid, teamposition, build, bucket,
        {ev_sums}
    FROM {_EV_STAGE} WHERE split = {lit}
    GROUP BY split, championid, teamposition, build, bucket
) AS e USING (split, championid, teamposition, build, bucket)
"""


def _cmd(client, sql: str) -> None:
    client.command(sql)


def _recreate(client, table: str, ddl: str) -> None:
    _cmd(client, f"DROP TABLE IF EXISTS {table}")
    _cmd(client, ddl)


def build_classification_tables(*, include_context: bool = True) -> None:
    """(Re)materialise the full-game sufficient-statistic tables for every split."""
    client = get_client()
    for table, ddl in (
        (IDENTITY_BASE, _identity_ddl()),
        (FINAL_BASE, _final_ddl(FINAL_BASE)),
        (_FINAL_STAGE, _final_ddl(_FINAL_STAGE)),
        (META_TABLE, _meta_ddl()),
    ):
        _recreate(client, table, ddl)
    if include_context:
        for table, ddl in (
            (CONTEXT_BASE, _context_base_ddl()),
            (_TEAM_STAGE, _context_stage_ddl(_TEAM_STAGE, TEAM_FEATURE_NAMES)),
            (
                _MATCHUP_STAGE,
                _context_stage_ddl(_MATCHUP_STAGE, MATCHUP_FEATURE_NAMES),
            ),
        ):
            _recreate(client, table, ddl)

    for split in SPLITS:
        logger.info("Building identity/final base for split=%s", split)
        _cmd(client, _identity_insert(split))
        for shard in range(K_SHARDS):
            _cmd(client, _final_insert(split, shard))
        _cmd(client, _final_combine(split))
        _cmd(client, f"TRUNCATE TABLE {_FINAL_STAGE}")
        if include_context:
            logger.info("Building context base for split=%s", split)
            for shard in range(K_SHARDS):
                team_sql, _ = team_share_query(split, K_SHARDS, shard)
                matchup_sql, _ = matchup_query(split, K_SHARDS, shard)
                _cmd(client, _context_stage_insert(_TEAM_STAGE, team_sql, TEAM_FEATURE_NAMES, split))
                _cmd(
                    client,
                    _context_stage_insert(_MATCHUP_STAGE, matchup_sql, MATCHUP_FEATURE_NAMES, split),
                )
            _cmd(client, _context_combine(split))
            _cmd(client, f"TRUNCATE TABLE {_TEAM_STAGE}")
            _cmd(client, f"TRUNCATE TABLE {_MATCHUP_STAGE}")

    _cmd(client, f"TRUNCATE TABLE {META_TABLE}")
    _cmd(client, f"INSERT INTO {META_TABLE} (catalogue_hash) VALUES ({sql_literal(catalogue_hash())})")
    logger.info("Built classification base tables")


def build_temporal_table() -> None:
    """(Re)materialise temporal_identity_bins via staged, sharded scans."""
    client = get_client()
    for table, ddl in (
        (T.BINS_TABLE, _bins_ddl()),
        (_STAT_STAGE, _temporal_stat_ddl(_STAT_STAGE)),
        (_EV_STAGE, _temporal_ev_ddl(_EV_STAGE)),
    ):
        _recreate(client, table, ddl)
    bounds = _matchid_bounds(client, K_SHARDS)
    for split in SPLITS:
        logger.info("Building temporal bins for split=%s", split)
        for shard in range(K_SHARDS):
            _cmd(client, _temporal_stat_insert(split, shard, bounds))
            _cmd(client, _temporal_ev_insert(split, shard, bounds))
        _cmd(client, _temporal_combine(split))
        _cmd(client, f"TRUNCATE TABLE {_STAT_STAGE}")
        _cmd(client, f"TRUNCATE TABLE {_EV_STAGE}")
    logger.info("Built %s", T.BINS_TABLE)


def assert_built(catalogue: str | None = None) -> None:
    """Raise if the base tables are absent or built from a stale catalogue."""
    catalogue = catalogue or catalogue_hash()
    rows = get_client().query(
        f"SELECT catalogue_hash FROM {META_TABLE} ORDER BY built_at DESC LIMIT 1"
    ).result_rows
    if not rows:
        raise RuntimeError(
            f"{IDENTITY_BASE} not built; run build_classification_tables()"
        )
    if str(rows[0][0]) != catalogue:
        raise RuntimeError(
            f"{IDENTITY_BASE} built from a stale catalogue "
            f"({rows[0][0]!r} != {catalogue!r}); rebuild with build_classification_tables()"
        )
