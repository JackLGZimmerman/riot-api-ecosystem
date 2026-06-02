"""Load non-temporal identity rows and derive hierarchical priors.

The embeddable population is one row per:
    (championid, teamposition, build)

Rows are aggregated directly from filtered participant stats plus final
participant timeline snapshots. Prior levels are derived in memory from the
baseline aggregate, so classification no longer depends on persisted temporal
ClickHouse tables.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.classification.embeddings.config import (
    ALL_METRICS,
    FINAL_SNAPSHOT_AVG_METRICS,
    ITEM_VALUE_TOTALS_TABLE,
    LARGEST_AVG_METRICS,
    LEVEL_KEY,
    ML_GAME_SPLIT_TABLE,
    PARTICIPANT_STATS_TABLE,
    PER_MINUTE_METRICS,
    PRIOR_LEVELS,
    RATE_METRICS,
    RATE_LIKE_METRICS,
    TIMELINE_CHECKPOINT_MINUTES,
    TIMELINE_SOURCE_METRICS,
    TIMELINE_STATS_TABLE,
    EmbeddingConfig,
    IdentityType,
)
from app.core.utils.common import sql_literal
from app.core.utils.smoothing import (
    SIBLING_BUILD_BY_LABEL,
    build_group_sql,
)
from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevelRows:
    """All rows for one target/prior level.

    Each column is a 1-D numpy array of length `n`, co-indexed by row.
    `key_columns` identify the lookup key.
    """

    level: IdentityType
    key_columns: tuple[str, ...]
    columns: dict[str, np.ndarray]
    n: int

    def with_columns(self, extra: dict[str, np.ndarray]) -> "LevelRows":
        merged = dict(self.columns)
        merged.update(extra)
        return LevelRows(self.level, self.key_columns, merged, self.n)


def _columns_to_arrays(
    rows: Sequence[Sequence[object]], col_names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    n = len(rows)
    arrays: dict[str, np.ndarray] = {}
    for i, name in enumerate(col_names):
        col = [r[i] for r in rows]
        if name == "championid":
            arrays[name] = np.asarray(col, dtype=np.int32)
        elif name == "sum_w_timeplayed":
            arrays[name] = np.asarray(col, dtype=np.float64)
        elif name == "matchups" or name in ALL_METRICS:
            arrays[name] = np.asarray(col, dtype=np.float32)
        else:
            arrays[name] = np.asarray(col, dtype=object)
    assert all(arr.shape == (n,) for arr in arrays.values())
    return arrays


def _query_to_level(
    *,
    level: IdentityType,
    query: str,
    col_names: tuple[str, ...],
) -> LevelRows:
    rows = get_client().query(query).result_rows
    arrays = _columns_to_arrays(rows, col_names)
    n = len(rows)
    logger.info("Loaded %s: %d rows", level.value, n)
    return LevelRows(
        level=level,
        key_columns=LEVEL_KEY[level],
        columns=arrays,
        n=n,
    )


def _raw_cache_path(cfg: EmbeddingConfig, label: str) -> Path:
    return cfg.cache_dir / "_raw" / f"{cfg.split}_{label}.npz"


def _save_npz_atomic(path: Path, columns: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "column_names": np.asarray(tuple(columns), dtype=object),
        **{f"col_{i}": value for i, value in enumerate(columns.values())},
    }
    with tmp.open("wb") as fh:
        np.savez(fh, **payload)
    tmp.replace(path)


def _load_npz_columns(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=True)
    names = tuple(str(name) for name in payload["column_names"].tolist())
    return {name: payload[f"col_{i}"] for i, name in enumerate(names)}


def _save_level_rows(path: Path, rows: LevelRows) -> None:
    _save_npz_atomic(
        path,
        {
            "__n": np.asarray([rows.n], dtype=np.int64),
            "__level": np.asarray([rows.level.value], dtype=object),
            "__key_columns": np.asarray(rows.key_columns, dtype=object),
            **rows.columns,
        },
    )


def _load_level_rows(path: Path, level: IdentityType) -> LevelRows | None:
    columns = _load_npz_columns(path)
    if columns is None:
        return None
    n = int(columns.pop("__n")[0])
    cached_level = str(columns.pop("__level")[0])
    key_columns = tuple(str(value) for value in columns.pop("__key_columns").tolist())
    if cached_level != level.value or key_columns != LEVEL_KEY[level]:
        logger.warning("Ignoring stale classification cache %s", path)
        return None
    if any(arr.shape != (n,) for arr in columns.values()):
        logger.warning("Ignoring malformed classification cache %s", path)
        return None
    if any("challenge" in name.lower() for name in columns):
        logger.warning("Ignoring challenge-derived classification cache %s", path)
        return None
    logger.info("Loaded cached %s: %d rows", level.value, n)
    return LevelRows(
        level=level,
        key_columns=LEVEL_KEY[level],
        columns=columns,
        n=n,
    )


def _rate_aggs(metrics: tuple[str, ...]) -> list[str]:
    return [f"toFloat32(sum(ps.{metric}) / count()) AS {metric}" for metric in metrics]


def _per_minute_aggs(metrics: tuple[str, ...]) -> list[str]:
    nullable_zero = {"damagedealttoepicmonsters"}
    out: list[str] = []
    for metric in metrics:
        value = f"coalesce(ps.{metric}, 0)" if metric in nullable_zero else f"ps.{metric}"
        out.append(
            "toFloat32("
            f"if(sum(ps.timeplayed) > 0, 60 * sum({value}) / sum(ps.timeplayed), 0)"
            f") AS {metric}"
        )
    return out


def _participant_context_cte(
    split: str,
    *,
    include_stats: bool = True,
    stat_columns: tuple[str, ...] | None = None,
) -> str:
    stat_select = ""
    if include_stats:
        columns = stat_columns or (
            "timeplayed",
            *RATE_METRICS,
            *LARGEST_AVG_METRICS,
            *PER_MINUTE_METRICS,
        )
        stat_select = (
            ",\n        "
            + ",\n        ".join(f"ps.{column} AS {column}" for column in columns)
        )
    return f"""
participant_context AS (
    SELECT
        ps.matchid AS matchid,
        ps.participantid AS participantid,
        assumeNotNull(ps.championid) AS championid_nn,
        toString(ps.teamposition) AS teamposition_str,
        toString(ivt.highest_value_label) AS build
        {stat_select}
    FROM {PARTICIPANT_STATS_TABLE} AS ps
    INNER JOIN {ML_GAME_SPLIT_TABLE} AS s
        ON ps.matchid = s.matchid
    INNER JOIN {ITEM_VALUE_TOTALS_TABLE} AS ivt
        ON
            ps.matchid = ivt.matchid
            AND ps.participantid = ivt.participantid
    WHERE
        s.split = {split}
        AND isNotNull(ps.championid)
        AND toString(ps.teamposition) != 'UNKNOWN'
)
"""


def _baseline_query(split: str) -> str:
    split_sql = sql_literal(split)
    placeholder_by_metric = {
        metric: f"toFloat32(0) AS {metric}"
        for metric in ALL_METRICS
    }
    metric_aggs = [placeholder_by_metric[metric] for metric in ALL_METRICS]
    return f"""
WITH
{_participant_context_cte(split_sql, stat_columns=("timeplayed",))}
SELECT
    ps.championid_nn AS championid,
    ps.teamposition_str AS teamposition,
    ps.build AS build,
    {build_group_sql("ps.build")},
    toUInt32(count()) AS matchups,
    toFloat64(sum(ps.timeplayed)) AS sum_w_timeplayed,
    {", ".join(metric_aggs)}
FROM participant_context AS ps
GROUP BY
    championid,
    teamposition,
    build,
    build_group
SETTINGS
    max_threads = 1,
    max_bytes_before_external_group_by = 500000000,
    max_bytes_before_external_sort = 500000000,
    join_algorithm = 'grace_hash'
"""


def _direct_metric_query(
    split: str,
    metrics: tuple[str, ...],
    *,
    per_minute: bool,
) -> str:
    split_sql = sql_literal(split)
    stat_columns = metrics
    if per_minute:
        stat_columns = ("timeplayed", *metrics)
        metric_aggs = _per_minute_aggs(metrics)
    else:
        metric_aggs = _rate_aggs(metrics)
    return f"""
WITH
{_participant_context_cte(split_sql, stat_columns=tuple(dict.fromkeys(stat_columns)))}
SELECT
    ps.championid_nn AS championid,
    ps.teamposition_str AS teamposition,
    ps.build AS build,
    {", ".join(metric_aggs)}
FROM participant_context AS ps
GROUP BY
    championid,
    teamposition,
    build
SETTINGS
    max_threads = 1,
    max_bytes_before_external_group_by = 500000000,
    max_bytes_before_external_sort = 500000000,
    join_algorithm = 'grace_hash'
"""


def _timeline_final_query(split: str) -> str:
    split_sql = sql_literal(split)
    tuple_select = ",\n                ".join(FINAL_SNAPSHOT_AVG_METRICS)
    metric_aggs = ",\n    ".join(
        f"toFloat32(coalesce(avg(tupleElement(ts.final_stats, {i})), 0)) AS {metric}"
        for i, metric in enumerate(FINAL_SNAPSHOT_AVG_METRICS, start=1)
    )
    return f"""
WITH
timeline_final AS (
    SELECT
        matchid,
        participantid,
        argMax(
            tuple(
                {tuple_select}
            ),
            frame_timestamp
        ) AS final_stats
    FROM {TIMELINE_STATS_TABLE}
    WHERE matchid IN (
        SELECT matchid
        FROM {ML_GAME_SPLIT_TABLE}
        WHERE split = {split_sql}
    )
    GROUP BY
        matchid,
        participantid
),
{_participant_context_cte(split_sql, include_stats=False)}
SELECT
    ps.championid_nn AS championid,
    ps.teamposition_str AS teamposition,
    ps.build AS build,
    {metric_aggs}
FROM participant_context AS ps
LEFT JOIN timeline_final AS ts
    ON
        ps.matchid = ts.matchid
        AND ps.participantid = ts.participantid
GROUP BY
    championid,
    teamposition,
    build
SETTINGS
    max_threads = 2,
    max_bytes_before_external_group_by = 2000000000,
    max_bytes_before_external_sort = 2000000000
"""


def _timeline_checkpoint_query(split: str, minute: int) -> str:
    split_sql = sql_literal(split)
    checkpoint_tuple = ",\n                ".join(
        expr for _, expr in TIMELINE_SOURCE_METRICS
    )
    has_expr = "coalesce(ts.matchid, '') != ''"
    metric_aggs = []
    for i, (metric, _) in enumerate(TIMELINE_SOURCE_METRICS, start=1):
        name = f"tl_{minute}_{metric}"
        metric_aggs.append(
            "toFloat32("
            f"coalesce(avgIf(tupleElement(ts.stats, {i}), {has_expr}), 0)"
            f") AS {name}"
        )
    metric_aggs.append(
        "toFloat32("
        f"1.0 - (countIf({has_expr}) / count())"
        f") AS tl_{minute}_missing"
    )
    threshold_ms = minute * 60_000
    return f"""
WITH
timeline_checkpoint AS (
    SELECT
        matchid,
        participantid,
        argMax(
            tuple(
                {checkpoint_tuple}
            ),
            frame_timestamp
        ) AS stats
    FROM {TIMELINE_STATS_TABLE}
    WHERE
        frame_timestamp <= {threshold_ms}
        AND matchid IN (
            SELECT matchid
            FROM {ML_GAME_SPLIT_TABLE}
            WHERE split = {split_sql}
        )
    GROUP BY
        matchid,
        participantid
),
{_participant_context_cte(split_sql, include_stats=False)}
SELECT
    ps.championid_nn AS championid,
    ps.teamposition_str AS teamposition,
    ps.build AS build,
    {", ".join(metric_aggs)}
FROM participant_context AS ps
LEFT JOIN timeline_checkpoint AS ts
    ON
        ps.matchid = ts.matchid
        AND ps.participantid = ts.participantid
GROUP BY
    championid,
    teamposition,
    build
SETTINGS
    max_threads = 2,
    max_bytes_before_external_group_by = 2000000000,
    max_bytes_before_external_sort = 2000000000
"""


def _row_key(row: Sequence[object]) -> tuple[int, str, str]:
    return (int(row[0]), str(row[1]), str(row[2]))


def _baseline_index(rows: LevelRows) -> dict[tuple[int, str, str], int]:
    return {
        (
            int(rows.columns["championid"][idx]),
            str(rows.columns["teamposition"][idx]),
            str(rows.columns["build"][idx]),
        ): idx
        for idx in range(rows.n)
    }


def _merge_identity_metrics(
    rows: LevelRows,
    query: str,
    metrics: tuple[str, ...],
    *,
    cache_path: Path | None = None,
) -> LevelRows:
    if cache_path is not None:
        cached = _load_npz_columns(cache_path)
        if cached is not None and set(cached) == set(metrics):
            if all(cached[metric].shape == (rows.n,) for metric in metrics):
                logger.info("Loaded cached timeline metrics: %s", cache_path.name)
                return rows.with_columns({metric: cached[metric] for metric in metrics})
            logger.warning("Ignoring malformed timeline cache %s", cache_path)

    fetched = get_client().query(query).result_rows
    index = _baseline_index(rows)
    extra = {
        metric: np.zeros(rows.n, dtype=np.float32)
        for metric in metrics
    }
    for row in fetched:
        idx = index.get(_row_key(row))
        if idx is None:
            continue
        for offset, metric in enumerate(metrics, start=3):
            extra[metric][idx] = np.float32(row[offset] or 0.0)
    logger.info("Merged %d timeline rows into %s", len(fetched), rows.level.value)
    if cache_path is not None:
        _save_npz_atomic(cache_path, extra)
    return rows.with_columns(extra)


def _chunks(values: tuple[str, ...], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(values[i : i + size] for i in range(0, len(values), size))


def load_baseline(cfg: EmbeddingConfig) -> LevelRows:
    key_cols = LEVEL_KEY[IdentityType.BASELINE]
    col_names = (
        *key_cols,
        "build_group",
        "matchups",
        "sum_w_timeplayed",
        *ALL_METRICS,
    )
    baseline_cache = _raw_cache_path(cfg, "baseline_direct")
    rows = _load_level_rows(baseline_cache, IdentityType.BASELINE)
    if rows is None:
        rows = _query_to_level(
            level=IdentityType.BASELINE,
            query=_baseline_query(cfg.split),
            col_names=col_names,
        )
        _save_level_rows(baseline_cache, rows)
    direct_rate_metrics = (*RATE_METRICS, *LARGEST_AVG_METRICS)
    rows = _merge_identity_metrics(
        rows,
        _direct_metric_query(cfg.split, direct_rate_metrics, per_minute=False),
        direct_rate_metrics,
        cache_path=_raw_cache_path(cfg, "direct_rates"),
    )
    for idx, metrics in enumerate(_chunks(PER_MINUTE_METRICS, 8)):
        rows = _merge_identity_metrics(
            rows,
            _direct_metric_query(cfg.split, metrics, per_minute=True),
            metrics,
            cache_path=_raw_cache_path(cfg, f"direct_per_minute_{idx:02d}"),
        )
    rows = _merge_identity_metrics(
        rows,
        _timeline_final_query(cfg.split),
        FINAL_SNAPSHOT_AVG_METRICS,
        cache_path=_raw_cache_path(cfg, "timeline_final"),
    )
    for minute in TIMELINE_CHECKPOINT_MINUTES:
        metrics = (
            *(f"tl_{minute}_{metric}" for metric, _ in TIMELINE_SOURCE_METRICS),
            f"tl_{minute}_missing",
        )
        rows = _merge_identity_metrics(
            rows,
            _timeline_checkpoint_query(cfg.split, minute),
            metrics,
            cache_path=_raw_cache_path(cfg, f"timeline_{minute}"),
        )
    return rows


def _prior_key(level: IdentityType, rows: LevelRows, idx: int) -> tuple | None:
    c = rows.columns
    championid = int(c["championid"][idx])
    teamposition = str(c["teamposition"][idx])
    build = str(c["build"][idx])
    build_group = str(c["build_group"][idx])

    if level is IdentityType.SIBLING:
        sibling = SIBLING_BUILD_BY_LABEL.get(build)
        return (championid, teamposition, sibling) if sibling else None
    if level is IdentityType.CHAMPION_ROLE:
        return (championid, teamposition)
    if level is IdentityType.ROLE_BUILD:
        return (teamposition, build_group)
    if level is IdentityType.CHAMPION_BUILD:
        return (championid, build_group)
    if level is IdentityType.BUILD:
        return (build_group,)
    raise ValueError(f"{level.value} is not a prior level")


def _key_array(name: str, values: list[object]) -> np.ndarray:
    if name == "championid":
        return np.asarray(values, dtype=np.int32)
    return np.asarray(values, dtype=object)


def _weighted_average(
    values: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
) -> np.float32:
    w = weights[indices].astype(np.float64, copy=False)
    denom = float(np.sum(w))
    if denom <= 0.0:
        return np.float32(0.0)
    v = values[indices].astype(np.float64, copy=False)
    return np.float32(np.sum(v * w) / denom)


def derive_prior(level: IdentityType, baseline: LevelRows) -> LevelRows:
    if level not in PRIOR_LEVELS:
        raise ValueError(f"{level.value} is not a prior level")

    grouped: dict[tuple, list[int]] = defaultdict(list)
    for idx in range(baseline.n):
        key = _prior_key(level, baseline, idx)
        if key is not None:
            grouped[key].append(idx)

    key_cols = LEVEL_KEY[level]
    keys = sorted(grouped)
    columns: dict[str, np.ndarray] = {
        name: _key_array(name, [key[i] for key in keys])
        for i, name in enumerate(key_cols)
    }

    matchups = baseline.columns["matchups"].astype(np.float64)
    timeplayed = baseline.columns["sum_w_timeplayed"].astype(np.float64)
    columns["matchups"] = np.asarray(
        [np.sum(matchups[grouped[key]]) for key in keys],
        dtype=np.float32,
    )

    for metric in RATE_LIKE_METRICS:
        values = baseline.columns[metric]
        columns[metric] = np.asarray(
            [
                _weighted_average(values, matchups, np.asarray(grouped[key], dtype=np.int64))
                for key in keys
            ],
            dtype=np.float32,
        )
    for metric in PER_MINUTE_METRICS:
        values = baseline.columns[metric]
        columns[metric] = np.asarray(
            [
                _weighted_average(values, timeplayed, np.asarray(grouped[key], dtype=np.int64))
                for key in keys
            ],
            dtype=np.float32,
        )

    rows = LevelRows(level, key_cols, columns, len(keys))
    logger.info("Derived %s prior: %d rows", level.value, rows.n)
    return rows


def load_all(cfg: EmbeddingConfig) -> dict[IdentityType, LevelRows]:
    baseline = load_baseline(cfg)
    out = {IdentityType.BASELINE: baseline}
    for level in PRIOR_LEVELS:
        out[level] = derive_prior(level, baseline)
    return out
