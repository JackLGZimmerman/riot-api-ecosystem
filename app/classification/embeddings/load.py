"""Load 6010 baseline rows plus 9000-9040 prior tables.

The embeddable population is the 6010 table:
    (championid, teamposition, build, phase)

The 9000-9040 tables are lookup priors used to smooth those baseline rows
before matrix construction. They are not embedded as standalone populations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.classification.embeddings.config import (
    ALL_METRICS,
    LEVEL_KEY,
    PRIOR_LEVELS,
    PRIOR_TABLE,
    RATE_LIKE_METRICS,
    SIBLING_BUILD_BY_LABEL,
    SOURCE_TABLE,
    EmbeddingConfig,
    IdentityType,
    PER_MINUTE_METRICS,
    build_group_sql,
    sibling_build_sql,
)
from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)


def _metric_select(metrics: tuple[str, ...]) -> str:
    return ", ".join(metrics)


@dataclass(frozen=True)
class LevelRows:
    """All rows for one target/prior level.

    Each column is a 1-D numpy array of length `n`, co-indexed by row.
    `key_columns` identify the lookup key excluding `phase`.
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


def _is_unknown_table_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "UNKNOWN_TABLE" in message or "Unknown table expression identifier" in message
    )


def load_baseline(cfg: EmbeddingConfig) -> LevelRows:
    key_cols = LEVEL_KEY[IdentityType.BASELINE]
    col_names = (
        *key_cols,
        "build_group",
        "phase",
        "matchups",
        "sum_w_timeplayed",
        *ALL_METRICS,
    )
    query = (
        "SELECT "
        "championid, teamposition, build, "
        f"{build_group_sql()}, "
        "phase, matchups, sum_w_timeplayed, "
        f"{_metric_select(ALL_METRICS)} "
        f"FROM {SOURCE_TABLE} "
        f"WHERE split = '{cfg.split}'"
    )
    return _query_to_level(
        level=IdentityType.BASELINE,
        query=query,
        col_names=col_names,
    )


def _prior_source_key_clauses(level: IdentityType) -> tuple[str, str, str]:
    if level is IdentityType.SIBLING:
        builds_sql = ",".join(f"'{build}'" for build in SIBLING_BUILD_BY_LABEL)
        return (
            f"championid, teamposition, {sibling_build_sql()} AS build, phase",
            "championid, teamposition, build, phase",
            f" AND build IN ({builds_sql})",
        )
    if level is IdentityType.CHAMPION_ROLE:
        return (
            "championid, teamposition, phase",
            "championid, teamposition, phase",
            "",
        )
    if level is IdentityType.ROLE_BUILD:
        return (
            f"teamposition, {build_group_sql(alias=None)} AS build_group, phase",
            "teamposition, build_group, phase",
            "",
        )
    if level is IdentityType.CHAMPION_BUILD:
        return (
            f"championid, {build_group_sql(alias=None)} AS build_group, phase",
            "championid, build_group, phase",
            "",
        )
    if level is IdentityType.BUILD:
        return (f"{build_group_sql()}, phase", "build_group, phase", "")
    raise ValueError(f"{level.value} is not a prior level")


def _build_prior_query_from_source(level: IdentityType, split: str) -> str:
    select_keys, group_by, where_extra = _prior_source_key_clauses(level)
    rate_aggs = ",".join(
        f"toFloat32(sum({metric} * _m) / sum(_m)) AS {metric}"
        for metric in RATE_LIKE_METRICS
    )
    per_minute_aggs = ",".join(
        f"toFloat32(sum({metric} * _sw) / sum(_sw)) AS {metric}"
        for metric in PER_MINUTE_METRICS
    )
    inner = (
        f"SELECT {select_keys}, "
        "matchups AS _m, sum_w_timeplayed AS _sw, "
        f"{_metric_select(ALL_METRICS)} "
        f"FROM {SOURCE_TABLE} "
        f"WHERE split = '{split}'{where_extra}"
    )
    return (
        f"SELECT {group_by}, "
        "toFloat32(sum(_m)) AS matchups, "
        f"{rate_aggs}, {per_minute_aggs} "
        f"FROM ({inner}) "
        f"GROUP BY {group_by}"
    )


def load_prior(level: IdentityType, cfg: EmbeddingConfig) -> LevelRows:
    if level not in PRIOR_TABLE:
        raise ValueError(f"{level.value} is not a prior level")
    key_cols = LEVEL_KEY[level]
    col_names = (*key_cols, "phase", "matchups", *ALL_METRICS)
    query = (
        f"SELECT {', '.join((*key_cols, 'phase', 'matchups'))}, "
        f"{_metric_select(ALL_METRICS)} "
        f"FROM {PRIOR_TABLE[level]} "
        f"WHERE split = '{cfg.split}'"
    )
    try:
        return _query_to_level(level=level, query=query, col_names=col_names)
    except Exception as exc:
        if not _is_unknown_table_error(exc):
            raise
        logger.warning(
            "Prior table for %s is missing; deriving prior from %s",
            level.value,
            SOURCE_TABLE,
        )
        fallback_query = _build_prior_query_from_source(level, cfg.split)
        return _query_to_level(level=level, query=fallback_query, col_names=col_names)


def load_all(cfg: EmbeddingConfig) -> dict[IdentityType, LevelRows]:
    out = {IdentityType.BASELINE: load_baseline(cfg)}
    for level in PRIOR_LEVELS:
        out[level] = load_prior(level, cfg)
    return out
