"""Load identity rows and prior levels as SQL rollups of sufficient statistics.

The embeddable population is one row per ``(championid, teamposition, build)``.
The heavy aggregation is materialised once into the ClickHouse base tables (see
[build_tables.py]); every prior level is then an exact ``GROUP BY`` rollup of
those sums. Python only issues the SELECTs and hands the rows to the shared
smoother. No metric value is computed in Python.

Each metric resolves to its stored sufficient statistic:

* rate / largest-avg -> ``sum_<m> / matchups``
* per-minute         -> ``60 * sum_<m> / sum_w_timeplayed``
* final-snapshot     -> ``sum_final_<m> / matchups`` (missing snapshot = 0)
* context feature    -> ``sum_<f> / cnt`` (team uses ``cnt_team``, matchup
  ``cnt_matchup``)

A prior level replaces every column with ``sum(...)`` over the coarser key, which
is the pooled value at that grain (equal to the former matchups-weighted average
for the count-denominated families).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.classification.embeddings.build_tables import (
    CONTEXT_BASE,
    FINAL_BASE,
    IDENTITY_BASE,
    assert_built,
)
from app.classification.embeddings.config import (
    ALL_METRICS,
    LEVEL_KEY,
    PRIOR_LEVELS,
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.context_features import (
    CONTEXT_FEATURE_NAMES,
    TEAM_FEATURE_NAMES,
)
from app.classification.embeddings.registry import RAW_SPECS, Source
from app.core.utils.common import sql_literal
from app.core.utils.smoothing import sibling_build_sql
from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)

_SOURCE_BY_METRIC: dict[str, Source] = {spec.name: spec.source for spec in RAW_SPECS}
_TEAM_SET = frozenset(TEAM_FEATURE_NAMES)
_NUMERIC_FLOAT32 = frozenset((*ALL_METRICS, *CONTEXT_FEATURE_NAMES))


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
        elif name == "matchups" or name in _NUMERIC_FLOAT32:
            arrays[name] = np.asarray(col, dtype=np.float32)
        else:
            arrays[name] = np.asarray(col, dtype=object)
    assert all(arr.shape == (n,) for arr in arrays.values())
    return arrays


def _query_to_level(
    *, level: IdentityType, query: str, col_names: tuple[str, ...]
) -> LevelRows:
    rows = get_client().query(query).result_rows
    arrays = _columns_to_arrays(rows, col_names)
    logger.info("Loaded %s: %d rows", level.value, len(rows))
    return LevelRows(level, LEVEL_KEY[level], arrays, len(rows))


def _metric_expr(metric: str, *, rollup: bool) -> str:
    source = _SOURCE_BY_METRIC[metric]
    s = (lambda e: f"sum({e})") if rollup else (lambda e: e)
    if source in (Source.RATE, Source.LARGEST_AVG):
        body = f"{s(f'b.sum_{metric}')} / {s('b.matchups')}"
    elif source is Source.FINAL_SNAPSHOT:
        body = f"{s(f'ifNull(f.sum_final_{metric}, 0)')} / {s('b.matchups')}"
    elif source is Source.PER_MINUTE:
        tp = s("b.sum_w_timeplayed")
        body = f"if({tp} > 0, 60 * {s(f'b.sum_{metric}')} / {tp}, 0)"
    else:  # pragma: no cover - raw specs are only the four sources above
        raise ValueError(f"{metric}: unsupported source {source}")
    return f"toFloat32({body}) AS {metric}"


def _context_expr(feature: str, *, rollup: bool) -> str:
    cnt = "c.cnt_team" if feature in _TEAM_SET else "c.cnt_matchup"
    s = (lambda e: f"sum({e})") if rollup else (lambda e: e)
    num = s(f"ifNull(c.sum_{feature}, 0)")
    den = f"greatest({s(f'ifNull({cnt}, 0)')}, 1)"
    return f"toFloat32({num} / {den}) AS {feature}"


def _join_sql(split: str, *, context: bool) -> str:
    keys = " AND ".join(f"b.{k} = f.{k}" for k in LEVEL_KEY[IdentityType.BASELINE])
    join = (
        f"FROM {IDENTITY_BASE} AS b\n"
        f"LEFT JOIN {FINAL_BASE} AS f ON b.split = f.split AND {keys}\n"
    )
    if context:
        ckeys = " AND ".join(
            f"b.{k} = c.{k}" for k in LEVEL_KEY[IdentityType.BASELINE]
        )
        join += f"LEFT JOIN {CONTEXT_BASE} AS c ON b.split = c.split AND {ckeys}\n"
    return join + f"WHERE b.split = {sql_literal(split)}"


def _value_exprs(*, rollup: bool, context: bool) -> list[str]:
    exprs = [_metric_expr(m, rollup=rollup) for m in ALL_METRICS]
    if context:
        exprs += [_context_expr(f, rollup=rollup) for f in CONTEXT_FEATURE_NAMES]
    return exprs


def _col_names(extra_keys: tuple[str, ...], *, context: bool) -> tuple[str, ...]:
    ctx = CONTEXT_FEATURE_NAMES if context else ()
    return (*extra_keys, "matchups", *ALL_METRICS, *ctx)


def _baseline_query(cfg: EmbeddingConfig) -> tuple[str, tuple[str, ...]]:
    context = cfg.include_context_features
    select = ",\n    ".join(
        [
            "b.championid AS championid",
            "b.teamposition AS teamposition",
            "b.build AS build",
            "b.build_group AS build_group",
            "toUInt32(b.matchups) AS matchups",
            "toFloat64(b.sum_w_timeplayed) AS sum_w_timeplayed",
            *_value_exprs(rollup=False, context=context),
        ]
    )
    query = f"SELECT\n    {select}\n{_join_sql(cfg.split, context=context)}"
    col_names = (
        "championid",
        "teamposition",
        "build",
        "build_group",
        "matchups",
        "sum_w_timeplayed",
        *ALL_METRICS,
        *(CONTEXT_FEATURE_NAMES if context else ()),
    )
    return query, col_names


def _level_key_exprs(level: IdentityType) -> tuple[list[str], str]:
    """Return (key select-expressions, extra WHERE) for a prior level."""
    if level is IdentityType.SIBLING:
        sib = sibling_build_sql("b.build")
        return (
            ["b.championid AS championid", "b.teamposition AS teamposition", f"{sib} AS build"],
            f" AND {sib} != ''",
        )
    if level is IdentityType.CHAMPION_ROLE:
        return (["b.championid AS championid", "b.teamposition AS teamposition"], "")
    if level is IdentityType.ROLE_BUILD:
        return (["b.teamposition AS teamposition", "b.build_group AS build_group"], "")
    if level is IdentityType.CHAMPION_BUILD:
        return (["b.championid AS championid", "b.build_group AS build_group"], "")
    if level is IdentityType.BUILD:
        return (["b.build_group AS build_group"], "")
    raise ValueError(f"{level.value} is not a prior level")


def _prior_query(level: IdentityType, cfg: EmbeddingConfig) -> tuple[str, tuple[str, ...]]:
    context = cfg.include_context_features
    key_exprs, where_extra = _level_key_exprs(level)
    key_cols = LEVEL_KEY[level]
    select = ",\n    ".join(
        [*key_exprs, "toUInt32(sum(b.matchups)) AS matchups", *_value_exprs(rollup=True, context=context)]
    )
    group_by = ", ".join(key_cols)
    query = (
        f"SELECT\n    {select}\n{_join_sql(cfg.split, context=context)}{where_extra}\n"
        f"GROUP BY {group_by}"
    )
    return query, _col_names(key_cols, context=context)


def load_baseline(cfg: EmbeddingConfig) -> LevelRows:
    assert_built()
    query, col_names = _baseline_query(cfg)
    return _query_to_level(level=IdentityType.BASELINE, query=query, col_names=col_names)


def derive_prior(level: IdentityType, cfg: EmbeddingConfig) -> LevelRows:
    if level not in PRIOR_LEVELS:
        raise ValueError(f"{level.value} is not a prior level")
    query, col_names = _prior_query(level, cfg)
    return _query_to_level(level=level, query=query, col_names=col_names)


def load_all(cfg: EmbeddingConfig) -> dict[IdentityType, LevelRows]:
    assert_built()
    out: dict[IdentityType, LevelRows] = {IdentityType.BASELINE: load_baseline(cfg)}
    for level in PRIOR_LEVELS:
        out[level] = derive_prior(level, cfg)
    return out
