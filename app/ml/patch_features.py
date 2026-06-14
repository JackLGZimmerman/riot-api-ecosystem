"""Production patch-only temporal feature extraction for the HGNN win model.

The single patch head captures per-(season, patch) blue-side base-rate drift.
The aggregate is train-only; for train rows the candidate game's own outcome is
removed from the matching (season, patch) cell before the empirical-Bayes logit
delta is computed. No participant, player, rune, or summoner-spell information
is read - the feature is a draft-time-known, player-agnostic side prior, so it
serves the RL drafting environment as a fixed per-episode input.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
import re
import time
from typing import Any

import numpy as np

from app.ml.config import PLAYER_PIVOT_TABLE, DatasetConfig
from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)

PATCH_FEATURE_NAMES: tuple[str, ...] = (
    "patch_blue_rate_logit_delta",
    "patch_blue_rate_coverage",
)
PATCH_FEATURE_DIM = len(PATCH_FEATURE_NAMES)
PATCH_SIGNED_FEATURE_INDICES: tuple[int, ...] = (0,)

EPS = 1.0e-6
SUPPORT_STRENGTH = 30.0
MIN_PATCH_N = 50
TMP_TABLE_PREFIX = "game_data_filtered.hgnn_prod_patch"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

SPLIT_DB_NAMES = {
    "train": "train",
    "test": "test",
}


def _sql_str(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _sql_ident(value: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"unsafe ClickHouse identifier: {value!r}")
    return value


def _validate_source_contract(cfg: DatasetConfig) -> None:
    if cfg.player_pivot_table != PLAYER_PIVOT_TABLE:
        raise ValueError(
            "Patch feature extraction is tied to the production population "
            f"{PLAYER_PIVOT_TABLE!r}; got {cfg.player_pivot_table!r}."
        )


def _ch_logit(expr: str) -> str:
    clipped = f"least(greatest(toFloat64({expr}), {EPS}), {1.0 - EPS})"
    return f"log({clipped} / (1.0 - {clipped}))"


def _logit(value: float) -> float:
    clipped = min(max(float(value), EPS), 1.0 - EPS)
    return math.log(clipped / (1.0 - clipped))


def _serving_feature_vector(matchups: int, blue_win_rate: float) -> np.ndarray:
    n = float(matchups)
    if n >= MIN_PATCH_N:
        delta = (_logit(blue_win_rate) - _logit(0.5)) * n / (n + SUPPORT_STRENGTH)
        coverage = 1.0
    else:
        delta = 0.0
        coverage = 0.0
    return np.asarray((delta, coverage), dtype=np.float32)


def _tmp_suffix() -> str:
    return f"{os.getpid()}_{int(time.time() * 1000)}"


def _aggregate_table_names(suffix: str) -> dict[str, str]:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in suffix)
    return {"patch_blue": f"{TMP_TABLE_PREFIX}_patch_blue_tmp_{safe}"}


def _drop_aggregate_tables(client: Any, tables: dict[str, str]) -> None:
    for table in tables.values():
        client.command(f"DROP TABLE IF EXISTS {table}")


def _prepare_aggregate_tables(client: Any, suffix: str) -> dict[str, str]:
    tables = _aggregate_table_names(suffix)
    _drop_aggregate_tables(client, tables)
    settings = {"max_query_size": 200_000_000}
    client.command(
        f"""
CREATE TABLE {tables["patch_blue"]}
ENGINE = Memory
AS
SELECT
  gi.season AS season,
  gi.patch AS patch,
  count() AS matchups,
  avg(p.blue_win) AS blue_win_rate
FROM game_data_filtered.ml_game_player_pivot AS p
INNER JOIN game_data.info AS gi
  ON p.matchid = gi.matchid
WHERE p.split = 'train'
GROUP BY season, patch
""",
        settings=settings,
    )
    return tables


def _selected_sql(cfg: DatasetConfig, split: str, limit: int) -> str:
    return f"""
SELECT matchid, blue_win
FROM {_sql_ident(cfg.player_pivot_table)}
WHERE split = {_sql_str(split)}
ORDER BY matchid
LIMIT {int(limit)}
"""


def _feature_query(
    cfg: DatasetConfig,
    split: str,
    limit: int,
    tables: dict[str, str],
) -> str:
    selected = _selected_sql(cfg, split, limit)
    is_train = "1" if split == "train" else "0"
    patch_blue_n_adj = f"greatest(toFloat64(patch_blue_n_raw) - {is_train}, 0.0)"
    patch_blue_wr_adj = (
        f"if({patch_blue_n_adj} > 0.0, "
        f"(toFloat64(patch_blue_wr_raw) * toFloat64(patch_blue_n_raw) "
        f"- ({is_train} * blue_win)) "
        f"/ greatest(toFloat64(patch_blue_n_raw) - {is_train}, 1.0), 0.5)"
    )
    patch_blue_delta = (
        f"if(patch_blue_n_adj >= {MIN_PATCH_N}, "
        f"({_ch_logit('patch_blue_wr_adj')} - {_ch_logit('0.5')}) "
        f"* patch_blue_n_adj / (patch_blue_n_adj + {SUPPORT_STRENGTH}), 0.0)"
    )
    return f"""
SELECT
  matchid,
  blue_win,
  {patch_blue_delta} AS {PATCH_FEATURE_NAMES[0]},
  toFloat64(patch_blue_n_adj >= {MIN_PATCH_N}) AS {PATCH_FEATURE_NAMES[1]}
FROM (
  SELECT
    matchid,
    blue_win,
    {patch_blue_n_adj} AS patch_blue_n_adj,
    {patch_blue_wr_adj} AS patch_blue_wr_adj
  FROM (
    SELECT
      s.matchid AS matchid,
      s.blue_win AS blue_win,
      ifNull(patch_blue.matchups, 0) AS patch_blue_n_raw,
      ifNull(patch_blue.blue_win_rate, 0.5) AS patch_blue_wr_raw
    FROM ({selected}) AS s
    INNER JOIN game_data.info AS gi
      ON s.matchid = gi.matchid
    LEFT JOIN {tables["patch_blue"]} AS patch_blue
      ON patch_blue.season = gi.season
     AND patch_blue.patch = gi.patch
  )
)
ORDER BY matchid
"""


def _load_features(
    client: Any,
    cfg: DatasetConfig,
    split: str,
    limit: int,
    tables: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0:
        return (
            np.zeros((0, PATCH_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
        )
    query = _feature_query(cfg, split, limit, tables)
    result = client.query(query, settings={"max_query_size": 200_000_000})
    if not result.result_rows:
        return (
            np.zeros((0, PATCH_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
        )
    columns = {name: idx for idx, name in enumerate(result.column_names)}
    patch = np.asarray(
        [
            [float(row[columns[name]]) for name in PATCH_FEATURE_NAMES]
            for row in result.result_rows
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [float(row[columns["blue_win"]]) for row in result.result_rows],
        dtype=np.float64,
    )
    return patch, labels


def _load_serving_patch_aggregate(
    client: Any,
    cfg: DatasetConfig,
    *,
    season: int,
    patch: int,
) -> tuple[int, float]:
    query = f"""
SELECT
  count() AS matchups,
  avg(p.blue_win) AS blue_win_rate
FROM {_sql_ident(cfg.player_pivot_table)} AS p
INNER JOIN game_data.info AS gi
  ON p.matchid = gi.matchid
WHERE p.split = 'train'
  AND gi.season = {int(season)}
  AND gi.patch = {int(patch)}
"""
    result = client.query(query, settings={"max_query_size": 200_000_000})
    if not result.result_rows:
        return 0, 0.5
    row = result.result_rows[0]
    matchups = int(row[0] or 0)
    blue_win_rate = float(row[1]) if matchups > 0 and row[1] is not None else 0.5
    if not math.isfinite(blue_win_rate):
        blue_win_rate = 0.5
    return matchups, blue_win_rate


@dataclass(frozen=True)
class ServingPatchFeatureProvider:
    """Fixed draft-time patch feature provider for a served `(season, patch)`."""

    season: int
    patch: int
    features: np.ndarray
    matchups: int
    blue_win_rate: float

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32).reshape(-1)
        if features.shape != (PATCH_FEATURE_DIM,):
            raise ValueError(
                f"serving patch feature vector must have shape ({PATCH_FEATURE_DIM},); "
                f"got {features.shape}"
            )
        if not np.isfinite(features).all():
            raise ValueError("serving patch feature vector must be finite")
        object.__setattr__(self, "season", int(self.season))
        object.__setattr__(self, "patch", int(self.patch))
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "matchups", int(self.matchups))
        object.__setattr__(self, "blue_win_rate", float(self.blue_win_rate))

    @classmethod
    def from_train_aggregate(
        cls,
        *,
        cfg: DatasetConfig,
        season: int,
        patch: int,
    ) -> "ServingPatchFeatureProvider":
        _validate_source_contract(cfg)
        client = get_client()
        matchups, blue_win_rate = _load_serving_patch_aggregate(
            client,
            cfg,
            season=int(season),
            patch=int(patch),
        )
        return cls(
            season=int(season),
            patch=int(patch),
            features=_serving_feature_vector(matchups, blue_win_rate),
            matchups=matchups,
            blue_win_rate=blue_win_rate,
        )

    def features_for_batch(self, n: int) -> np.ndarray:
        n_rows = int(n)
        if n_rows < 0:
            raise ValueError("batch size must be non-negative")
        return np.tile(self.features.reshape(1, PATCH_FEATURE_DIM), (n_rows, 1))


def write_patch_feature_arrays(
    *,
    cfg: DatasetConfig,
    arrays: dict[str, np.ndarray],
    split_counts: dict[str, int],
    split_order: tuple[str, ...],
) -> None:
    """Fill the production patch feature memmap.

    The aggregate is train-only. Train rows subtract the candidate game's own
    blue-side outcome from the matching (season, patch) cell before computing
    the empirical-Bayes logit delta against an even 0.5 base rate.
    """
    required = {"patch_features", "blue_win"}
    missing = sorted(name for name in required if name not in arrays)
    if missing:
        raise ValueError("feature arrays are missing: " + ", ".join(missing))
    _validate_source_contract(cfg)
    client = get_client()
    tables = _prepare_aggregate_tables(client, _tmp_suffix())
    try:
        offset = 0
        for split_name in split_order:
            db_split = SPLIT_DB_NAMES[split_name]
            count = int(split_counts[split_name])
            patch, labels = _load_features(client, cfg, db_split, count, tables)
            if patch.shape[0] != count:
                raise RuntimeError(
                    f"{split_name} feature rows do not match cache rows: "
                    f"patch={patch.shape[0]} expected={count}"
                )
            cached_labels = arrays["blue_win"][offset : offset + count]
            if not np.array_equal(
                labels.astype(np.uint8), cached_labels.astype(np.uint8)
            ):
                raise RuntimeError(
                    f"{split_name} patch labels do not align with cache labels"
                )
            arrays["patch_features"][offset : offset + count] = patch
            offset += count
            logger.info("Wrote split %s patch features: %d games", split_name, count)
    finally:
        _drop_aggregate_tables(client, tables)
    flush = getattr(arrays["patch_features"], "flush", None)
    if flush is not None:
        flush()


def feature_metadata() -> dict[str, object]:
    return {
        "patch_feature_names": list(PATCH_FEATURE_NAMES),
        "patch_signed_feature_indices": list(PATCH_SIGNED_FEATURE_INDICES),
        "patch_temporal_scope": "season_patch_blue_side_only",
        "uses_player_identity_features": False,
    }


__all__ = [
    "PATCH_FEATURE_DIM",
    "PATCH_FEATURE_NAMES",
    "PATCH_SIGNED_FEATURE_INDICES",
    "ServingPatchFeatureProvider",
    "feature_metadata",
    "write_patch_feature_arrays",
]
