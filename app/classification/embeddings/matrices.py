"""Assemble per-identity feature matrices from smoothed LevelRows.

Each feature column is signed-log1p compressed then standardised with median/MAD
across all identity rows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from app.classification.embeddings.config import (
    ALL_METRICS,
    DERIVED_METRIC_FUNCS,
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.load import LevelRows
from app.core.utils.common import median_mad_standardise

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevelMatrix:
    level: IdentityType
    keys: list[tuple]  # length n; one tuple per identity
    key_columns: tuple[str, ...]
    matrix: np.ndarray  # (n, features) float32, standardised
    feature_names: tuple[str, ...]  # names along the feature axis
    matchups: np.ndarray  # (n,) float32


def _identity_key_strings(rows: LevelRows, key_cols: tuple[str, ...]) -> np.ndarray:
    cols = [rows.columns[c] for c in key_cols]
    return np.array(
        ["\x00".join(str(c[i]) for c in cols) for i in range(rows.n)],
        dtype=object,
    )


class _MetricValues(Mapping[str, np.ndarray]):
    def __init__(self, rows: LevelRows, row_idx: np.ndarray) -> None:
        self._rows = rows
        self._row_idx = row_idx
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, metric: str) -> np.ndarray:
        if metric not in ALL_METRICS:
            raise KeyError(metric)
        if metric not in self._cache:
            self._cache[metric] = self._rows.columns[f"smoothed_{metric}"][
                self._row_idx
            ].astype(np.float32)
        return self._cache[metric]

    def __iter__(self) -> Iterator[str]:
        return iter(ALL_METRICS)

    def __len__(self) -> int:
        return len(ALL_METRICS)


def _resolve_feature_values(
    rows: LevelRows,
    feature_set: tuple[str, ...],
    sorted_row_idx: np.ndarray,
) -> list[np.ndarray]:
    """Return one (n_rows,) float32 array per requested feature.

    Raw entries pull from the `smoothed_<name>` columns of `rows`. Derived
    entries are computed via `DERIVED_METRIC_FUNCS` against the same values.
    """
    metric_values = _MetricValues(rows, sorted_row_idx)

    out: list[np.ndarray] = []
    for name in feature_set:
        if name in ALL_METRICS:
            out.append(metric_values[name])
        elif name in DERIVED_METRIC_FUNCS:
            out.append(
                DERIVED_METRIC_FUNCS[name](metric_values).astype(np.float32)
            )
        else:
            raise KeyError(
                f"Unknown feature '{name}' (not in ALL_METRICS or DERIVED_METRIC_FUNCS)"
            )
    return out


def build_level_matrix(
    rows: LevelRows, cfg: EmbeddingConfig | None = None
) -> LevelMatrix | None:
    if rows.n == 0:
        return None
    cfg = cfg or EmbeddingConfig()
    key_cols = rows.key_columns

    identity_strs = _identity_key_strings(rows, key_cols)
    unique_keys, first_idx, counts = np.unique(
        identity_strs, return_index=True, return_counts=True
    )
    if unique_keys.size == 0:
        return None
    if np.any(counts > 1):
        logger.warning(
            "Level %s has %d duplicate identity rows; keeping first occurrence",
            rows.level.value,
            int(np.sum(counts > 1)),
        )

    sorted_row_idx = first_idx[np.argsort(unique_keys)]

    feature_cols = _resolve_feature_values(rows, cfg.feature_set, sorted_row_idx)
    base_matrix = np.stack(feature_cols, axis=-1)
    matchups = rows.columns["matchups"][sorted_row_idx].astype(np.float32)

    standardised, _, _ = median_mad_standardise(base_matrix)
    if cfg.matrix_clip_value is not None:
        clip = float(cfg.matrix_clip_value)
        if clip <= 0.0:
            raise ValueError("matrix_clip_value must be positive when set")
        standardised = np.clip(standardised, -clip, clip)
    matrix = standardised.astype(np.float32)
    feature_names = tuple(cfg.feature_set)

    keys = [tuple(rows.columns[c][i] for c in key_cols) for i in sorted_row_idx.tolist()]
    return LevelMatrix(
        level=rows.level,
        keys=keys,
        key_columns=key_cols,
        matrix=matrix,
        feature_names=feature_names,
        matchups=matchups,
    )


def build_all_matrices(
    smoothed_levels: dict[IdentityType, LevelRows],
    cfg: EmbeddingConfig | None = None,
) -> dict[IdentityType, LevelMatrix]:
    cfg = cfg or EmbeddingConfig()
    out: dict[IdentityType, LevelMatrix] = {}
    for level, rows in smoothed_levels.items():
        lm = build_level_matrix(rows, cfg)
        if lm is None:
            logger.warning(
                "No matrix for level %s (empty after filtering)", level.value
            )
            continue
        logger.info(
            "Matrix %s: n=%d, features=%d",
            level.value,
            *lm.matrix.shape,
        )
        out[level] = lm
    return out
