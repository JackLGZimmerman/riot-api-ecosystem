"""Assemble per-identity snapshot feature matrices from smoothed LevelRows.

Identities with fewer than `len(PHASES)` phase rows are dropped. Each feature
column is signed-log1p compressed then standardised with median/MAD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.classification.embeddings.config import (
    ALL_METRICS,
    DERIVED_METRIC_FUNCS,
    PHASE_INDEX,
    PHASES,
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.load import LevelRows

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevelMatrix:
    level: IdentityType
    keys: list[tuple]  # length n; one tuple per identity
    key_columns: tuple[str, ...]
    matrix: np.ndarray  # (n, D) float32, standardised
    feature_names: tuple[str, ...]
    matchups: np.ndarray  # (n, N) float32


def _identity_key_strings(rows: LevelRows, key_cols: tuple[str, ...]) -> np.ndarray:
    cols = [rows.columns[c] for c in key_cols]
    return np.array(
        ["\x00".join(str(c[i]) for c in cols) for i in range(rows.n)],
        dtype=object,
    )


def _standardise_columns(values: np.ndarray) -> np.ndarray:
    flat = np.sign(values) * np.log1p(np.abs(values))
    med = np.median(flat, axis=0, keepdims=True)
    mad = np.median(np.abs(flat - med), axis=0, keepdims=True) * 1.4826
    mad = np.where(mad > 1e-8, mad, 1.0)
    return ((flat - med) / mad).astype(np.float32)


def _resolve_feature_values(
    rows: LevelRows,
    feature_set: tuple[str, ...],
    sorted_row_idx: np.ndarray,
) -> list[np.ndarray]:
    """Return one (n_rows,) float32 array per requested feature.

    Raw entries map to `smoothed_<name>`. Derived entries are computed via
    `DERIVED_METRIC_FUNCS` against the smoothed columns of the same rows.
    """
    smoothed_cache: dict[str, np.ndarray] = {}

    def smoothed(metric: str) -> np.ndarray:
        key = f"smoothed_{metric}"
        if key not in smoothed_cache:
            smoothed_cache[key] = rows.columns[key][sorted_row_idx].astype(np.float32)
        return smoothed_cache[key]

    # Materialise raw smoothed columns for any metric mentioned by raw or
    # derived entries — derived funcs reach into the cache by `smoothed_<name>`.
    for metric in ALL_METRICS:
        smoothed(metric)

    out: list[np.ndarray] = []
    for name in feature_set:
        if name in ALL_METRICS:
            out.append(smoothed(name))
        elif name in DERIVED_METRIC_FUNCS:
            out.append(
                DERIVED_METRIC_FUNCS[name](smoothed_cache).astype(np.float32)
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
    n_phases = len(PHASES)

    identity_strs = _identity_key_strings(rows, key_cols)
    _, inverse, counts = np.unique(
        identity_strs, return_inverse=True, return_counts=True
    )
    complete_mask = counts[inverse] == n_phases
    if not complete_mask.any():
        return None

    keep_idx = np.where(complete_mask)[0]
    phase_arr = rows.columns["phase"]
    phase_idx = np.fromiter(
        (PHASE_INDEX[str(phase_arr[i])] for i in keep_idx),
        dtype=np.int64,
        count=keep_idx.size,
    )
    sorter = np.lexsort((phase_idx, inverse[keep_idx]))
    sorted_row_idx = keep_idx[sorter]
    n_identities = sorted_row_idx.size // n_phases

    feature_cols = _resolve_feature_values(rows, cfg.feature_set, sorted_row_idx)
    flat = np.stack(feature_cols, axis=-1)
    n_features = len(cfg.feature_set)
    base_matrix = flat.reshape(n_identities, n_phases, n_features)
    matchups = (
        rows.columns["matchups"][sorted_row_idx]
        .astype(np.float32)
        .reshape(n_identities, n_phases)
    )

    raw = base_matrix.reshape(-1, base_matrix.shape[-1])
    standardised = _standardise_columns(raw).reshape(base_matrix.shape)
    matrix = standardised.reshape(n_identities, -1).astype(np.float32)
    feature_names = tuple(
        f"{phase}_{name}" for phase in PHASES for name in cfg.feature_set
    )

    first_rows = sorted_row_idx[::n_phases]
    keys = [tuple(rows.columns[c][i] for c in key_cols) for i in first_rows.tolist()]
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
            "Matrix %s: n=%d, D=%d",
            level.value,
            lm.matrix.shape[0],
            lm.matrix.shape[1],
        )
        out[level] = lm
    return out
