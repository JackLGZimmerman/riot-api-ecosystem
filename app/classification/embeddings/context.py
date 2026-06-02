"""identity_context descriptor: the "context atlas" artifact.

Generalises the 9-dim matchup profile to a per-identity context vector keyed by
``(championid, teamposition, build)``. The first axes are interpretable
natural-unit axes (offense mix, resistance fractions, damage / damage-taken /
heal-shield / crowd-control / siege / scaling pressure); the tail is a dense
low-rank PCA summary of the remaining allowed metrics. The HGNN's unified
antisymmetric context head crosses one team's context against the other's, so
interpretable axes keep their physical meaning (a physical-damage share, an
armor fraction) while the tail captures residual variance.

No challenge metrics enter this path (see ``identity_context_feature_set``); all
values are train-split identity aggregates, so the descriptor is draft-time
estimable and never reads the current match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.classification.embeddings.config import (
    IDENTITY_CONTEXT_CACHE_PATH,
    IDENTITY_CONTEXT_INTERP_DIM,
    IDENTITY_CONTEXT_INTERP_FEATURES,
    IDENTITY_CONTEXT_INTERP_SOURCE_FEATURES,
    IDENTITY_CONTEXT_LOWRANK_DIM,
    IDENTITY_CONTEXT_RAW_FEATURES,
    EmbeddingConfig,
    IdentityType,
    identity_context_feature_set,
    identity_context_raw_extra_feature_set,
)
from app.classification.embeddings.dense import _fixed_pca
from app.classification.embeddings.load import LevelRows, load_all
from app.classification.embeddings.matrices import (
    _identity_key_strings,
    _resolve_feature_values,
    build_level_matrix,
)
from app.core.logging.logger import setup_logging_config
from app.core.utils.smoothing import apply_hierarchical_shrinkage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentityContextResult:
    path: Path
    n_identities: int
    dim: int
    interpretable_dim: int
    lowrank_dim: int
    raw_dim: int


def _pressure_axis(values: np.ndarray) -> np.ndarray:
    """Robust-scale a non-negative per-minute volume into a [0, 1] pressure axis.

    Same convention as the matchup profile's ``champion_damage_pressure``: divide
    by the 95th percentile of finite-positive values, then clip to [0, 1].
    """
    finite = values[np.isfinite(values) & (values > 0.0)]
    scale = float(np.percentile(finite, 95)) if finite.size else 1.0
    if not np.isfinite(scale) or scale <= 1.0e-9:
        scale = 1.0
    return np.clip(values / scale, 0.0, 1.0)


def _interpretable_axes(rows: LevelRows, sorted_row_idx: np.ndarray) -> np.ndarray:
    raw = np.stack(
        _resolve_feature_values(rows, IDENTITY_CONTEXT_INTERP_SOURCE_FEATURES, sorted_row_idx),
        axis=-1,
    ).astype(np.float64)
    (
        phys_off,
        magic_off,
        true_off,
        armor,
        magicresist,
        champion_damage,
        damage_taken,
        heal_shield,
        cc,
        structure_damage,
        goldearned,
    ) = (raw[:, i] for i in range(raw.shape[1]))

    resist_denom = armor + magicresist
    safe = resist_denom > 1e-9
    armor_frac = np.where(safe, armor / np.where(safe, resist_denom, 1.0), 0.5)
    mr_frac = np.where(safe, magicresist / np.where(safe, resist_denom, 1.0), 0.5)
    damage_pressure = _pressure_axis(champion_damage)

    axes = np.stack(
        [
            phys_off,
            magic_off,
            true_off,
            armor_frac,
            mr_frac,
            damage_pressure,
            damage_pressure * phys_off,
            damage_pressure * magic_off,
            damage_pressure * true_off,
            _pressure_axis(damage_taken),
            _pressure_axis(heal_shield),
            _pressure_axis(cc),
            _pressure_axis(structure_damage),
            _pressure_axis(goldearned),
        ],
        axis=-1,
    )
    return np.clip(axes, 0.0, 1.0).astype(np.float32)


def write_identity_context_embeddings(
    smoothed_levels: dict[IdentityType, LevelRows] | None = None,
    *,
    output_path: Path = IDENTITY_CONTEXT_CACHE_PATH,
    lowrank_dim: int = IDENTITY_CONTEXT_LOWRANK_DIM,
) -> IdentityContextResult:
    if smoothed_levels is None:
        cfg = EmbeddingConfig()
        smoothed_levels = apply_hierarchical_shrinkage(load_all(cfg), cfg)

    rows = smoothed_levels[IdentityType.BASELINE]
    key_cols = rows.key_columns
    identity_strs = _identity_key_strings(rows, key_cols)
    unique_keys, first_idx = np.unique(identity_strs, return_index=True)[:2]
    sorted_row_idx = first_idx[np.argsort(unique_keys)]

    interpretable = _interpretable_axes(rows, sorted_row_idx)

    # Dense low-rank tail over the remaining allowed (challenge-free) metrics.
    lowrank_matrix = build_level_matrix(
        rows,
        EmbeddingConfig(feature_set=identity_context_feature_set()),
    )
    if lowrank_matrix is None:
        raise ValueError("Cannot write identity_context for an empty baseline")
    lowrank = _fixed_pca(lowrank_matrix.matrix, lowrank_dim)

    # Wide RAW block: interpretable natural-unit axes (kept first so indices line
    # up with the compressed descriptor) followed by median/MAD-standardised
    # extra draft-safe metrics. NOT PCA-compressed: this is the primary
    # interaction source for the identity-conditioned head.
    raw_extra_matrix = build_level_matrix(
        rows,
        EmbeddingConfig(feature_set=identity_context_raw_extra_feature_set()),
    )
    if raw_extra_matrix is None:
        raise ValueError("Cannot write identity_context raw block for an empty baseline")

    keys = [tuple(rows.columns[c][i] for c in key_cols) for i in sorted_row_idx.tolist()]
    if lowrank_matrix.keys != keys or raw_extra_matrix.keys != keys:
        raise AssertionError("identity_context interpretable / low-rank / raw rows are misaligned")

    # Clip the standardised extra block to a robust range: a handful of ratio
    # features (e.g. enchanter ally-support per gold) have near-zero MAD, so the
    # few identities that express them land thousands of std units out and would
    # otherwise dominate the linear projector. The natural-unit interpretable
    # axes (kept first, in [0, 1]) already carry the bounded version of those
    # signals.
    raw_extra = np.clip(raw_extra_matrix.matrix.astype(np.float32), -8.0, 8.0)
    raw_embeddings = np.concatenate([interpretable, raw_extra], axis=1).astype(np.float32)
    raw_feature_names = IDENTITY_CONTEXT_RAW_FEATURES

    embeddings = np.concatenate([interpretable, lowrank], axis=1).astype(np.float32)
    feature_names = (
        *IDENTITY_CONTEXT_INTERP_FEATURES,
        *(f"lowrank_{i}" for i in range(lowrank.shape[1])),
    )
    matchups = rows.columns["matchups"][sorted_row_idx].astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        keys=np.array(keys, dtype=object),
        key_columns=np.array(key_cols, dtype=object),
        embeddings=embeddings,
        raw_embeddings=raw_embeddings,
        matchups=matchups,
        feature_names=np.array(feature_names, dtype=object),
        raw_feature_names=np.array(raw_feature_names, dtype=object),
        dim=np.array(embeddings.shape[1], dtype=np.int32),
        interpretable_dim=np.array(IDENTITY_CONTEXT_INTERP_DIM, dtype=np.int32),
        raw_dim=np.array(raw_embeddings.shape[1], dtype=np.int32),
    )
    result = IdentityContextResult(
        path=output_path,
        n_identities=embeddings.shape[0],
        dim=embeddings.shape[1],
        interpretable_dim=IDENTITY_CONTEXT_INTERP_DIM,
        lowrank_dim=lowrank.shape[1],
        raw_dim=raw_embeddings.shape[1],
    )
    logger.info(
        "Wrote identity context: path=%s identities=%d dim=%d (interp=%d lowrank=%d raw=%d)",
        output_path,
        result.n_identities,
        result.dim,
        result.interpretable_dim,
        result.lowrank_dim,
        result.raw_dim,
    )
    return result


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.INFO)
    cfg = EmbeddingConfig()
    smoothed_levels = apply_hierarchical_shrinkage(load_all(cfg), cfg)
    write_identity_context_embeddings(smoothed_levels)


if __name__ == "__main__":
    main()
