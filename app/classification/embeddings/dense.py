"""Dense identity semantic embedding artifact.

The specialist files remain inspectable labels. This module writes the compact
continuous identity descriptor consumed by the HGNN node feature path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    DENSE_IDENTITY_CACHE_PATH,
    DENSE_IDENTITY_DIM,
    IDENTITY_PROFILE_CACHE_PATH,
    IDENTITY_PROFILE_DIM,
    IDENTITY_PROFILE_FEATURES,
    IDENTITY_PROFILE_SOURCE_FEATURES,
    EmbeddingConfig,
    IdentityType,
    identity_semantic_feature_set,
)
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
class DenseIdentityResult:
    path: Path
    n_identities: int
    dim: int
    n_features: int


def _fixed_pca(values: np.ndarray, dim: int) -> np.ndarray:
    x, _, eigenvectors, k, _ = embed.fit_pca_basis(values, keep_variance=1.0)
    keep = min(dim, k, eigenvectors.shape[1])
    projected = (x @ eigenvectors[:, :keep]).astype(np.float32) if keep else x[:, :0].astype(np.float32)
    if projected.shape[1] < dim:
        pad = np.zeros((projected.shape[0], dim - projected.shape[1]), dtype=np.float32)
        projected = np.concatenate([projected, pad], axis=1)
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    return (projected / np.where(norms > 1e-8, norms, 1.0)).astype(np.float32)


def write_dense_identity_embeddings(
    smoothed_levels: dict[IdentityType, LevelRows] | None = None,
    *,
    output_path: Path = DENSE_IDENTITY_CACHE_PATH,
    dim: int = DENSE_IDENTITY_DIM,
) -> DenseIdentityResult:
    if smoothed_levels is None:
        cfg = EmbeddingConfig()
        smoothed_levels = apply_hierarchical_shrinkage(load_all(cfg), cfg)

    feature_set = identity_semantic_feature_set()
    matrix = build_level_matrix(
        smoothed_levels[IdentityType.BASELINE],
        EmbeddingConfig(feature_set=feature_set),
    )
    if matrix is None:
        raise ValueError("Cannot write dense identity embeddings for an empty baseline")

    embeddings = _fixed_pca(matrix.matrix, dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        keys=np.array(matrix.keys, dtype=object),
        key_columns=np.array(matrix.key_columns, dtype=object),
        embeddings=embeddings,
        matchups=matrix.matchups.astype(np.float32),
        feature_names=np.array(matrix.feature_names, dtype=object),
        dim=np.array(dim, dtype=np.int32),
    )
    result = DenseIdentityResult(
        path=output_path,
        n_identities=embeddings.shape[0],
        dim=embeddings.shape[1],
        n_features=len(feature_set),
    )
    logger.info(
        "Wrote dense identity embeddings: path=%s identities=%d dim=%d features=%d",
        output_path,
        result.n_identities,
        result.dim,
        result.n_features,
    )
    return result


def write_identity_profile_embeddings(
    smoothed_levels: dict[IdentityType, LevelRows] | None = None,
    *,
    output_path: Path = IDENTITY_PROFILE_CACHE_PATH,
) -> DenseIdentityResult:
    """Interpretable per-identity matchup profile (offense + resistance axes).

    Unlike the dense semantic descriptor this keeps RAW [0, 1] axes (no PCA, no
    median/MAD standardisation): the HGNN crosses one team's profile against the
    other's, so the axes must keep their physical meaning (a physical-damage
    share, an armor-resistance fraction). Keyed identically to the dense
    descriptor (championid, teamposition, build) for the same runtime lookup.
    """
    if smoothed_levels is None:
        cfg = EmbeddingConfig()
        smoothed_levels = apply_hierarchical_shrinkage(load_all(cfg), cfg)

    rows = smoothed_levels[IdentityType.BASELINE]
    key_cols = rows.key_columns
    identity_strs = _identity_key_strings(rows, key_cols)
    unique_keys, first_idx = np.unique(identity_strs, return_index=True)[:2]
    sorted_row_idx = first_idx[np.argsort(unique_keys)]
    raw = np.stack(
        _resolve_feature_values(rows, IDENTITY_PROFILE_SOURCE_FEATURES, sorted_row_idx),
        axis=-1,
    ).astype(np.float64)

    phys_off, magic_off, true_off, armor, magicresist, champion_damage = (
        raw[:, i] for i in range(6)
    )
    resist_denom = armor + magicresist
    safe = resist_denom > 1e-9
    armor_frac = np.where(safe, armor / np.where(safe, resist_denom, 1.0), 0.5)
    mr_frac = np.where(safe, magicresist / np.where(safe, resist_denom, 1.0), 0.5)
    finite_damage = champion_damage[np.isfinite(champion_damage) & (champion_damage > 0.0)]
    damage_scale = float(np.percentile(finite_damage, 95)) if finite_damage.size else 1.0
    if not np.isfinite(damage_scale) or damage_scale <= 1.0e-9:
        damage_scale = 1.0
    damage_pressure = np.clip(champion_damage / damage_scale, 0.0, 1.0)
    profile = np.clip(
        np.stack(
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
            ],
            axis=-1,
        ),
        0.0,
        1.0,
    ).astype(np.float32)

    keys = [tuple(rows.columns[c][i] for c in key_cols) for i in sorted_row_idx.tolist()]
    matchups = rows.columns["matchups"][sorted_row_idx].astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        keys=np.array(keys, dtype=object),
        key_columns=np.array(key_cols, dtype=object),
        embeddings=profile,
        matchups=matchups,
        feature_names=np.array(IDENTITY_PROFILE_FEATURES, dtype=object),
        dim=np.array(IDENTITY_PROFILE_DIM, dtype=np.int32),
    )
    result = DenseIdentityResult(
        path=output_path,
        n_identities=profile.shape[0],
        dim=profile.shape[1],
        n_features=len(IDENTITY_PROFILE_SOURCE_FEATURES),
    )
    logger.info(
        "Wrote identity matchup profiles: path=%s identities=%d dim=%d",
        output_path,
        result.n_identities,
        result.dim,
    )
    return result


def main() -> None:
    # Local import avoids a module-load cycle (context imports _fixed_pca here).
    from app.classification.embeddings.context import write_identity_context_embeddings

    setup_logging_config()
    logging.getLogger().setLevel(logging.INFO)
    cfg = EmbeddingConfig()
    smoothed_levels = apply_hierarchical_shrinkage(load_all(cfg), cfg)
    write_dense_identity_embeddings(smoothed_levels)
    write_identity_profile_embeddings(smoothed_levels)
    write_identity_context_embeddings(smoothed_levels)


if __name__ == "__main__":
    main()
