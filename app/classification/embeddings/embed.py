"""PCA-truncate temporal feature matrices then L2-normalise.

Temporal matrices stay shaped as `(identity, phase, feature)`. PCA is fit once
over all identity-phase rows, giving every phase a shared latent space without
pooling phase rows into a single identity embedding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.classification.embeddings.config import (
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.matrices import LevelMatrix

logger = logging.getLogger(__name__)

PCAFit = tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]


@dataclass(frozen=True)
class LevelEmbeddings:
    level: IdentityType
    keys: list[tuple]
    key_columns: tuple[str, ...]
    embeddings: np.ndarray  # (n, phases, D) float32, L2-normalised
    feature_names: tuple[str, ...]
    matchups: np.ndarray


def fit_pca_basis(flat: np.ndarray, keep_variance: float) -> PCAFit:
    x = flat.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return x, eigenvalues, eigenvectors, 0, np.zeros_like(eigenvalues)
    ratios = eigenvalues / total
    cum = np.cumsum(ratios)
    k = int(np.searchsorted(cum, max(0.0, min(keep_variance, 1.0))) + 1)
    k = max(1, min(k, eigenvectors.shape[1]))
    return x, eigenvalues, eigenvectors, k, ratios


def _pca_truncate(flat: np.ndarray, keep_variance: float) -> np.ndarray:
    x, _, eigenvectors, k, _ = fit_pca_basis(flat, keep_variance)
    if k <= 0:
        return x.astype(np.float32)
    return (x @ eigenvectors[:, :k]).astype(np.float32)


def _l2_normalise(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return (values / np.where(norms > 1e-8, norms, 1.0)).astype(np.float32)


def embed_level(
    matrix: LevelMatrix, cfg: EmbeddingConfig | None = None
) -> LevelEmbeddings:
    cfg = cfg or EmbeddingConfig()
    if matrix.matrix.ndim == 3:
        n_identities, n_phases, n_features = matrix.matrix.shape
        stacked = matrix.matrix.reshape(n_identities * n_phases, n_features)
        projected = _pca_truncate(stacked, cfg.projection_keep_variance)
        z = _l2_normalise(projected).reshape(n_identities, n_phases, -1)
    elif matrix.matrix.ndim == 2:
        projected = _pca_truncate(matrix.matrix, cfg.projection_keep_variance)
        z = _l2_normalise(projected)
    else:
        raise ValueError(f"matrix must be 2-D or 3-D, got shape {matrix.matrix.shape}")
    return LevelEmbeddings(
        level=matrix.level,
        keys=matrix.keys,
        key_columns=matrix.key_columns,
        embeddings=z,
        feature_names=matrix.feature_names,
        matchups=matrix.matchups,
    )


def embed_all(
    matrices: dict[IdentityType, LevelMatrix],
    cfg: EmbeddingConfig | None = None,
) -> dict[IdentityType, LevelEmbeddings]:
    cfg = cfg or EmbeddingConfig()
    out = {level: embed_level(m, cfg) for level, m in matrices.items()}
    for level, e in out.items():
        if e.embeddings.ndim == 3:
            logger.info(
                "Embedded %s: n=%d, phases=%d, D=%d",
                level.value,
                *e.embeddings.shape,
            )
        else:
            logger.info("Embedded %s: n=%d, D=%d", level.value, *e.embeddings.shape)
    return out
