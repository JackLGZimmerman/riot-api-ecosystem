"""PCA-truncate per-identity feature matrices then L2-normalise."""

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


@dataclass(frozen=True)
class LevelEmbeddings:
    level: IdentityType
    keys: list[tuple]
    key_columns: tuple[str, ...]
    embeddings: np.ndarray  # (n, D) float32, L2-normalised
    feature_names: tuple[str, ...]
    matchups: np.ndarray


def _pca_truncate(flat: np.ndarray, keep_variance: float) -> np.ndarray:
    x = flat.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return x.astype(np.float32)
    cum = np.cumsum(eigenvalues) / total
    k = int(np.searchsorted(cum, max(0.0, min(keep_variance, 1.0))) + 1)
    k = max(1, min(k, eigenvectors.shape[1]))
    return (x @ eigenvectors[:, :k]).astype(np.float32)


def embed_level(
    matrix: LevelMatrix, cfg: EmbeddingConfig | None = None
) -> LevelEmbeddings:
    cfg = cfg or EmbeddingConfig()
    flat = matrix.matrix.reshape(matrix.matrix.shape[0], -1)
    flat = _pca_truncate(flat, cfg.projection_keep_variance)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    z = (flat / np.where(norms > 1e-8, norms, 1.0)).astype(np.float32)
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
        logger.info("Embedded %s: n=%d, D=%d", level.value, *e.embeddings.shape)
    return out
