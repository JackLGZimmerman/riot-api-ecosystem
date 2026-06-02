"""Build raw and derived classification metric matrices.

Run with:
    python -m app.classification.embeddings.pipeline
"""

from __future__ import annotations

import logging

from app.classification.embeddings.config import EmbeddingConfig, IdentityType
from app.classification.embeddings.load import load_all
from app.classification.embeddings.matrices import LevelMatrix, build_all_matrices
from app.core.logging.logger import setup_logging_config
from app.core.utils.smoothing import apply_hierarchical_shrinkage

logger = logging.getLogger(__name__)


def build_metric_matrices(
    cfg: EmbeddingConfig | None = None,
) -> dict[IdentityType, LevelMatrix]:
    """Load, smooth, and standardise the preserved raw/derived metric catalogue."""
    cfg = cfg or EmbeddingConfig()
    smoothed = apply_hierarchical_shrinkage(load_all(cfg), cfg)
    return build_all_matrices(smoothed, cfg)


def run(cfg: EmbeddingConfig | None = None) -> dict[IdentityType, LevelMatrix]:
    setup_logging_config()
    matrices = build_metric_matrices(cfg)
    for level, matrix in matrices.items():
        logger.info(
            "%s raw/derived matrix: rows=%d features=%d",
            level.value,
            matrix.matrix.shape[0],
            matrix.matrix.shape[1],
        )
    return matrices


if __name__ == "__main__":
    run()
