"""End-to-end orchestrator: 6010 + 9000-9040 priors -> posteriors ->
matrices -> embeddings -> saved npz + diagnostics.

Run with:
    python -m app.classification.embeddings.pipeline
"""

from __future__ import annotations

import logging

from app.classification.embeddings import embed
from app.classification.embeddings.config import EmbeddingConfig, IdentityType
from app.classification.embeddings.load import load_all
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.classification.embeddings.report import write_grouping_report
from app.classification.embeddings.test_pairs import (
    evaluate_semantic_pairs,
    log_semantic_score,
)
from app.classification.embeddings.validate import diagnose_all
from app.core.logging.logger import setup_logging_config

setup_logging_config()
logger = logging.getLogger(__name__)


def run(cfg: EmbeddingConfig | None = None) -> None:
    cfg = cfg or EmbeddingConfig()
    levels = load_all(cfg)
    smoothed = apply_hierarchical_shrinkage(levels, cfg)
    matrices = build_all_matrices(smoothed, cfg)
    embeddings = embed.embed_all(matrices, cfg)
    embed.save(embeddings, cfg.cache_dir)
    if baseline := embeddings.get(IdentityType.BASELINE):
        write_grouping_report(baseline, cfg)
    diagnose_all(
        embeddings,
        cfg.similarity_threshold,
        min_matchups_by_level=cfg.group_min_matchups,
    )
    if baseline := embeddings.get(IdentityType.BASELINE):
        log_semantic_score(evaluate_semantic_pairs(baseline, cfg.similarity_threshold))


if __name__ == "__main__":
    run()
