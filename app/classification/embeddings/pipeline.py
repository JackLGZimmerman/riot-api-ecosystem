"""End-to-end orchestrator: aggregate rows -> smoothed rows -> matrices ->
per-specialist embeddings -> saved labels + singular orderings + report.

Run with:
    python -m app.classification.embeddings.pipeline
"""

from __future__ import annotations

from app.classification.embeddings.config import EmbeddingConfig
from app.classification.embeddings.context import write_identity_context_embeddings
from app.classification.embeddings.dense import write_dense_identity_embeddings
from app.classification.embeddings.load import load_all
from app.classification.embeddings.relationship_details import (
    write_relationship_detail_embeddings,
)
from app.classification.embeddings.report import write_specialist_report
from app.classification.embeddings.singular_metrics import (
    log_singular_metric_results,
    run_all_singular_metrics,
)
from app.classification.embeddings.specialists import (
    log_results,
    run_all_specialists,
)
from app.core.logging.logger import setup_logging_config
from app.core.utils.smoothing import apply_hierarchical_shrinkage

setup_logging_config()


def run(cfg: EmbeddingConfig | None = None) -> None:
    cfg = cfg or EmbeddingConfig()
    smoothed = apply_hierarchical_shrinkage(load_all(cfg), cfg)
    write_dense_identity_embeddings(smoothed)
    write_identity_context_embeddings(smoothed)
    write_relationship_detail_embeddings()
    log_results(run_all_specialists(smoothed))
    log_singular_metric_results(run_all_singular_metrics(smoothed))
    write_specialist_report(cfg)


if __name__ == "__main__":
    run()
