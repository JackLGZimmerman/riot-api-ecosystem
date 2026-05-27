"""End-to-end orchestrator: 6010 + 9000-9040 priors -> posteriors ->
matrices -> per-specialist embeddings -> saved labels + report.

Run with:
    python -m app.classification.embeddings.pipeline
"""

from __future__ import annotations

import logging

from app.classification.embeddings.config import EmbeddingConfig
from app.classification.embeddings.load import load_all
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.classification.embeddings.report import write_specialist_report
from app.classification.embeddings.specialists import (
    log_results,
    run_all_specialists,
)
from app.core.logging.logger import setup_logging_config

setup_logging_config()
logger = logging.getLogger(__name__)


def run(cfg: EmbeddingConfig | None = None) -> None:
    cfg = cfg or EmbeddingConfig()
    smoothed = apply_hierarchical_shrinkage(load_all(cfg), cfg)
    log_results(run_all_specialists(smoothed))
    write_specialist_report(cfg)


if __name__ == "__main__":
    run()
