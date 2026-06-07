"""Runtime lookup for smoothed identity semantic context rows.

The hierarchical Bayesian smoothing itself lives in app.core.utils.smoothing.
This module only adapts the smoothed classification identity rows into the
compact semantic context surface consumed by the HGNN semantic group features.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from app.ml.semantic_group_features import (
    SEMANTIC_CONTEXT_RAW_DIM,
    build_identity_context_raw_from_metrics,
)


@dataclass(frozen=True)
class SemanticContextRawLookup:
    values: dict[tuple[int, str, str], np.ndarray]

    def lookup(self, tuples: Iterable[tuple[int, str, str]]) -> np.ndarray:
        items = list(tuples)
        out = np.zeros((len(items), SEMANTIC_CONTEXT_RAW_DIM), dtype=np.float32)
        get = self.values.get
        for i, key in enumerate(items):
            value = get(key)
            if value is not None:
                out[i] = value
        return out


def load_semantic_context_raw_lookup() -> SemanticContextRawLookup:
    from app.classification.embeddings.config import (
        ALL_METRICS,
        EmbeddingConfig,
        IdentityType,
    )
    from app.classification.embeddings.load import load_all
    from app.classification.embeddings.registry import DERIVED_METRIC_FUNCS
    from app.core.utils.smoothing import apply_hierarchical_shrinkage

    cfg = EmbeddingConfig(split="train", include_context_features=False)
    levels = apply_hierarchical_shrinkage(load_all(cfg), cfg)
    baseline = levels[IdentityType.BASELINE]
    metric_values: dict[str, np.ndarray] = {
        name: np.asarray(baseline.columns[name], dtype=np.float32)
        for name in ALL_METRICS
    }
    for name, func in DERIVED_METRIC_FUNCS.items():
        metric_values[name] = np.asarray(func(metric_values), dtype=np.float32)
    context_raw = build_identity_context_raw_from_metrics(metric_values)
    champion = np.asarray(baseline.columns["championid"], dtype=np.int64)
    position = np.asarray(baseline.columns["teamposition"], dtype=object)
    build = np.asarray(baseline.columns["build"], dtype=object)
    return SemanticContextRawLookup(
        values={
            (int(champion[i]), str(position[i]), str(build[i])): context_raw[i].copy()
            for i in range(context_raw.shape[0])
        }
    )


__all__ = ["SemanticContextRawLookup", "load_semantic_context_raw_lookup"]
