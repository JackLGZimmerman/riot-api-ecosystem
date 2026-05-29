"""Phase-relative ordered features for one-dimensional metrics.

Singular metrics complement specialists. A specialist clusters a compact
feature set into semantic groups; a singular metric keeps one meaningful axis as
a continuous ordering so downstream models can compare identities within the
same phase.

Run:
    uv run python -m app.classification.embeddings.singular_metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.classification.embeddings.config import (
    PHASES,
    SINGULAR_METRIC_CACHE_DIR,
    SINGULAR_METRICS,
    EmbeddingConfig,
    IdentityType,
    SingularMetricSpec,
)
from app.classification.embeddings.load import LevelRows, load_all
from app.classification.embeddings.matrices import LevelMatrix, build_all_matrices
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.core.logging.logger import setup_logging_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingularMetricPhaseResult:
    phase: str
    top: list[tuple[float, tuple]]
    bottom: list[tuple[float, tuple]]


@dataclass(frozen=True)
class SingularMetricResult:
    name: str
    feature: str
    n_identities: int
    phase_results: tuple[SingularMetricPhaseResult, ...]


def _average_descending_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based descending ranks, averaging ties."""
    n = values.size
    if n == 0:
        return np.empty(0, dtype=np.float32)

    order = np.argsort(-values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and np.isclose(ordered[j], ordered[i], rtol=1e-6, atol=1e-6):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _normalised_ordering(
    values: np.ndarray, *, higher_is_more: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oriented = values if higher_is_more else -values
    ranks = _average_descending_ranks(oriented.astype(np.float32, copy=False))
    if values.size <= 1:
        percentiles = np.full(values.shape, 0.5, dtype=np.float32)
    else:
        percentiles = (1.0 - ((ranks - 1.0) / (values.size - 1.0))).astype(np.float32)
    scores = ((percentiles - 0.5) * 2.0).astype(np.float32)
    return ranks.astype(np.float32), percentiles, scores


def _phase_names(n_phases: int) -> tuple[str, ...]:
    phase_names = PHASES[:n_phases]
    if len(phase_names) != n_phases:
        return tuple(f"phase_{i}" for i in range(n_phases))
    return phase_names


def _save_ordering(
    spec: SingularMetricSpec,
    baseline: LevelMatrix,
    standardised_values: np.ndarray,
    ranks: np.ndarray,
    percentiles: np.ndarray,
    scores: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{spec.name}.npz",
        keys=np.array(baseline.keys, dtype=object),
        key_columns=np.array(baseline.key_columns, dtype=object),
        phases=np.array(_phase_names(standardised_values.shape[1]), dtype=object),
        metric=np.array(spec.feature, dtype=object),
        description=np.array(spec.description, dtype=object),
        higher_is_more=np.array(spec.higher_is_more, dtype=np.bool_),
        score_axes=np.array(("identity", "phase"), dtype=object),
        standardised_values=standardised_values.astype(np.float32),
        ranks=ranks.astype(np.float32),
        percentiles=percentiles.astype(np.float32),
        scores=scores.astype(np.float32),
    )


def run_singular_metric(
    spec: SingularMetricSpec,
    baseline: LevelMatrix,
    feature_index: dict[str, int],
    *,
    output_dir: Path = SINGULAR_METRIC_CACHE_DIR,
) -> SingularMetricResult:
    if baseline.matrix.ndim != 3:
        raise ValueError(
            f"singular metric ordering requires 3-D matrices, got {baseline.matrix.shape}"
        )
    if spec.feature not in feature_index:
        available = ", ".join(feature_index)
        raise KeyError(f"Unknown singular metric feature {spec.feature!r}; loaded {available}")

    values = baseline.matrix[:, :, feature_index[spec.feature]].astype(np.float32)
    n_identities, n_phases = values.shape
    ranks = np.empty_like(values, dtype=np.float32)
    percentiles = np.empty_like(values, dtype=np.float32)
    scores = np.empty_like(values, dtype=np.float32)

    phase_results: list[SingularMetricPhaseResult] = []
    phases = _phase_names(n_phases)
    for phase_index, phase in enumerate(phases):
        phase_values = values[:, phase_index]
        phase_ranks, phase_percentiles, phase_scores = _normalised_ordering(
            phase_values,
            higher_is_more=spec.higher_is_more,
        )
        ranks[:, phase_index] = phase_ranks
        percentiles[:, phase_index] = phase_percentiles
        scores[:, phase_index] = phase_scores

        ordered = np.argsort(phase_ranks, kind="mergesort")
        top_idx = ordered[:5].astype(int).tolist()
        bottom_idx = ordered[-5:][::-1].astype(int).tolist()
        phase_results.append(
            SingularMetricPhaseResult(
                phase=phase,
                top=[(float(phase_scores[i]), baseline.keys[i]) for i in top_idx],
                bottom=[
                    (float(phase_scores[i]), baseline.keys[i]) for i in bottom_idx
                ],
            )
        )

    _save_ordering(
        spec,
        baseline,
        values,
        ranks,
        percentiles,
        scores,
        output_dir,
    )
    return SingularMetricResult(
        name=spec.name,
        feature=spec.feature,
        n_identities=n_identities,
        phase_results=tuple(phase_results),
    )


def run_all_singular_metrics(
    smoothed_levels: dict[IdentityType, LevelRows] | None = None,
    *,
    output_dir: Path = SINGULAR_METRIC_CACHE_DIR,
    specs: tuple[SingularMetricSpec, ...] = SINGULAR_METRICS,
) -> list[SingularMetricResult]:
    if not specs:
        return []
    if smoothed_levels is None:
        cfg = EmbeddingConfig()
        smoothed_levels = apply_hierarchical_shrinkage(load_all(cfg), cfg)

    feature_set = tuple(dict.fromkeys(spec.feature for spec in specs))
    matrices = build_all_matrices(
        smoothed_levels,
        EmbeddingConfig(feature_set=feature_set),
    )
    baseline = matrices[IdentityType.BASELINE]
    feature_index = {feature: i for i, feature in enumerate(baseline.feature_names)}
    return [
        run_singular_metric(
            spec,
            baseline,
            feature_index,
            output_dir=output_dir,
        )
        for spec in specs
    ]


def log_singular_metric_results(results: list[SingularMetricResult]) -> None:
    for result in results:
        logger.info(
            "[%s] feature=%s identities=%d",
            result.name,
            result.feature,
            result.n_identities,
        )
        for phase in result.phase_results:
            logger.info(
                "    %s top=%s bottom=%s",
                phase.phase,
                phase.top[:3],
                phase.bottom[:3],
            )


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.INFO)
    log_singular_metric_results(run_all_singular_metrics())


if __name__ == "__main__":
    main()
