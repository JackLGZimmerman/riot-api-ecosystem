"""Specialist embeddings: narrow single-question groupings on top of the base.

Each `SpecialistSpec` asks one behavioural question over a small feature subset
with its own similarity threshold. Specialists emit one group label per
identity; the global base partition is unchanged.

Run:
    uv run python -m app.classification.embeddings.specialists
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    SPECIALIST_CACHE_DIR,
    SPECIALISTS,
    EmbeddingConfig,
    IdentityType,
    SpecialistSpec,
)
from app.classification.embeddings.embed import LevelEmbeddings
from app.classification.embeddings.load import LevelRows, load_all
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.similarity import (
    cosine_similarity_matrix,
    group_by_threshold,
    median_pair_similarity,
)
from app.core.logging.logger import setup_logging_config
from app.core.utils.smoothing import apply_hierarchical_shrinkage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Grouping:
    sim: np.ndarray
    kept: list[list[int]]
    dropped: list[list[int]]


@dataclass(frozen=True)
class SpecialistResult:
    name: str
    n_identities: int
    n_kept_groups: int
    n_dropped_groups: int
    coverage: float
    largest_group: int
    median_within_sim: float
    top_groups: list[tuple[int, float, list[tuple]]]


def _split_by_coherence(
    sim: np.ndarray,
    groups: list[list[int]],
    min_median_sim: float,
) -> tuple[list[list[int]], list[list[int]]]:
    kept: list[list[int]] = []
    dropped: list[list[int]] = []
    for group in groups:
        median = median_pair_similarity(sim, group)
        (kept if median >= min_median_sim else dropped).append(group)
    return kept, dropped


def _save_labels(
    name: str,
    embeddings: LevelEmbeddings,
    grouping: Grouping,
    output_dir: Path,
) -> None:
    if embeddings.embeddings.ndim != 2:
        raise ValueError(
            f"identity labels require 2-D embeddings, got {embeddings.embeddings.shape}"
        )
    labels = np.full(embeddings.embeddings.shape[0], -1, dtype=np.int32)
    for gid, members in enumerate(grouping.kept):
        for idx in members:
            labels[idx] = gid
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{name}.npz",
        keys=np.array(embeddings.keys, dtype=object),
        key_columns=np.array(embeddings.key_columns, dtype=object),
        label_axes=np.array(("identity",), dtype=object),
        labels=labels,
        n_groups=np.array(len(grouping.kept), dtype=np.int32),
    )


def group_specialist(
    embeddings: LevelEmbeddings,
    spec: SpecialistSpec,
) -> Grouping:
    """Cluster identities in the specialist latent space."""
    if embeddings.embeddings.ndim != 2:
        raise ValueError(
            f"expected embeddings shaped (n, d), got {embeddings.embeddings.shape}"
        )
    sim = cosine_similarity_matrix(embeddings.embeddings)
    raw_groups = group_by_threshold(embeddings.embeddings, spec.similarity_threshold)
    kept, dropped = _split_by_coherence(sim, raw_groups, spec.min_median_sim)
    return Grouping(sim=sim, kept=kept, dropped=dropped)


def run_specialist(
    spec: SpecialistSpec,
    smoothed_levels: dict[IdentityType, LevelRows],
    *,
    output_dir: Path = SPECIALIST_CACHE_DIR,
) -> SpecialistResult:
    cfg = EmbeddingConfig(
        feature_set=spec.feature_set,
        similarity_threshold=spec.similarity_threshold,
        projection_keep_variance=spec.projection_keep_variance,
    )
    matrices = build_all_matrices(smoothed_levels, cfg)
    baseline = embed.embed_all(matrices, cfg)[IdentityType.BASELINE]
    grouping = group_specialist(baseline, spec)
    _save_labels(spec.name, baseline, grouping, output_dir)

    sizes = [len(g) for g in grouping.kept]
    medians = [median_pair_similarity(grouping.sim, group) for group in grouping.kept]
    n = baseline.embeddings.shape[0]
    top = sorted(grouping.kept, key=len, reverse=True)[:5]
    return SpecialistResult(
        name=spec.name,
        n_identities=n,
        n_kept_groups=len(grouping.kept),
        n_dropped_groups=len(grouping.dropped),
        coverage=sum(sizes) / n if n else 0.0,
        largest_group=max(sizes, default=0),
        median_within_sim=(
            float(np.median(medians))
            if medians
            else float("nan")
        ),
        top_groups=[
            (
                len(group),
                median_pair_similarity(grouping.sim, group),
                [baseline.keys[i] for i in group[:8]],
            )
            for group in top
        ],
    )


def run_all_specialists(
    smoothed_levels: dict[IdentityType, LevelRows] | None = None,
    *,
    output_dir: Path = SPECIALIST_CACHE_DIR,
) -> list[SpecialistResult]:
    if smoothed_levels is None:
        cfg = EmbeddingConfig()
        smoothed_levels = apply_hierarchical_shrinkage(load_all(cfg), cfg)
    results: list[SpecialistResult] = []
    for spec in SPECIALISTS:
        if not spec.feature_set:
            logger.warning("Skipping %s: no feature_set configured", spec.name)
            continue
        results.append(run_specialist(spec, smoothed_levels, output_dir=output_dir))
    return results


def log_results(results: list[SpecialistResult]) -> None:
    for r in results:
        logger.info(
            "[%s] groups=%d dropped=%d coverage=%.2f largest=%d median_within=%.3f",
            r.name,
            r.n_kept_groups,
            r.n_dropped_groups,
            r.coverage,
            r.largest_group,
            r.median_within_sim,
        )
        for size, median, members in r.top_groups:
            logger.info(
                "    size=%d median=%.3f e.g. %s",
                size,
                median,
                members[:5],
            )


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.INFO)
    log_results(run_all_specialists())


if __name__ == "__main__":
    main()
