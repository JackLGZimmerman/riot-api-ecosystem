"""Specialist embeddings: narrow single-question groupings on top of the base.

Each `SpecialistSpec` asks one behavioural question over a small feature subset
with its own similarity threshold. Specialists emit per-spec group labels for
each identity in each temporal bin; the global base partition is unchanged.

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
    PHASES,
    SPECIALIST_CACHE_DIR,
    SPECIALISTS,
    EmbeddingConfig,
    IdentityType,
    SpecialistSpec,
)
from app.classification.embeddings.embed import LevelEmbeddings
from app.classification.embeddings.load import LevelRows, load_all
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.classification.embeddings.similarity import (
    cosine_similarity_matrix,
    group_by_threshold,
    median_pair_similarity,
)
from app.core.logging.logger import setup_logging_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseGrouping:
    phase: str
    phase_index: int
    sim: np.ndarray
    kept: list[list[int]]
    dropped: list[list[int]]


@dataclass(frozen=True)
class SpecialistPhaseResult:
    phase: str
    n_kept_groups: int
    n_dropped_groups: int
    coverage: float
    largest_group: int
    median_within_sim: float
    top_groups: list[tuple[int, float, list[tuple]]]


@dataclass(frozen=True)
class SpecialistResult:
    name: str
    n_identities: int
    n_kept_groups: int
    n_dropped_groups: int
    coverage: float
    largest_group: int
    median_within_sim: float
    top_groups: list[tuple[str, int, float, list[tuple]]]
    phase_results: tuple[SpecialistPhaseResult, ...]


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
    groupings: list[PhaseGrouping],
    output_dir: Path,
) -> None:
    if embeddings.embeddings.ndim != 3:
        raise ValueError(
            f"temporal labels require 3-D embeddings, got {embeddings.embeddings.shape}"
        )
    n_identities, n_phases = embeddings.embeddings.shape[:2]
    labels = np.full((n_identities, n_phases), -1, dtype=np.int32)
    n_groups = np.zeros(n_phases, dtype=np.int32)
    phases = [""] * n_phases
    for grouping in groupings:
        phases[grouping.phase_index] = grouping.phase
        n_groups[grouping.phase_index] = len(grouping.kept)
        for gid, members in enumerate(grouping.kept):
            for idx in members:
                labels[idx, grouping.phase_index] = gid
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{name}.npz",
        keys=np.array(embeddings.keys, dtype=object),
        key_columns=np.array(embeddings.key_columns, dtype=object),
        phases=np.array(phases, dtype=object),
        label_axes=np.array(("identity", "phase"), dtype=object),
        labels=labels,
        n_groups=n_groups,
    )


def group_specialist_by_phase(
    embeddings: LevelEmbeddings,
    spec: SpecialistSpec,
) -> list[PhaseGrouping]:
    """Cluster each temporal bin independently in a shared latent space."""
    if embeddings.embeddings.ndim != 3:
        raise ValueError(
            f"expected temporal embeddings shaped (n, phases, d), got {embeddings.embeddings.shape}"
        )
    n_phases = embeddings.embeddings.shape[1]
    phase_names = PHASES[:n_phases]
    if len(phase_names) != n_phases:
        phase_names = tuple(f"phase_{i}" for i in range(n_phases))

    groupings: list[PhaseGrouping] = []
    for phase_index, phase in enumerate(phase_names):
        phase_embeddings = embeddings.embeddings[:, phase_index, :]
        sim = cosine_similarity_matrix(phase_embeddings)
        raw_groups = group_by_threshold(phase_embeddings, spec.similarity_threshold)
        kept, dropped = _split_by_coherence(
            sim, raw_groups, spec.min_median_sim
        )
        groupings.append(
            PhaseGrouping(
                phase=phase,
                phase_index=phase_index,
                sim=sim,
                kept=kept,
                dropped=dropped,
            )
        )
    return groupings


def _phase_result(
    grouping: PhaseGrouping,
    baseline: LevelEmbeddings,
) -> SpecialistPhaseResult:
    sizes = [len(g) for g in grouping.kept]
    n = baseline.embeddings.shape[0]
    top = sorted(grouping.kept, key=len, reverse=True)[:5]
    medians = [median_pair_similarity(grouping.sim, g) for g in grouping.kept]
    return SpecialistPhaseResult(
        phase=grouping.phase,
        n_kept_groups=len(grouping.kept),
        n_dropped_groups=len(grouping.dropped),
        coverage=sum(sizes) / n if n else 0.0,
        largest_group=max(sizes, default=0),
        median_within_sim=(
            float(np.median(medians)) if medians else float("nan")
        ),
        top_groups=[
            (len(g), median_pair_similarity(grouping.sim, g), [baseline.keys[i] for i in g[:8]])
            for g in top
        ],
    )


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
    groupings = group_specialist_by_phase(baseline, spec)
    _save_labels(spec.name, baseline, groupings, output_dir)

    phase_results = tuple(_phase_result(grouping, baseline) for grouping in groupings)
    sizes = [len(g) for grouping in groupings for g in grouping.kept]
    medians = [
        median_pair_similarity(grouping.sim, group)
        for grouping in groupings
        for group in grouping.kept
    ]
    n = baseline.embeddings.shape[0]
    total_slots = n * len(groupings)
    top = sorted(
        (
            (grouping.phase, group, grouping.sim)
            for grouping in groupings
            for group in grouping.kept
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )[:5]
    return SpecialistResult(
        name=spec.name,
        n_identities=n,
        n_kept_groups=sum(len(grouping.kept) for grouping in groupings),
        n_dropped_groups=sum(len(grouping.dropped) for grouping in groupings),
        coverage=sum(sizes) / total_slots if total_slots else 0.0,
        largest_group=max(sizes, default=0),
        median_within_sim=(
            float(np.median(medians))
            if medians
            else float("nan")
        ),
        top_groups=[
            (
                phase,
                len(group),
                median_pair_similarity(sim, group),
                [baseline.keys[i] for i in group[:8]],
            )
            for phase, group, sim in top
        ],
        phase_results=phase_results,
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
            "[%s] phase_groups=%d dropped=%d coverage=%.2f largest=%d median_within=%.3f",
            r.name,
            r.n_kept_groups,
            r.n_dropped_groups,
            r.coverage,
            r.largest_group,
            r.median_within_sim,
        )
        for phase in r.phase_results:
            logger.info(
                "    %s kept=%d coverage=%.2f largest=%d median=%.3f",
                phase.phase,
                phase.n_kept_groups,
                phase.coverage,
                phase.largest_group,
                phase.median_within_sim,
            )
        for phase, size, median, members in r.top_groups:
            logger.info(
                "    %s size=%d median=%.3f e.g. %s",
                phase,
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
