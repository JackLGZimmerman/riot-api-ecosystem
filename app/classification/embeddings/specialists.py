"""Specialist embeddings: narrow single-question groupings on top of the base.

Each `SpecialistSpec` asks one behavioural question over a small feature subset
with its own similarity threshold. Specialists emit per-spec group labels for
each identity; the global base partition is unchanged.

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
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.classification.embeddings.similarity import (
    cosine_similarity_matrix,
    group_by_threshold,
)
from app.core.logging.logger import setup_logging_config

logger = logging.getLogger(__name__)


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
    min_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    kept: list[list[int]] = []
    dropped: list[list[int]] = []
    for group in groups:
        if len(group) < min_size:
            dropped.append(group)
            continue
        arr = np.asarray(group, dtype=np.int64)
        iu, ju = np.triu_indices(arr.size, k=1)
        median = float(np.median(sim[arr[iu], arr[ju]]))
        (kept if median >= min_median_sim else dropped).append(group)
    return kept, dropped


def _median_pair_sim(sim: np.ndarray, members: list[int]) -> float:
    if len(members) < 2:
        return 1.0
    arr = np.asarray(members, dtype=np.int64)
    iu, ju = np.triu_indices(arr.size, k=1)
    return float(np.median(sim[arr[iu], arr[ju]]))


def _save_labels(
    name: str,
    embeddings: LevelEmbeddings,
    kept: list[list[int]],
    output_dir: Path,
) -> None:
    labels = np.full(embeddings.embeddings.shape[0], -1, dtype=np.int32)
    for gid, members in enumerate(kept):
        for idx in members:
            labels[idx] = gid
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{name}.npz",
        keys=np.array(embeddings.keys, dtype=object),
        key_columns=np.array(embeddings.key_columns, dtype=object),
        labels=labels,
        n_groups=np.int32(len(kept)),
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
    sim = cosine_similarity_matrix(baseline.embeddings)
    raw_groups = group_by_threshold(baseline.embeddings, spec.similarity_threshold)
    kept, dropped = _split_by_coherence(
        sim, raw_groups, spec.min_median_sim, spec.min_group_size
    )
    _save_labels(spec.name, baseline, kept, output_dir)

    sizes = [len(g) for g in kept]
    n = baseline.embeddings.shape[0]
    top = sorted(kept, key=len, reverse=True)[:5]
    return SpecialistResult(
        name=spec.name,
        n_identities=n,
        n_kept_groups=len(kept),
        n_dropped_groups=len(dropped),
        coverage=sum(sizes) / n if n else 0.0,
        largest_group=max(sizes, default=0),
        median_within_sim=(
            float(np.median([_median_pair_sim(sim, g) for g in kept]))
            if kept
            else float("nan")
        ),
        top_groups=[
            (len(g), _median_pair_sim(sim, g), [baseline.keys[i] for i in g[:8]])
            for g in top
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
    return [
        run_specialist(spec, smoothed_levels, output_dir=output_dir)
        for spec in SPECIALISTS
    ]


def log_results(results: list[SpecialistResult]) -> None:
    for r in results:
        logger.info(
            "[%s] kept=%d dropped=%d coverage=%.2f largest=%d median_within=%.3f",
            r.name,
            r.n_kept_groups,
            r.n_dropped_groups,
            r.coverage,
            r.largest_group,
            r.median_within_sim,
        )
        for size, median, members in r.top_groups:
            logger.info("    size=%d median=%.3f e.g. %s", size, median, members[:5])


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.INFO)
    log_results(run_all_specialists())


if __name__ == "__main__":
    main()
