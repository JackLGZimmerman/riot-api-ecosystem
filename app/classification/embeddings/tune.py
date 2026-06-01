"""Lean sweep harness for the specialists.

One row per candidate. Filters out candidates outside the target group-count
band. Caches raw non-temporal levels on disk so repeat runs are fast.

Run:
    uv run python -m app.classification.embeddings.tune
    uv run python -m app.classification.embeddings.tune --name durability
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    SPECIALISTS,
    EmbeddingConfig,
    IdentityType,
    SpecialistSpec,
)
from app.classification.embeddings.load import LevelRows, load_all
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.similarity import median_pair_similarity
from app.core.logging.logger import setup_logging_config
from app.core.utils.smoothing import apply_hierarchical_shrinkage

OUTPUT_DIR = Path("/tmp/embed_exp")
RAW_CACHE_PATH = OUTPUT_DIR / "raw_levels_non_temporal.pkl"


def load_raw_cached() -> dict[IdentityType, LevelRows]:
    if RAW_CACHE_PATH.exists():
        with RAW_CACHE_PATH.open("rb") as f:
            return pickle.load(f)
    RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    levels = load_all(EmbeddingConfig())
    with RAW_CACHE_PATH.open("wb") as f:
        pickle.dump(levels, f)
    return levels


@dataclass(frozen=True)
class SpecRow:
    spec: str
    kv: float
    t: float
    kept: int
    coverage: float
    largest_share: float
    median_within: float

    def fmt(self) -> str:
        return (
            f"{self.spec:<14} kv={self.kv:.3f} t={self.t:.2f} | "
            f"g={self.kept:>3d} cov={self.coverage:>4.2f} "
            f"lg%={self.largest_share:>4.2f} med={self.median_within:>4.2f}"
        )


@dataclass(frozen=True)
class _SweepContext:
    sim: np.ndarray
    linkage_matrix: np.ndarray | None
    n: int


def _sort_groups(groups: list[list[int]]) -> list[list[int]]:
    return sorted(
        (sorted(group) for group in groups),
        key=lambda group: (-len(group), group[0]),
    )


def _sweep_context(embeddings: np.ndarray) -> _SweepContext:
    sim = embeddings @ embeddings.T
    if embeddings.shape[0] < 2:
        return _SweepContext(
            sim=sim,
            linkage_matrix=None,
            n=embeddings.shape[0],
        )
    distance = 1.0 - sim
    distance = np.clip(distance, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    return _SweepContext(
        sim=sim,
        linkage_matrix=linkage(squareform(distance, checks=False), method="average"),
        n=embeddings.shape[0],
    )


def _groups_for_threshold(
    context: _SweepContext,
    threshold: float,
) -> list[list[int]]:
    if context.linkage_matrix is None or threshold > 1.0:
        return [[i] for i in range(context.n)]
    clusters = fcluster(
        context.linkage_matrix,
        t=1.0 - threshold,
        criterion="distance",
    )
    groups = [
        np.flatnonzero(clusters == cluster).astype(int).tolist()
        for cluster in np.unique(clusters)
    ]
    return _sort_groups(groups)


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


def sweep_specialist(
    spec: SpecialistSpec,
    smoothed_default: dict[IdentityType, LevelRows],
    kvs: tuple[float, ...],
    ts: tuple[float, ...],
    target: tuple[int, int],
    min_median_override: float | None = None,
) -> list[SpecRow]:
    floor = spec.min_median_sim if min_median_override is None else min_median_override
    matrices = build_all_matrices(
        smoothed_default, EmbeddingConfig(feature_set=spec.feature_set)
    )
    baseline_matrix = matrices[IdentityType.BASELINE]

    rows: list[SpecRow] = []
    for kv in kvs:
        baseline = embed.embed_level(
            baseline_matrix,
            EmbeddingConfig(feature_set=spec.feature_set, projection_keep_variance=kv),
        )
        context = _sweep_context(baseline.embeddings)
        for t in ts:
            kept, _ = _split_by_coherence(
                context.sim,
                _groups_for_threshold(context, t),
                floor,
            )
            sizes = [len(group) for group in kept]
            medians = [median_pair_similarity(context.sim, group) for group in kept]
            n = baseline.embeddings.shape[0]
            largest = max(sizes, default=0)
            rows.append(
                SpecRow(
                    spec=spec.name,
                    kv=kv,
                    t=t,
                    kept=len(kept),
                    coverage=sum(sizes) / n if n else 0.0,
                    largest_share=largest / max(n, 1),
                    median_within=(
                        float(np.median(medians)) if medians else float("nan")
                    ),
                )
            )

    lo, hi = target

    def key(r: SpecRow) -> tuple:
        in_band = lo <= r.kept <= hi
        ok = r.coverage >= 0.85 and r.largest_share <= 0.30
        midpoint = (lo + hi) / 2
        spread = abs(midpoint - r.kept)
        return (in_band, ok, r.median_within, -spread)

    rows.sort(key=key, reverse=True)
    return rows


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_specialists(args: argparse.Namespace) -> None:
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())

    if args.name:
        names = set(args.name)
        target_specs = [s for s in SPECIALISTS if s.name in names]
        if not target_specs:
            raise SystemExit(f"no specialists matched {args.name}")
    else:
        target_specs = list(SPECIALISTS)

    target = (args.lo, args.hi)
    summary: list[str] = []
    out_lines: list[str] = []
    for spec in target_specs:
        rows = sweep_specialist(
            spec,
            smoothed,
            tuple(args.kvs),
            tuple(args.ts),
            target,
            min_median_override=args.min_median,
        )
        in_band = [r for r in rows if target[0] <= r.kept <= target[1]]
        keep = in_band[:5] if in_band else rows[:5]
        out_lines.append(
            f"# {spec.name} (curr kv={spec.projection_keep_variance:.3f} "
            f"t={spec.similarity_threshold:.2f})"
        )
        out_lines.extend(r.fmt() for r in keep)
        out_lines.append("")
        if in_band:
            best = in_band[0]
            summary.append(
                f"PICK {spec.name}: kv={best.kv:.3f} t={best.t:.2f} "
                f"g={best.kept} cov={best.coverage:.2f} "
                f"lg%={best.largest_share:.2f} med={best.median_within:.2f}"
            )
        else:
            summary.append(f"NO PICK {spec.name} (in-band empty)")

    full = out_lines + ["# summary"] + summary
    for line in full:
        print(line)
    _write_lines(OUTPUT_DIR / "tune_specialists.txt", full)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", nargs="*")
    parser.add_argument("--lo", type=int, default=2)
    parser.add_argument("--hi", type=int, default=14)
    parser.add_argument("--kvs", type=float, nargs="*", default=(0.6, 0.7, 0.8, 0.9))
    parser.add_argument(
        "--ts",
        type=float,
        nargs="*",
        default=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
    )
    parser.add_argument("--min-median", type=float)
    return parser


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.INFO)
    run_specialists(_parser().parse_args())


if __name__ == "__main__":
    main()
