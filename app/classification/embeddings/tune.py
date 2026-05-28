"""Lean sweep harness for the specialists.

One row per candidate. Filters out candidates outside the per-phase target
group-count band. Caches raw 6010 + 9000-9040 levels on disk so repeat runs are
fast.

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

from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    SPECIALISTS,
    EmbeddingConfig,
    IdentityType,
    SpecialistSpec,
)
from app.classification.embeddings.load import LevelRows, load_all
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.classification.embeddings.similarity import median_pair_similarity
from app.classification.embeddings.specialists import group_specialist_by_phase
from app.core.logging.logger import setup_logging_config

OUTPUT_DIR = Path("/tmp/embed_exp")
RAW_CACHE_PATH = OUTPUT_DIR / "raw_levels.pkl"


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
    phase_group_counts: tuple[int, ...]
    coverage: float
    largest_share: float
    median_within: float

    def fmt(self) -> str:
        phase_counts = ",".join(str(count) for count in self.phase_group_counts)
        return (
            f"{self.spec:<14} kv={self.kv:.3f} t={self.t:.2f} | "
            f"g={self.kept:>3d} phase_g=[{phase_counts}] cov={self.coverage:>4.2f} "
            f"lg%={self.largest_share:>4.2f} med={self.median_within:>4.2f}"
        )


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
        for t in ts:
            candidate = SpecialistSpec(
                name=spec.name,
                feature_set=spec.feature_set,
                similarity_threshold=t,
                projection_keep_variance=kv,
                min_median_sim=floor,
            )
            groupings = group_specialist_by_phase(baseline, candidate)
            phase_counts = tuple(len(grouping.kept) for grouping in groupings)
            sizes = [len(g) for grouping in groupings for g in grouping.kept]
            medians = [
                median_pair_similarity(grouping.sim, group)
                for grouping in groupings
                for group in grouping.kept
            ]
            n = baseline.embeddings.shape[0]
            total_slots = n * len(groupings)
            largest = max(sizes, default=0)
            rows.append(
                SpecRow(
                    spec=spec.name,
                    kv=kv,
                    t=t,
                    kept=sum(phase_counts),
                    phase_group_counts=phase_counts,
                    coverage=sum(sizes) / total_slots if total_slots else 0.0,
                    largest_share=largest / max(n, 1),
                    median_within=(
                        float(np.median(medians)) if medians else float("nan")
                    ),
                )
            )

    lo, hi = target

    def key(r: SpecRow) -> tuple:
        in_band = all(lo <= count <= hi for count in r.phase_group_counts)
        ok = r.coverage >= 0.85 and r.largest_share <= 0.30
        midpoint = (lo + hi) / 2
        spread = max((abs(midpoint - count) for count in r.phase_group_counts), default=0)
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
        in_band = [
            r
            for r in rows
            if all(target[0] <= count <= target[1] for count in r.phase_group_counts)
        ]
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
                f"PICK {spec.name}: kv={best.kv:.3f} t={best.t:.2f} g={best.kept} "
                f"phase_g={list(best.phase_group_counts)} "
                f"cov={best.coverage:.2f} lg%={best.largest_share:.2f} "
                f"med={best.median_within:.2f}"
            )
        else:
            summary.append(f"NO PICK {spec.name} (in-band empty)")

    full = out_lines + ["# summary"] + summary
    for line in full:
        print(line)
    _write_lines(OUTPUT_DIR / "tune_specialists.txt", full)


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kvs", type=float, nargs="*", default=[0.90, 0.93, 0.95, 0.97, 0.99]
    )
    parser.add_argument(
        "--ts", type=float, nargs="*", default=[0.80, 0.85, 0.90, 0.93, 0.95]
    )
    parser.add_argument("--lo", type=int, default=6)
    parser.add_argument("--hi", type=int, default=14)
    parser.add_argument("--name", type=str, nargs="*", default=None)
    parser.add_argument("--min-median", type=float, default=None)
    args = parser.parse_args()
    run_specialists(args)


if __name__ == "__main__":
    main()
