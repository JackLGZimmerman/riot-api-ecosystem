"""Inspect specialist embedding group semantics.

Example:
    uv run python -m app.classification.embeddings.inspection.base \
        --name <specialist> --kv <kv> --t <threshold>
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    SPECIALISTS,
    EmbeddingConfig,
    IdentityType,
    SpecialistSpec,
)
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.report import _load_champion_names
from app.classification.embeddings.similarity import median_pair_similarity
from app.classification.embeddings.specialists import group_specialist
from app.classification.embeddings.tune import load_raw_cached
from app.core.utils.smoothing import apply_hierarchical_shrinkage


def specialist_spec(name: str) -> SpecialistSpec:
    for spec in SPECIALISTS:
        if spec.name == name:
            return spec
    options = ", ".join(spec.name for spec in SPECIALISTS)
    raise ValueError(f"No active specialist named {name!r}. Available: {options}")


def inspect_specialist(
    *,
    name: str,
    features: tuple[str, ...] | None = None,
    keep_variance: float | None = None,
    threshold: float | None = None,
    min_median_sim: float | None = None,
    champion_limit: int = 10,
) -> None:
    spec = specialist_spec(name)
    resolved_features = spec.feature_set if features is None else features
    kv = spec.projection_keep_variance if keep_variance is None else keep_variance
    t = spec.similarity_threshold if threshold is None else threshold
    min_median = spec.min_median_sim if min_median_sim is None else min_median_sim

    cfg = EmbeddingConfig(feature_set=resolved_features, projection_keep_variance=kv)
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    matrix = build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]
    baseline = embed.embed_level(matrix, cfg)
    grouping = group_specialist(
        baseline,
        SpecialistSpec(
            name=spec.name,
            feature_set=resolved_features,
            similarity_threshold=t,
            projection_keep_variance=kv,
            min_median_sim=min_median,
        ),
    )
    sim = grouping.sim
    kept = grouping.kept
    dropped = grouping.dropped

    names = _load_champion_names()
    columns = {column: i for i, column in enumerate(baseline.key_columns)}
    x = matrix.matrix
    mu = x.mean(axis=0)
    sd = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)

    print(
        f"{name} kv={kv:.2f} t={t:.2f} "
        f"min_median={min_median:.2f} kept={len(kept)} "
        f"coverage={sum(len(g) for g in kept) / baseline.embeddings.shape[0]:.3f} "
        f"dropped={len(dropped)} largest={max(map(len, kept), default=0)}"
    )
    _print_pca_axes(x, matrix.feature_names, kv)

    for gid, group in enumerate(sorted(kept, key=len, reverse=True), start=1):
        arr = np.asarray(group, dtype=np.int64)
        z = (x[arr].mean(axis=0) - mu) / sd
        ranked_z = sorted(
            zip(resolved_features, z, strict=True),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        builds = Counter(str(baseline.keys[i][columns["build"]]) for i in group)
        roles = Counter(str(baseline.keys[i][columns["teamposition"]]) for i in group)
        champions = Counter(
            names.get(
                int(baseline.keys[i][columns["championid"]]),
                str(baseline.keys[i][columns["championid"]]),
            )
            for i in group
        )
        print(
            f"G{gid:02d} size={len(group)} "
            f"med={median_pair_similarity(sim, group):.3f}"
        )
        print("  z      " + ", ".join(f"{k}={v:+.2f}" for k, v in ranked_z))
        print("  builds " + ", ".join(f"{k}:{v}" for k, v in builds.most_common(6)))
        print("  roles  " + ", ".join(f"{k}:{v}" for k, v in roles.most_common(5)))
        print(
            "  champs "
            + ", ".join(f"{k}:{v}" for k, v in champions.most_common(champion_limit))
        )
        print()


def _print_pca_axes(x: np.ndarray, feature_names: tuple[str, ...], keep: float) -> None:
    _, _, eigenvectors, n_axes, ratios = embed.fit_pca_basis(x, keep)
    if n_axes <= 0:
        print("PCA axes: no variance")
        return

    print(
        "PCA axes: "
        f"keep={keep:.2f} retained={n_axes} "
        f"ratios={', '.join(f'{ratio:.3f}' for ratio in ratios[:n_axes])}"
    )
    for axis in range(n_axes):
        weights = eigenvectors[:, axis]
        ranked = np.argsort(np.abs(weights))[::-1][:6]
        summary = ", ".join(
            f"{feature_names[i]}={weights[i]:+.2f}" for i in ranked
        )
        print(f"  PC{axis + 1} {ratios[axis]:.3f}: {summary}")
    print()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--kv", type=float)
    parser.add_argument("--t", type=float)
    parser.add_argument("--min-median", type=float)
    parser.add_argument("--champion-limit", type=int, default=10)
    parser.add_argument("--features", nargs="*")
    return parser


def main() -> None:
    args = _parser().parse_args()
    inspect_specialist(
        name=args.name,
        features=tuple(args.features) if args.features else None,
        keep_variance=args.kv,
        threshold=args.t,
        min_median_sim=args.min_median,
        champion_limit=args.champion_limit,
    )


if __name__ == "__main__":
    main()
