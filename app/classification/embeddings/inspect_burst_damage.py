"""Inspect candidate burst_damage specialist group semantics.

Run after a sweep to verify that retained PCA directions and group reads line
up with the metrics being tested:

    uv run python -m app.classification.embeddings.inspect_burst_damage --kv 0.85 --t 0.50
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    EmbeddingConfig,
    IdentityType,
    PHASES,
    SPECIALISTS,
    SpecialistSpec,
)
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.classification.embeddings.report import _load_champion_names
from app.classification.embeddings.similarity import (
    cosine_similarity_matrix,
    group_by_threshold,
)
from app.classification.embeddings.specialists import _split_by_coherence
from app.classification.embeddings.tune import load_raw_cached


def _burst_damage_spec() -> SpecialistSpec:
    for spec in SPECIALISTS:
        if spec.name == "burst_damage":
            return spec
    raise ValueError("No active burst_damage specialist found in SPECIALISTS")


def _print_pca_axes(x: np.ndarray, feature_names: tuple[str, ...], keep: float) -> None:
    centered = x.astype(np.float64, copy=False) - x.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if total <= 0.0:
        print("PCA axes: no variance")
        return

    ratios = eigenvalues / total
    cum = np.cumsum(ratios)
    n_axes = int(np.searchsorted(cum, max(0.0, min(keep, 1.0))) + 1)
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


def main() -> None:
    spec = _burst_damage_spec()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv", type=float, default=spec.projection_keep_variance)
    parser.add_argument("--t", type=float, default=spec.similarity_threshold)
    parser.add_argument("--min-median", type=float, default=spec.min_median_sim)
    parser.add_argument("--features", nargs="*", default=list(spec.feature_set))
    args = parser.parse_args()

    features = tuple(args.features)
    cfg = EmbeddingConfig(feature_set=features, projection_keep_variance=args.kv)
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    matrix = build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]
    baseline = embed.embed_level(matrix, cfg)
    sim = cosine_similarity_matrix(baseline.embeddings)
    raw_groups = group_by_threshold(baseline.embeddings, args.t)
    candidate = SpecialistSpec(
        "burst_damage", features, args.t, args.kv, args.min_median
    )
    kept, dropped = _split_by_coherence(
        sim, raw_groups, candidate.min_median_sim, candidate.min_group_size
    )

    names = _load_champion_names()
    columns = {name: i for i, name in enumerate(baseline.key_columns)}
    x = matrix.matrix.reshape(matrix.matrix.shape[0], -1)
    mu = x.mean(axis=0)
    sd = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)

    print(
        f"burst_damage kv={args.kv:.2f} t={args.t:.2f} "
        f"min_median={args.min_median:.2f} kept={len(kept)} "
        f"coverage={sum(len(g) for g in kept) / baseline.embeddings.shape[0]:.3f} "
        f"dropped={len(dropped)} largest={max(map(len, kept), default=0)}"
    )
    _print_pca_axes(x, matrix.feature_names, args.kv)

    for gid, group in enumerate(sorted(kept, key=len, reverse=True), start=1):
        arr = np.asarray(group, dtype=np.int64)
        z = (
            ((x[arr].mean(axis=0) - mu) / sd)
            .reshape(len(PHASES), len(features))
            .mean(axis=0)
        )
        ranked_z = sorted(zip(features, z), key=lambda p: abs(p[1]), reverse=True)
        builds = Counter(str(baseline.keys[i][columns["build"]]) for i in group)
        roles = Counter(str(baseline.keys[i][columns["teamposition"]]) for i in group)
        champions = Counter(
            names.get(
                int(baseline.keys[i][columns["championid"]]),
                str(baseline.keys[i][columns["championid"]]),
            )
            for i in group
        )
        median = 1.0
        if len(group) > 1:
            iu, ju = np.triu_indices(arr.size, k=1)
            median = float(np.median(sim[arr[iu], arr[ju]]))
        print(f"G{gid:02d} size={len(group)} med={median:.3f}")
        print("  z      " + ", ".join(f"{k}={v:+.2f}" for k, v in ranked_z))
        print("  builds " + ", ".join(f"{k}:{v}" for k, v in builds.most_common(6)))
        print("  roles  " + ", ".join(f"{k}:{v}" for k, v in roles.most_common(5)))
        print(
            "  champs "
            + ", ".join(f"{k}:{v}" for k, v in champions.most_common(10))
        )
        print()


if __name__ == "__main__":
    main()
