"""Inspect specialist embedding group semantics.

Example:
    uv run python -m app.classification.embeddings.inspection.base \
        --name <specialist> --kv <kv> --t <threshold>

Replacement metric checks:
    uv run python -m app.classification.embeddings.inspection.base \
        --name <specialist> \
        --compare-features <raw_feature> <transformed_feature> \
        --denominator-check <numerator_feature> <denominator_feature>
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import NamedTuple

import numpy as np

from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    PHASES,
    SPECIALISTS,
    EmbeddingConfig,
    IdentityType,
    SpecialistSpec,
)
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.classification.embeddings.report import _load_champion_names
from app.classification.embeddings.similarity import median_pair_similarity
from app.classification.embeddings.specialists import group_specialist_by_phase
from app.classification.embeddings.tune import load_raw_cached


class FeaturePair(NamedTuple):
    left: str
    right: str


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
    phase: str = PHASES[0],
    champion_limit: int = 10,
) -> None:
    spec = specialist_spec(name)
    resolved_features = spec.feature_set if features is None else features
    kv = spec.projection_keep_variance if keep_variance is None else keep_variance
    t = spec.similarity_threshold if threshold is None else threshold
    min_median = spec.min_median_sim if min_median_sim is None else min_median_sim
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

    cfg = EmbeddingConfig(feature_set=resolved_features, projection_keep_variance=kv)
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    matrix = build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]
    baseline = embed.embed_level(matrix, cfg)
    grouping_spec = SpecialistSpec(
        name=spec.name,
        feature_set=resolved_features,
        similarity_threshold=t,
        projection_keep_variance=kv,
        min_median_sim=min_median,
    )
    groupings = group_specialist_by_phase(baseline, grouping_spec)
    grouping = next(grouping for grouping in groupings if grouping.phase == phase)
    sim = grouping.sim
    kept = grouping.kept
    dropped = grouping.dropped

    names = _load_champion_names()
    columns = {column: i for i, column in enumerate(baseline.key_columns)}
    all_phase_x = matrix.matrix.reshape(-1, matrix.matrix.shape[-1])
    x = matrix.matrix[:, grouping.phase_index, :]
    mu = x.mean(axis=0)
    sd = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)

    print(
        f"{name} phase={phase} kv={kv:.2f} t={t:.2f} "
        f"min_median={min_median:.2f} kept={len(kept)} "
        f"coverage={sum(len(g) for g in kept) / baseline.embeddings.shape[0]:.3f} "
        f"dropped={len(dropped)} largest={max(map(len, kept), default=0)}"
    )
    _print_pca_axes(all_phase_x, matrix.feature_names, kv)

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


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.compare_features or args.denominator_check:
        inspect_replacement_metrics(
            name=args.name,
            features=tuple(args.features) if args.features else None,
            comparisons=tuple(
                FeaturePair(left, right) for left, right in (args.compare_features or ())
            ),
            denominator_checks=tuple(
                FeaturePair(left, right) for left, right in (args.denominator_check or ())
            ),
            topn=args.topn,
            tail_quantiles=tuple(args.tail_quantiles),
            include_bottom_tails=args.include_bottom_tails,
        )
        return
    inspect_specialist(
        name=args.name,
        features=tuple(args.features) if args.features else None,
        keep_variance=args.kv,
        threshold=args.t,
        min_median_sim=args.min_median,
        phase=args.phase,
        champion_limit=args.champion_limit,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--kv", type=float)
    parser.add_argument("--t", type=float)
    parser.add_argument("--min-median", type=float)
    parser.add_argument("--phase", choices=PHASES, default=PHASES[0])
    parser.add_argument("--champion-limit", type=int, default=10)
    parser.add_argument("--features", nargs="*")
    parser.add_argument(
        "--compare-features",
        nargs=2,
        action="append",
        metavar=("RAW", "TRANSFORMED"),
        help=(
            "Compare raw vs transformed feature rankings using tail correlation "
            "and top/bottom identity-set Jaccard. Repeatable."
        ),
    )
    parser.add_argument(
        "--denominator-check",
        nargs=2,
        action="append",
        metavar=("NUMERATOR", "DENOMINATOR"),
        help=(
            "Print numerator-vs-denominator correlations for ratio sanity checks. "
            "Repeatable."
        ),
    )
    parser.add_argument("--topn", type=int, default=50)
    parser.add_argument(
        "--tail-quantiles",
        type=float,
        nargs="*",
        default=(0.90, 0.95),
        help="Raw-feature quantiles used for top-tail correlation checks.",
    )
    parser.add_argument(
        "--include-bottom-tails",
        action="store_true",
        help="Also run bottom-tail correlation checks at 1-q for each tail quantile.",
    )
    return parser


def inspect_replacement_metrics(
    *,
    name: str,
    features: tuple[str, ...] | None,
    comparisons: tuple[FeaturePair, ...],
    denominator_checks: tuple[FeaturePair, ...],
    topn: int,
    tail_quantiles: tuple[float, ...],
    include_bottom_tails: bool,
) -> None:
    spec = specialist_spec(name)
    requested_features = _dedupe_preserve_order(
        (
            *(features or spec.feature_set),
            *(feature for pair in comparisons for feature in pair),
            *(feature for pair in denominator_checks for feature in pair),
        )
    )
    if not requested_features:
        raise ValueError("replacement checks need at least one feature")
    if topn <= 0:
        raise ValueError("--topn must be positive")
    if not tail_quantiles:
        raise ValueError("--tail-quantiles must include at least one value")
    if any(q <= 0.0 or q >= 1.0 for q in tail_quantiles):
        raise ValueError("--tail-quantiles must be between 0 and 1")

    cfg = EmbeddingConfig(feature_set=requested_features)
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    matrix = build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]
    feature_index = {feature: i for i, feature in enumerate(matrix.feature_names)}

    print(f"{name} replacement-metric checks")
    print(
        "thresholds: tail Spearman >= 0.90, tail Pearson >= 0.80; "
        f"mean top{topn} Jaccard >= 0.70"
    )
    if comparisons:
        print()
        print("Raw-vs-transformed comparisons")
    for pair in comparisons:
        _print_replacement_comparison(
            matrix,
            feature_index,
            pair,
            topn=topn,
            tail_quantiles=tail_quantiles,
            include_bottom_tails=include_bottom_tails,
        )
    if denominator_checks:
        print()
        print("Denominator sanity checks")
    for pair in denominator_checks:
        _print_denominator_check(matrix, feature_index, pair)


def _print_replacement_comparison(
    matrix,
    feature_index: dict[str, int],
    pair: FeaturePair,
    *,
    topn: int,
    tail_quantiles: tuple[float, ...],
    include_bottom_tails: bool,
) -> None:
    print(f"\n{pair.left} -> {pair.right}")
    tail_rows: list[tuple[float, float, float]] = []
    bottom_tail_rows: list[tuple[float, float, float]] = []
    top_jaccards: list[float] = []
    bottom_jaccards: list[float] = []
    for phase_index, phase in enumerate(PHASES):
        left = _phase_feature(matrix, feature_index, pair.left, phase_index)
        right = _phase_feature(matrix, feature_index, pair.right, phase_index)
        top_left, bottom_left = _top_bottom_indices(left, topn)
        top_right, bottom_right = _top_bottom_indices(right, topn)
        top_j = _jaccard(top_left, top_right)
        bottom_j = _jaccard(bottom_left, bottom_right)
        top_jaccards.append(top_j)
        bottom_jaccards.append(bottom_j)
        print(
            f"  {phase:<8} top{topn}_j={top_j:.3f} "
            f"bottom{topn}_j={bottom_j:.3f}"
        )
        for quantile in tail_quantiles:
            mask = left >= np.quantile(left, quantile)
            pearson = _correlation(left[mask], right[mask], spearman=False)
            spearman = _correlation(left[mask], right[mask], spearman=True)
            tail_rows.append((quantile, pearson, spearman))
            print(
                f"    top q={quantile:.2f} n={int(mask.sum())} "
                f"pearson={pearson:+.3f} spearman={spearman:+.3f}"
            )
            if include_bottom_tails:
                bottom_q = 1.0 - quantile
                bottom_mask = left <= np.quantile(left, bottom_q)
                bottom_pearson = _correlation(
                    left[bottom_mask], right[bottom_mask], spearman=False
                )
                bottom_spearman = _correlation(
                    left[bottom_mask], right[bottom_mask], spearman=True
                )
                bottom_tail_rows.append((bottom_q, bottom_pearson, bottom_spearman))
                print(
                    f"    bot q={bottom_q:.2f} n={int(bottom_mask.sum())} "
                    f"pearson={bottom_pearson:+.3f} "
                    f"spearman={bottom_spearman:+.3f}"
                )
    _print_replacement_summary(
        top_jaccards=top_jaccards,
        bottom_jaccards=bottom_jaccards,
        tail_rows=tail_rows,
        bottom_tail_rows=bottom_tail_rows,
        topn=topn,
    )


def _print_replacement_summary(
    *,
    top_jaccards: list[float],
    bottom_jaccards: list[float],
    tail_rows: list[tuple[float, float, float]],
    bottom_tail_rows: list[tuple[float, float, float]],
    topn: int,
) -> None:
    tail_pearsons = [row[1] for row in tail_rows]
    tail_spearmans = [row[2] for row in tail_rows]
    print(
        f"  summary mean_top{topn}_j={np.mean(top_jaccards):.3f} "
        f"mean_bottom{topn}_j={np.mean(bottom_jaccards):.3f} "
        f"min_tail_pearson={np.nanmin(tail_pearsons):+.3f} "
        f"min_tail_spearman={np.nanmin(tail_spearmans):+.3f}"
    )
    if bottom_tail_rows:
        bottom_pearsons = [row[1] for row in bottom_tail_rows]
        bottom_spearmans = [row[2] for row in bottom_tail_rows]
        print(
            f"  bottom-tail summary min_pearson={np.nanmin(bottom_pearsons):+.3f} "
            f"min_spearman={np.nanmin(bottom_spearmans):+.3f}"
        )


def _print_denominator_check(
    matrix,
    feature_index: dict[str, int],
    pair: FeaturePair,
) -> None:
    pearsons: list[float] = []
    spearmans: list[float] = []
    print(f"\n{pair.left} vs {pair.right}")
    for phase_index, phase in enumerate(PHASES):
        left = _phase_feature(matrix, feature_index, pair.left, phase_index)
        right = _phase_feature(matrix, feature_index, pair.right, phase_index)
        pearson = _correlation(left, right, spearman=False)
        spearman = _correlation(left, right, spearman=True)
        pearsons.append(pearson)
        spearmans.append(spearman)
        print(f"  {phase:<8} pearson={pearson:+.3f} spearman={spearman:+.3f}")
    print(
        f"  summary mean_pearson={np.nanmean(pearsons):+.3f} "
        f"mean_spearman={np.nanmean(spearmans):+.3f}"
    )


def _phase_feature(
    matrix,
    feature_index: dict[str, int],
    feature: str,
    phase_index: int,
) -> np.ndarray:
    if feature not in feature_index:
        available = ", ".join(feature_index)
        raise ValueError(f"feature {feature!r} not available; loaded: {available}")
    return matrix.matrix[:, phase_index, feature_index[feature]]


def _dedupe_preserve_order(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _top_bottom_indices(values: np.ndarray, n: int) -> tuple[set[int], set[int]]:
    if n > values.shape[0]:
        raise ValueError(f"--topn={n} exceeds identity count {values.shape[0]}")
    order = np.argsort(values, kind="mergesort")
    return set(order[-n:].tolist()), set(order[:n].tolist())


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else float("nan")


def _correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    spearman: bool,
) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask].astype(np.float64)
    right = right[mask].astype(np.float64)
    if left.shape[0] < 3:
        return float("nan")
    if spearman:
        left = _rankdata(left)
        right = _rankdata(right)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


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


if __name__ == "__main__":
    main()
