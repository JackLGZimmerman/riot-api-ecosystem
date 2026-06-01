"""Audit active specialist and singular metric specs.

This is a heavier companion to ``tune.py``. It scores each specialist against
its group budget and prints enough semantic diagnostics to decide whether a
candidate is extracting useful context or just fragmenting.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    SINGULAR_METRICS,
    SPECIALISTS,
    EmbeddingConfig,
    IdentityType,
    SingularMetricSpec,
    SpecialistSpec,
)
from app.classification.embeddings.matrices import LevelMatrix, build_all_matrices
from app.classification.embeddings.report import _load_champion_names
from app.classification.embeddings.similarity import median_pair_similarity
from app.classification.embeddings.singular_metrics import _normalised_ordering
from app.classification.embeddings.specialists import group_specialist
from app.classification.embeddings.tune import (
    _groups_for_threshold,
    _split_by_coherence,
    _sweep_context,
    load_raw_cached,
)
from app.core.utils.smoothing import apply_hierarchical_shrinkage


@dataclass(frozen=True)
class CandidateAudit:
    name: str
    kv: float
    threshold: float
    budget: int
    axes: int
    count: int
    coverage: float
    largest_share: float
    median_within: float
    mean_top_abs_z: float
    weak_groups: int
    small_groups: int
    score: float


@dataclass(frozen=True)
class CurrentAudit:
    name: str
    kind: str
    metrics: int
    budget: int | None
    config: str
    count: int | None
    status: str
    note: str


def _budget(spec: SpecialistSpec) -> int:
    return math.ceil(len(spec.feature_set) * 1.5)


def _specialists_by_name(names: tuple[str, ...] | None) -> list[SpecialistSpec]:
    if not names:
        return list(SPECIALISTS)
    requested = set(names)
    specs = [spec for spec in SPECIALISTS if spec.name in requested]
    missing = sorted(requested - {spec.name for spec in specs})
    if missing:
        raise SystemExit(f"unknown specialist(s): {', '.join(missing)}")
    return specs


def _singulars_by_name(names: tuple[str, ...] | None) -> list[SingularMetricSpec]:
    if not names:
        return list(SINGULAR_METRICS)
    requested = set(names)
    specs = [spec for spec in SINGULAR_METRICS if spec.name in requested]
    missing = sorted(requested - {spec.name for spec in specs})
    if missing:
        raise SystemExit(f"unknown singular metric(s): {', '.join(missing)}")
    return specs


def _matrix_for(
    smoothed,
    feature_set: tuple[str, ...],
    keep_variance: float = 0.95,
) -> LevelMatrix:
    cfg = EmbeddingConfig(
        feature_set=feature_set,
        projection_keep_variance=keep_variance,
    )
    return build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]


def _current_specialist_audit(spec: SpecialistSpec, smoothed) -> CurrentAudit:
    cfg = EmbeddingConfig(
        feature_set=spec.feature_set,
        projection_keep_variance=spec.projection_keep_variance,
    )
    matrix = build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]
    baseline = embed.embed_level(matrix, cfg)
    grouping = group_specialist(baseline, spec)
    count = len(grouping.kept)
    budget = _budget(spec)
    status = "OK" if count <= budget else "OVER"
    return CurrentAudit(
        name=spec.name,
        kind="specialist",
        metrics=len(spec.feature_set),
        budget=budget,
        config=f"kv={spec.projection_keep_variance:.2f},t={spec.similarity_threshold:.2f}",
        count=count,
        status=status,
        note="",
    )


def _current_singular_audit(spec: SingularMetricSpec, smoothed) -> CurrentAudit:
    matrix = _matrix_for(smoothed, (spec.feature,))
    values = matrix.matrix[:, 0]
    return CurrentAudit(
        name=spec.name,
        kind="singular",
        metrics=1,
        budget=None,
        config=f"feature={spec.feature},higher_is_more={spec.higher_is_more}",
        count=None,
        status="OK",
        note=f"unique={int(np.unique(values).size)}",
    )


def current_audit(*, names: tuple[str, ...] | None = None) -> None:
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    rows: list[CurrentAudit] = []
    if names:
        requested = set(names)
        specialist_specs = [spec for spec in SPECIALISTS if spec.name in requested]
        singular_specs = [spec for spec in SINGULAR_METRICS if spec.name in requested]
        found = {spec.name for spec in specialist_specs} | {
            spec.name for spec in singular_specs
        }
        missing = sorted(requested - found)
        if missing:
            raise SystemExit(f"unknown spec(s): {', '.join(missing)}")
    else:
        specialist_specs = list(SPECIALISTS)
        singular_specs = list(SINGULAR_METRICS)
    rows.extend(_current_specialist_audit(spec, smoothed) for spec in specialist_specs)
    rows.extend(_current_singular_audit(spec, smoothed) for spec in singular_specs)
    for row in rows:
        count = "" if row.count is None else f" count={row.count}"
        budget = "" if row.budget is None else f" budget={row.budget}"
        print(
            f"{row.kind:<10} {row.name:<28} metrics={row.metrics:<2}{budget:<11} "
            f"{row.config:<34}{count:<12} {row.status} {row.note}"
        )


def _candidate_groups(
    baseline,
    threshold: float,
    min_median_sim: float,
) -> tuple[list[list[int]], np.ndarray]:
    context = _sweep_context(baseline.embeddings)
    kept, _ = _split_by_coherence(
        context.sim,
        _groups_for_threshold(context, threshold),
        min_median_sim,
    )
    return kept, context.sim


def _candidate_semantics(
    matrix: LevelMatrix,
    groups: list[list[int]],
) -> tuple[float, int, int]:
    top_abs_z: list[float] = []
    weak = 0
    small = 0
    x = matrix.matrix
    mu = x.mean(axis=0)
    sd = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    for group in groups:
        arr = np.asarray(group, dtype=np.int64)
        z = (x[arr].mean(axis=0) - mu) / sd
        max_abs = float(np.max(np.abs(z))) if z.size else 0.0
        top_abs_z.append(max_abs)
        if max_abs < 0.35:
            weak += 1
        if len(group) < 40:
            small += 1
    return (float(np.mean(top_abs_z)) if top_abs_z else 0.0, weak, small)


def _candidate_score(row: CandidateAudit) -> float:
    if row.count > row.budget:
        return -1000.0
    if row.coverage < 0.85:
        return -500.0
    target = max(2.0, row.budget * 0.70)
    count_score = 1.0 - min(abs(row.count - target) / max(target, 1.0), 1.0)
    giant_penalty = max(0.0, row.largest_share - 0.70)
    weak_penalty = row.weak_groups * 0.08
    small_penalty = row.small_groups * 0.03
    z_score = min(row.mean_top_abs_z / 0.9, 1.5)
    median_score = min(max(row.median_within - 0.85, 0.0) / 0.15, 1.0)
    return (
        count_score * 2.0
        + z_score
        + median_score
        + row.coverage
        - giant_penalty
        - weak_penalty
        - small_penalty
    )


def sweep_specialist(
    spec: SpecialistSpec,
    smoothed,
    *,
    kvs: tuple[float, ...],
    thresholds: tuple[float, ...],
    progress: bool = False,
) -> list[CandidateAudit]:
    rows: list[CandidateAudit] = []
    budget = _budget(spec)
    floor = spec.min_median_sim
    for kv in kvs:
        if progress:
            print(f"  kv={kv:.2f}", flush=True)
        cfg = EmbeddingConfig(
            feature_set=spec.feature_set,
            projection_keep_variance=kv,
        )
        matrix = build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]
        baseline = embed.embed_level(matrix, cfg)
        axes = baseline.embeddings.shape[-1]
        for threshold in thresholds:
            groups, sim = _candidate_groups(baseline, threshold, floor)
            sizes = [len(group) for group in groups]
            medians = [median_pair_similarity(sim, group) for group in groups]
            n_identities = baseline.embeddings.shape[0]
            mean_top_abs_z, weak_groups, small_groups = _candidate_semantics(
                matrix,
                groups,
            )
            row = CandidateAudit(
                name=spec.name,
                kv=kv,
                threshold=threshold,
                budget=budget,
                axes=axes,
                count=len(groups),
                coverage=sum(sizes) / max(n_identities, 1),
                largest_share=max(sizes, default=0) / max(n_identities, 1),
                median_within=float(np.median(medians)) if medians else float("nan"),
                mean_top_abs_z=mean_top_abs_z,
                weak_groups=weak_groups,
                small_groups=small_groups,
                score=0.0,
            )
            rows.append(row.__class__(**{**asdict(row), "score": _candidate_score(row)}))
    return sorted(rows, key=lambda row: row.score, reverse=True)


def sweep_audit(
    *,
    names: tuple[str, ...] | None,
    kvs: tuple[float, ...],
    thresholds: tuple[float, ...],
    topn: int,
    output_json: Path | None,
) -> None:
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    payload: dict[str, list[dict]] = {}
    for spec in _specialists_by_name(names):
        print(
            f"# sweeping {spec.name} metrics={len(spec.feature_set)} "
            f"budget={_budget(spec)} current=kv={spec.projection_keep_variance:.2f},"
            f"t={spec.similarity_threshold:.2f}",
            flush=True,
        )
        rows = sweep_specialist(
            spec,
            smoothed,
            kvs=kvs,
            thresholds=thresholds,
            progress=True,
        )
        payload[spec.name] = [asdict(row) for row in rows[:topn]]
        print(
            f"# results {spec.name} metrics={len(spec.feature_set)} "
            f"budget={_budget(spec)} current=kv={spec.projection_keep_variance:.2f},"
            f"t={spec.similarity_threshold:.2f}",
            flush=True,
        )
        for row in rows[:topn]:
            print(
                f"kv={row.kv:.2f} t={row.threshold:.2f} axes={row.axes} "
                f"count={row.count} cov={row.coverage:.2f} "
                f"lg={row.largest_share:.2f} med={row.median_within:.2f} "
                f"z={row.mean_top_abs_z:.2f} weak={row.weak_groups} "
                f"small={row.small_groups} score={row.score:.2f}",
                flush=True,
            )
        print(flush=True)
        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if output_json:
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def singular_audit(*, names: tuple[str, ...] | None, topn: int) -> None:
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    champion_names = _load_champion_names()
    for spec in _singulars_by_name(names):
        matrix = _matrix_for(smoothed, (spec.feature,))
        columns = {column: index for index, column in enumerate(matrix.key_columns)}
        values = matrix.matrix[:, 0]
        _, _, scores = _normalised_ordering(
            values,
            higher_is_more=spec.higher_is_more,
        )
        order = np.argsort(-scores, kind="mergesort")
        top = order[:topn]
        bottom = order[-topn:][::-1]
        top_roles = Counter(
            str(matrix.keys[index][columns["teamposition"]]) for index in top
        )
        top_builds = Counter(str(matrix.keys[index][columns["build"]]) for index in top)
        top_champs = Counter(
            champion_names.get(
                int(matrix.keys[index][columns["championid"]]),
                str(matrix.keys[index][columns["championid"]]),
            )
            for index in top
        )
        bottom_roles = Counter(
            str(matrix.keys[index][columns["teamposition"]]) for index in bottom
        )
        bottom_builds = Counter(
            str(matrix.keys[index][columns["build"]]) for index in bottom
        )
        bottom_champs = Counter(
            champion_names.get(
                int(matrix.keys[index][columns["championid"]]),
                str(matrix.keys[index][columns["championid"]]),
            )
            for index in bottom
        )
        print(
            f"# {spec.name} feature={spec.feature} higher_is_more={spec.higher_is_more}"
        )
        print(
            f"unique={np.unique(values).size:<4} "
            f"top_roles={top_roles.most_common(3)} "
            f"top_builds={top_builds.most_common(3)} "
            f"top_champs={top_champs.most_common(5)}"
        )
        print(
            f"bottom_roles={bottom_roles.most_common(3)} "
            f"bottom_builds={bottom_builds.most_common(3)} "
            f"bottom_champs={bottom_champs.most_common(5)}"
        )
        print()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    current = sub.add_parser("current")
    current.add_argument("--name", nargs="*")

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--name", nargs="*")
    sweep.add_argument(
        "--kvs",
        type=float,
        nargs="*",
        default=(0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.89, 0.90, 0.92, 0.95),
    )
    sweep.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=(0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80),
    )
    sweep.add_argument("--topn", type=int, default=8)
    sweep.add_argument("--output-json", type=Path)

    singular = sub.add_parser("singular")
    singular.add_argument("--name", nargs="*")
    singular.add_argument("--topn", type=int, default=50)
    return parser


def main() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    args = _parser().parse_args()
    if args.command == "current":
        current_audit(names=tuple(args.name) if args.name else None)
    elif args.command == "sweep":
        sweep_audit(
            names=tuple(args.name) if args.name else None,
            kvs=tuple(args.kvs),
            thresholds=tuple(args.thresholds),
            topn=args.topn,
            output_json=args.output_json,
        )
    elif args.command == "singular":
        singular_audit(
            names=tuple(args.name) if args.name else None,
            topn=args.topn,
        )


if __name__ == "__main__":
    main()
