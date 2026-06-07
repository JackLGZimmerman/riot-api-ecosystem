"""Estimate accuracy lift from central-band miss audit artifacts.

The estimates are diagnostic, not an ablation. They intentionally separate
miss-side upper bounds from a simple margin-conditioned heuristic so the report
does not confuse signal coverage with guaranteed model lift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_METRICS_PATH = Path("app/ml/data/metrics_latest.json")
MODEL_THRESHOLD = 0.516


def _load_candidates(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if summary is None:
            summary = data["summary"]
        rows.extend(data["batch"])
    return rows, summary or {}


def _load_allowed(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["all_games"]:
            out[str(row["matchid"])] = row
    return out


def _load_deep(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["matchid"]): row for row in data["all_games"]}


def _positive(value: Any) -> float:
    if value is None:
        return 0.0
    return max(float(value), 0.0)


def _edge_sum(allowed: dict[str, Any], deep: dict[str, Any]) -> float:
    values = [
        allowed.get("spell_edge_for_actual"),
        allowed.get("rune_edge_for_actual"),
        allowed.get("patch_edge_for_actual_train_overlap"),
    ]
    values.extend((deep.get("signal_edges") or {}).values())
    return sum(_positive(value) for value in values)


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    ordered = sorted(values)

    def q(frac: float) -> float:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * frac))]

    return {"mean": mean(values), "p50": q(0.50), "p75": q(0.75), "p90": q(0.90)}


def estimate(
    *,
    candidate_paths: list[Path],
    allowed_paths: list[Path],
    deep_path: Path,
    metrics_path: Path,
    output_path: Path,
    label: str,
    threshold: float,
) -> dict[str, Any]:
    candidates, band_summary = _load_candidates(candidate_paths)
    allowed_by_match = _load_allowed(allowed_paths)
    deep_by_match = _load_deep(deep_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    counts = {
        "games": 0,
        "threshold_correct": 0,
        "first_pass_unaccounted": 0,
        "deep_unaccounted": 0,
        "first_or_deep_unaccounted": 0,
        "threshold_or_any_signal": 0,
        "threshold_wrong_with_any_signal": 0,
        "margin_crosses_raw_with_any_signal": 0,
        "margin_crosses_threshold_with_any_signal": 0,
    }
    by_split: dict[str, dict[str, int]] = {}
    edge_sums: list[float] = []

    for candidate in candidates:
        matchid = str(candidate["matchid"])
        allowed = allowed_by_match[matchid]
        deep = deep_by_match[matchid]
        actual = str(candidate["actual_winner"])
        pred_blue = float(candidate["pred_blue_win"])
        threshold_correct = bool(allowed["threshold_correct"])
        first_signal = bool(allowed["unaccounted_influences"])
        deep_signal = bool(deep["deep_unaccounted_influences"])
        any_signal = first_signal or deep_signal
        edge_sum = _edge_sum(allowed, deep)
        raw_margin = abs(pred_blue - 0.5)
        threshold_margin = (
            threshold - pred_blue if actual == "blue" else pred_blue - threshold
        )
        threshold_margin = max(threshold_margin, 0.0)

        values = {
            "games": 1,
            "threshold_correct": int(threshold_correct),
            "first_pass_unaccounted": int(first_signal),
            "deep_unaccounted": int(deep_signal),
            "first_or_deep_unaccounted": int(any_signal),
            "threshold_or_any_signal": int(threshold_correct or any_signal),
            "threshold_wrong_with_any_signal": int((not threshold_correct) and any_signal),
            "margin_crosses_raw_with_any_signal": int(
                any_signal and edge_sum > 0.0 and edge_sum >= raw_margin
            ),
            "margin_crosses_threshold_with_any_signal": int(
                any_signal
                and (not threshold_correct)
                and edge_sum > 0.0
                and edge_sum >= threshold_margin
            ),
        }
        for key, value in values.items():
            counts[key] += value
        split_counts = by_split.setdefault(
            str(candidate["split"]), {key: 0 for key in counts}
        )
        for key, value in values.items():
            split_counts[key] += value
        edge_sums.append(edge_sum)

    rates = {
        key: (value / counts["games"] if counts["games"] else 0.0)
        for key, value in counts.items()
        if key != "games"
    }
    split_estimates: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        split_metrics = metrics[split]
        split_band = band_summary[split]
        total_n = int(split_metrics["n"])
        miss_n = int(split_band["central_misclassified_n"])
        raw_accuracy = float(split_metrics["accuracy"])
        threshold_accuracy = float(split_metrics["threshold_accuracy"])

        def lift(rate_key: str) -> float:
            return miss_n * rates[rate_key] / total_n

        split_estimates[split] = {
            "current_accuracy": raw_accuracy,
            "current_threshold_accuracy": threshold_accuracy,
            "band_games": int(split_band["central_n"]),
            "band_accuracy": float(split_band["central_accuracy"]),
            "band_misses": miss_n,
            "all_signal_miss_side_upper_bound_accuracy": raw_accuracy
            + lift("threshold_or_any_signal"),
            "signal_only_miss_side_upper_bound_accuracy": raw_accuracy
            + lift("first_or_deep_unaccounted"),
            "post_threshold_signal_upper_bound_accuracy": threshold_accuracy
            + lift("threshold_wrong_with_any_signal"),
            "margin_conditioned_raw_accuracy": raw_accuracy
            + lift("margin_crosses_raw_with_any_signal"),
            "margin_conditioned_threshold_accuracy": threshold_accuracy
            + lift("margin_crosses_threshold_with_any_signal"),
        }

    output = {
        "label": label,
        "candidate_paths": [str(path) for path in candidate_paths],
        "allowed_analysis_paths": [str(path) for path in allowed_paths],
        "deep_review_path": str(deep_path),
        "metrics_path": str(metrics_path),
        "decision_threshold": threshold,
        "sample_counts": counts,
        "sample_rates": rates,
        "sample_counts_by_split": by_split,
        "edge_sum_distribution": _quantiles(edge_sums),
        "split_estimates": split_estimates,
        "methodology": (
            "Upper bounds assume sampled miss-side signal coverage extrapolates to "
            "all band misses and causes no regressions. Margin-conditioned estimates "
            "count only sampled misses whose positive actual-side prior edges meet "
            "the current decision margin; these are still heuristics, not ablations."
        ),
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--allowed-analysis", type=Path, action="append", required=True)
    parser.add_argument("--deep-review", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--decision-threshold", type=float, default=MODEL_THRESHOLD)
    args = parser.parse_args()

    result = estimate(
        candidate_paths=args.candidate,
        allowed_paths=args.allowed_analysis,
        deep_path=args.deep_review,
        metrics_path=args.metrics,
        output_path=args.output,
        label=args.label,
        threshold=args.decision_threshold,
    )
    print(json.dumps(result["split_estimates"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
