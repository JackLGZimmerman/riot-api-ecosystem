from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


METRICS_FILE = "metrics.jsonl"


def _resolve_metrics_path(path: Path) -> Path:
    if path.is_dir():
        return path / METRICS_FILE
    return path


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    metrics_path = _resolve_metrics_path(path)
    rows: list[dict[str, Any]] = []
    with metrics_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _fmt_float(value: object, digits: int = 5) -> str:
    v = _as_float(value)
    if v is None or math.isnan(v):
        return "-"
    return f"{v:.{digits}f}"


def _fmt_delta(value: object, digits: int = 5) -> str:
    v = _as_float(value)
    if v is None or math.isnan(v):
        return "-"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    v = _as_float(value)
    if v is None or math.isnan(v):
        return "-"
    return f"{100.0 * v:.{digits}f}%"


def _parse_percent_band(label: str) -> tuple[float, float] | None:
    try:
        lo, hi = label.split("-")
        return float(lo.rstrip("%")) / 100.0, float(hi.rstrip("%")) / 100.0
    except ValueError:
        return None


def _weighted_accuracy_pct(rows: list[dict[str, Any]]) -> float | None:
    total = 0
    correct_pct_sum = 0.0
    for row in rows:
        count = _as_int(row.get("count"))
        accuracy_pct = _as_float(row.get("accuracy_pct"))
        if count is None or accuracy_pct is None or math.isnan(accuracy_pct):
            continue
        total += count
        correct_pct_sum += count * accuracy_pct
    if total == 0:
        return None
    return correct_pct_sum / total


def _prediction_density(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bands: list[tuple[float, float, dict[str, Any]]] = []
    for row in rows:
        parsed = _parse_percent_band(str(row.get("band", "")))
        if parsed is not None:
            bands.append((*parsed, row))
    total = sum(_as_int(row.get("count")) or 0 for _, _, row in bands)

    def collect(lo: float, hi: float) -> dict[str, Any]:
        selected = [
            row
            for band_lo, band_hi, row in bands
            if band_lo >= lo and band_hi <= hi
        ]
        count = sum(_as_int(row.get("count")) or 0 for row in selected)
        return {
            "count": count,
            "share": count / total if total else None,
            "accuracy": (
                (_weighted_accuracy_pct(selected) or float("nan")) / 100.0
                if selected
                else None
            ),
        }

    low_tail = collect(0.0, 0.4)
    high_tail = collect(0.6, 1.0)
    confident_low = collect(0.0, 0.3)
    confident_high = collect(0.7, 1.0)

    def combine(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
        count = int(first["count"]) + int(second["count"])
        acc_values = []
        for item in (first, second):
            acc = _as_float(item.get("accuracy"))
            item_count = _as_int(item.get("count")) or 0
            if acc is not None and not math.isnan(acc):
                acc_values.append((item_count, acc))
        weighted_acc = (
            sum(count_i * acc for count_i, acc in acc_values) / count
            if count
            else None
        )
        return {
            "count": count,
            "share": count / total if total else None,
            "accuracy": weighted_acc,
        }

    return {
        "total": total,
        "central_40_60": collect(0.4, 0.6),
        "central_45_55": collect(0.45, 0.55),
        "tails_40_60": combine(low_tail, high_tail),
        "confident_30_70": combine(confident_low, confident_high),
    }


def summarize_metrics_file(path: Path) -> dict[str, Any]:
    metrics_path = _resolve_metrics_path(path)
    rows = _iter_rows(metrics_path)
    if not rows:
        raise ValueError(f"no readable metrics rows in {metrics_path}")

    run_start = next((row for row in rows if row.get("event") == "run_start"), {})

    epoch_rows = [row for row in rows if row.get("event") == "epoch_end"]
    best_loss_row = min(
        (
            row
            for row in epoch_rows
            if isinstance(row.get("val_loss"), (int, float))
        ),
        key=lambda row: float(row["val_loss"]),
        default={},
    )
    best_acc_row = max(
        (
            row
            for row in epoch_rows
            if isinstance(row.get("val_accuracy"), (int, float))
        ),
        key=lambda row: float(row["val_accuracy"]),
        default={},
    )
    early_stop = next(
        (row for row in reversed(rows) if row.get("event") == "early_stop"),
        {},
    )
    test = next((row for row in reversed(rows) if row.get("event") == "test"), {})
    prediction_bands = next(
        (
            row.get("rows")
            for row in reversed(rows)
            if row.get("event") == "prediction_bands"
        ),
        None,
    )

    return {
        "path": str(metrics_path),
        "name": metrics_path.parent.name if metrics_path.name == METRICS_FILE else metrics_path.stem,
        "lr": run_start.get("lr"),
        "weight_decay": run_start.get("weight_decay"),
        "warmup_steps": run_start.get("warmup_steps"),
        "lr_center_epoch": run_start.get("lr_center_epoch"),
        "lr_sharpness": run_start.get("lr_sharpness"),
        "lr_tail_strength": run_start.get("lr_tail_strength"),
        "best_epoch": best_loss_row.get("epoch"),
        "best_val_loss": best_loss_row.get("val_loss"),
        "best_val_accuracy": best_loss_row.get("val_accuracy"),
        "best_val_auc": best_loss_row.get("val_auc"),
        "best_val_brier": best_loss_row.get("val_brier"),
        "best_val_ece": best_loss_row.get("val_ece"),
        "max_val_accuracy": best_acc_row.get("val_accuracy"),
        "max_val_accuracy_epoch": best_acc_row.get("epoch"),
        "last_epoch": epoch_rows[-1].get("epoch") if epoch_rows else None,
        "early_stop_epoch": early_stop.get("epoch"),
        "test_loss": test.get("test_loss"),
        "test_accuracy": test.get("test_accuracy"),
        "test_auc": test.get("test_auc"),
        "test_brier": test.get("test_brier"),
        "test_ece": test.get("test_ece"),
        "test_n": test.get("test_n"),
        "prediction_density": (
            _prediction_density(prediction_bands)
            if isinstance(prediction_bands, list)
            else None
        ),
    }


def _delta_from(summary: dict[str, Any], base: dict[str, Any], key: str) -> float | None:
    left = _as_float(summary.get(key))
    right = _as_float(base.get(key))
    if left is None or right is None:
        return None
    return left - right


def format_summary_table(
    summaries: list[dict[str, Any]],
    *,
    baseline: dict[str, Any] | None = None,
) -> str:
    baseline = baseline or (summaries[0] if summaries else None)
    headers = [
        "run",
        "best val",
        "d val",
        "val acc",
        "max acc",
        "test",
        "test acc",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for summary in summaries:
        d_val = _delta_from(summary, baseline, "best_val_loss") if baseline else None
        row = [
            str(summary.get("name") or "-"),
            _fmt_float(summary.get("best_val_loss")),
            _fmt_delta(d_val),
            _fmt_pct(summary.get("best_val_accuracy"), 3),
            _fmt_pct(summary.get("max_val_accuracy"), 3),
            _fmt_float(summary.get("test_loss")),
            _fmt_pct(summary.get("test_accuracy"), 3),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_run_details(summary: dict[str, Any]) -> str:
    density = summary.get("prediction_density")
    density_bits: list[str] = []
    if isinstance(density, dict):
        for key, label in (
            ("central_40_60", "40-60"),
            ("central_45_55", "45-55"),
            ("tails_40_60", "tails"),
            ("confident_30_70", "30/70+"),
        ):
            item = density.get(key)
            if not isinstance(item, dict):
                continue
            density_bits.append(
                f"{label}: {item.get('count', '-')}"
                f" ({_fmt_pct(item.get('share'), 2)},"
                f" acc {_fmt_pct(item.get('accuracy'), 2)})"
            )
    train_bits = [
        f"lr={summary.get('lr')}",
        f"wd={summary.get('weight_decay')}",
        f"warmup={summary.get('warmup_steps')}",
        f"center={summary.get('lr_center_epoch')}",
        f"sharp={summary.get('lr_sharpness')}",
        f"tail={summary.get('lr_tail_strength')}",
    ]
    return "\n".join(
        [
            f"### {summary.get('name')}",
            f"- path: `{summary.get('path')}`",
            f"- train: {', '.join(train_bits)}",
            f"- best: epoch {summary.get('best_epoch')} val_loss "
            f"{_fmt_float(summary.get('best_val_loss'))}, val_acc "
            f"{_fmt_pct(summary.get('best_val_accuracy'), 3)}",
            f"- density: {'; '.join(density_bits) if density_bits else '-'}",
        ]
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise ML metrics JSONL files into decision-grade tables."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Metrics file or run directory to use for deltas.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable summary JSON instead of Markdown.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-run config and prediction-density notes.",
    )
    parser.add_argument(
        "--sort",
        choices=("input", "best_val_loss", "max_val_accuracy"),
        default="input",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    summaries = [summarize_metrics_file(path) for path in args.paths]
    if args.sort == "best_val_loss":
        summaries.sort(key=lambda row: _as_float(row.get("best_val_loss")) or math.inf)
    elif args.sort == "max_val_accuracy":
        summaries.sort(
            key=lambda row: _as_float(row.get("max_val_accuracy")) or -math.inf,
            reverse=True,
        )

    baseline = summarize_metrics_file(args.baseline) if args.baseline else None
    if args.json:
        print(json.dumps({"runs": summaries, "baseline": baseline}, indent=2))
        return 0

    print(format_summary_table(summaries, baseline=baseline))
    if args.details:
        print()
        print("\n\n".join(_format_run_details(summary) for summary in summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
