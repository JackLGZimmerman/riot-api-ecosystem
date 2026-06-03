"""Model-alignment helpers for HGNN semantic-context threshold audits."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

DEFAULT_AUDIT_PATH = Path("app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md")


@dataclass(frozen=True)
class ThresholdBin:
    """Inclusive/exclusive numeric threshold bin over side-row axis values."""

    label: str
    lower: float | None = None
    upper: float | None = None
    include_lower: bool = True
    include_upper: bool = True

    def mask(self, values: np.ndarray) -> np.ndarray:
        mask = np.ones(values.shape, dtype=bool)
        if self.lower is not None:
            if self.include_lower:
                mask &= values >= float(self.lower)
            else:
                mask &= values > float(self.lower)
        if self.upper is not None:
            if self.include_upper:
                mask &= values <= float(self.upper)
            else:
                mask &= values < float(self.upper)
        return mask


@dataclass(frozen=True)
class AuditBinResult:
    """One threshold-bin row with empirical and model-alignment rates."""

    label: str
    n: int
    empirical_wr: float
    base_pred_wr: float
    final_pred_wr: float
    base_gap: float
    final_gap: float


def side_row_focus_probabilities(
    *,
    blue_win: Sequence[float] | np.ndarray,
    base_blue_probability: Sequence[float] | np.ndarray,
    final_blue_probability: Sequence[float] | np.ndarray,
    slots_per_team: int = 5,
) -> dict[str, np.ndarray]:
    """Expand game-level blue-win probabilities into focus-side slot rows.

    Blue slots keep the blue-team probability and label. Red slots use the
    mirrored probability/label, matching the side-row perspective used by the
    semantic context examples audit.
    """

    labels = _as_1d_float("blue_win", blue_win)
    base = _as_1d_float("base_blue_probability", base_blue_probability)
    final = _as_1d_float("final_blue_probability", final_blue_probability)
    if base.size != labels.size or final.size != labels.size:
        raise ValueError("blue_win, base_blue_probability, and final_blue_probability sizes differ")
    if slots_per_team <= 0:
        raise ValueError("slots_per_team must be positive")

    blue_labels = np.repeat(labels[:, None], slots_per_team, axis=1)
    red_labels = np.repeat((1.0 - labels)[:, None], slots_per_team, axis=1)
    blue_base = np.repeat(base[:, None], slots_per_team, axis=1)
    red_base = np.repeat((1.0 - base)[:, None], slots_per_team, axis=1)
    blue_final = np.repeat(final[:, None], slots_per_team, axis=1)
    red_final = np.repeat((1.0 - final)[:, None], slots_per_team, axis=1)
    return {
        "label": np.concatenate([blue_labels, red_labels], axis=1).reshape(-1),
        "base_prediction": np.concatenate([blue_base, red_base], axis=1).reshape(-1),
        "final_prediction": np.concatenate([blue_final, red_final], axis=1).reshape(-1),
    }


def evaluate_threshold_bins(
    *,
    axis_values: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    base_predictions: Sequence[float] | np.ndarray,
    final_predictions: Sequence[float] | np.ndarray,
    bins: Sequence[ThresholdBin],
) -> list[AuditBinResult]:
    """Compute empirical WR, model WR, and prediction gaps for threshold bins."""

    axis = _as_1d_float("axis_values", axis_values)
    y = _as_1d_float("labels", labels)
    base = _as_1d_float("base_predictions", base_predictions)
    final = _as_1d_float("final_predictions", final_predictions)
    if not (axis.size == y.size == base.size == final.size):
        raise ValueError("axis_values, labels, base_predictions, and final_predictions sizes differ")
    rows: list[AuditBinResult] = []
    for bin_spec in bins:
        mask = bin_spec.mask(axis)
        n = int(mask.sum())
        empirical = _mean_or_nan(y[mask])
        base_wr = _mean_or_nan(base[mask])
        final_wr = _mean_or_nan(final[mask])
        rows.append(
            AuditBinResult(
                label=bin_spec.label,
                n=n,
                empirical_wr=empirical,
                base_pred_wr=base_wr,
                final_pred_wr=final_wr,
                base_gap=base_wr - empirical,
                final_gap=final_wr - empirical,
            )
        )
    return rows


def gap_summary(rows: Sequence[AuditBinResult]) -> dict[str, float | int]:
    """Mean/max absolute gap and gap MSE for base and final model columns."""

    base_gaps = _finite_gaps([row.base_gap for row in rows if row.n > 0])
    final_gaps = _finite_gaps([row.final_gap for row in rows if row.n > 0])
    return {
        "n_bins": int(len(rows)),
        "n_populated_bins": int(sum(row.n > 0 for row in rows)),
        "base_mean_abs_gap": _mean_or_nan(np.abs(base_gaps)),
        "base_max_abs_gap": _max_or_nan(np.abs(base_gaps)),
        "base_gap_mse": _mean_or_nan(base_gaps**2),
        "final_mean_abs_gap": _mean_or_nan(np.abs(final_gaps)),
        "final_max_abs_gap": _max_or_nan(np.abs(final_gaps)),
        "final_gap_mse": _mean_or_nan(final_gaps**2),
    }


def render_model_alignment_audit(
    sections: Mapping[str, Sequence[AuditBinResult]],
    *,
    title: str = "HGNN Context Examples Audit",
    updated: str | None = None,
) -> str:
    """Render model-alignment audit sections as markdown."""

    updated = updated or date.today().isoformat()
    lines = [
        f"# {title}",
        "",
        f"Updated: {updated}.",
        "",
        "This model-alignment audit compares empirical focus-side WR against "
        "base and final HGNN predictions by threshold bin. Zero prediction gap "
        "is the optimization target.",
        "",
        "## Gap Summary",
        "",
        "| Section | Bins | Base mean abs gap | Base max abs gap | Base gap MSE | Final mean abs gap | Final max abs gap | Final gap MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for section_name, rows in sections.items():
        summary = gap_summary(rows)
        lines.append(
            "| "
            + " | ".join(
                [
                    section_name,
                    str(summary["n_populated_bins"]),
                    _format_pp(summary["base_mean_abs_gap"], signed=False),
                    _format_pp(summary["base_max_abs_gap"], signed=False),
                    _format_pp_mse(summary["base_gap_mse"]),
                    _format_pp(summary["final_mean_abs_gap"], signed=False),
                    _format_pp(summary["final_max_abs_gap"], signed=False),
                    _format_pp_mse(summary["final_gap_mse"]),
                ]
            )
            + " |"
        )
    for section_name, rows in sections.items():
        lines.extend(
            [
                "",
                f"## {section_name}",
                "",
                "| Bin | n | Empirical WR | Base predicted WR | Final predicted WR | Base gap | Final gap |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if not rows:
            lines.append("| N/A | 0 | N/A | N/A | N/A | N/A | N/A |")
            continue
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.label,
                        f"{row.n:,}",
                        _format_pct(row.empirical_wr),
                        _format_pct(row.base_pred_wr),
                        _format_pct(row.final_pred_wr),
                        _format_pp(row.base_gap),
                        _format_pp(row.final_gap),
                    ]
                )
                + " |"
            )
    all_rows = [row for rows in sections.values() for row in rows]
    summary = gap_summary(all_rows)
    lines.extend(
        [
            "",
            "## Overall Summary",
            "",
            "| Tests | Populated bins | Base mean abs gap | Base max abs gap | Base gap MSE | Final mean abs gap | Final max abs gap | Final gap MSE |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| "
            + " | ".join(
                [
                    str(summary["n_bins"]),
                    str(summary["n_populated_bins"]),
                    _format_pp(summary["base_mean_abs_gap"], signed=False),
                    _format_pp(summary["base_max_abs_gap"], signed=False),
                    _format_pp_mse(summary["base_gap_mse"]),
                    _format_pp(summary["final_mean_abs_gap"], signed=False),
                    _format_pp(summary["final_max_abs_gap"], signed=False),
                    _format_pp_mse(summary["final_gap_mse"]),
                ]
            )
            + " |",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def write_model_alignment_audit(
    path: Path,
    sections: Mapping[str, Sequence[AuditBinResult]],
    *,
    title: str = "HGNN Context Examples Audit",
    updated: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_model_alignment_audit(sections, title=title, updated=updated),
        encoding="utf-8",
    )


def threshold_bins_from_json(path: Path) -> list[ThresholdBin]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("threshold bin JSON must be a list")
    bins: list[ThresholdBin] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("threshold bin JSON entries must be objects")
        bins.append(ThresholdBin(**item))
    return bins


def _as_1d_float(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _max_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def _finite_gaps(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{100.0 * value:.2f}%"


def _format_pp(value: float | int, *, signed: bool = True) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return "N/A"
    sign = "+" if signed and numeric >= 0.0 else ""
    return f"{sign}{100.0 * numeric:.2f} pp"


def _format_pp_mse(value: float | int) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return "N/A"
    return f"{10000.0 * numeric:.2f} pp^2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-values", type=Path, required=True)
    parser.add_argument("--blue-win", type=Path, required=True)
    parser.add_argument("--base-blue-probability", type=Path, required=True)
    parser.add_argument("--final-blue-probability", type=Path, required=True)
    parser.add_argument("--bins-json", type=Path, required=True)
    parser.add_argument("--section", default="Model alignment")
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    side_rows = side_row_focus_probabilities(
        blue_win=np.load(args.blue_win),
        base_blue_probability=np.load(args.base_blue_probability),
        final_blue_probability=np.load(args.final_blue_probability),
    )
    rows = evaluate_threshold_bins(
        axis_values=np.load(args.axis_values),
        labels=side_rows["label"],
        base_predictions=side_rows["base_prediction"],
        final_predictions=side_rows["final_prediction"],
        bins=threshold_bins_from_json(args.bins_json),
    )
    write_model_alignment_audit(args.output, {args.section: rows})


if __name__ == "__main__":
    main()
