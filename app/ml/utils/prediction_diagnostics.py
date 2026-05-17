from __future__ import annotations

import logging

import torch

from app.ml.utils.metrics import metric_scalar

ECE_BINS = 15

PREDICTION_BUCKET_EDGES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)

# The 0.475-0.525 band is the critical decision region for this model. Only
# auc / logloss / calibration are emitted: brier ~ 0.25, accuracy ~ 0.5 by
# construction for predictions in this band, so they carry no extra signal.
HEADLINE_CENTRAL_RANGE = (0.475, 0.525)
CENTRAL_BAND_KEYS = ("auc", "logloss", "calibration")


def central_range_label(lower: float, upper: float) -> str:
    def label_part(value: float) -> str:
        percent = round(value * 100, 10)
        if percent.is_integer():
            return str(int(percent))
        per_mille = round(value * 1000, 10)
        if per_mille.is_integer():
            return str(int(per_mille))
        return f"{value:.6g}".removeprefix("0.").replace(".", "_")

    return f"{label_part(lower)}_{label_part(upper)}"


HEADLINE_CENTRAL_BAND = central_range_label(*HEADLINE_CENTRAL_RANGE)


def central_band_summary(
    metrics: dict[str, object],
    band: str = HEADLINE_CENTRAL_BAND,
) -> dict[str, float]:
    """Headline central-band metrics, compact enough to log every epoch."""
    summary: dict[str, float] = {}
    for key in CENTRAL_BAND_KEYS:
        scalar = metric_scalar(metrics.get(f"pred_central_{band}_{key}"))
        if scalar is not None:
            summary[f"central_{band}_{key}"] = float(scalar)
    return summary


def generalization_gaps(
    train_metrics: dict[str, object],
    held_out_metrics: dict[str, object],
) -> dict[str, float]:
    """Train-vs-held-out metric gaps.

    Signs are normalised so a positive gap always means the train split
    outperforms the held-out split - the overfitting direction - regardless of
    whether the underlying metric is better high (AUC) or low (loss).
    """
    gaps: dict[str, float] = {}

    def _gap(name: str, key: str, sign: float) -> None:
        train_value = metric_scalar(train_metrics.get(key))
        held_out_value = metric_scalar(held_out_metrics.get(key))
        if train_value is not None and held_out_value is not None:
            gaps[name] = float(sign * (float(train_value) - float(held_out_value)))

    _gap("gen_loss_gap", "loss", -1.0)
    _gap("gen_accuracy_gap", "accuracy", 1.0)
    _gap("gen_auc_gap", "auc", 1.0)
    band = HEADLINE_CENTRAL_BAND
    _gap(
        f"gen_central_{band}_auc_gap",
        f"pred_central_{band}_auc",
        1.0,
    )
    return gaps


def binary_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic. Returns NaN if degenerate."""
    n_pos = int(targets.sum().item())
    n_neg = targets.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(
        1, scores.numel() + 1, dtype=torch.float64, device=scores.device
    )
    sum_pos_ranks = ranks[targets > 0.5].sum().item()
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def binary_ece(
    scores: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = ECE_BINS,
) -> float:
    scores = scores.double()
    targets = targets.double()
    n = scores.numel()
    if n == 0:
        return float("nan")

    ece = torch.zeros((), dtype=torch.float64)
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        count = int(mask.sum().item())
        if count == 0:
            continue
        confidence = scores[mask].mean()
        accuracy = targets[mask].mean()
        ece += (count / n) * torch.abs(confidence - accuracy)
    return float(ece.item())


def baseline_logloss(targets: torch.Tensor) -> float:
    targets = targets.double()
    if targets.numel() == 0:
        return float("nan")
    rate = targets.mean()
    eps = torch.finfo(torch.float64).eps
    p = torch.clamp(rate, eps, 1.0 - eps)
    loss = -(rate * torch.log(p) + (1.0 - rate) * torch.log1p(-p))
    return float(loss.item())


def binary_logloss(scores: torch.Tensor, targets: torch.Tensor) -> float:
    if scores.numel() == 0:
        return float("nan")
    scores = scores.double()
    targets = targets.double()
    eps = torch.finfo(torch.float64).eps
    probs = torch.clamp(scores, eps, 1.0 - eps)
    loss = -(targets * torch.log(probs) + (1.0 - targets) * torch.log1p(-probs))
    return float(loss.mean().item())


def prediction_bucket_rows(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> list[dict[str, object]]:
    """Rows for the heavy-cadence console bucket table.

    Scalar bucket fields are not emitted to JSONL - the graduated band table
    (prediction_band_diagnostics) is the canonical per-bucket record.
    """
    n = scores.numel()
    rows: list[dict[str, object]] = []
    edges = PREDICTION_BUCKET_EDGES
    bucket_specs: list[tuple[str, torch.Tensor]] = [
        (f"<{edges[0]:.2f}", scores < edges[0])
    ]
    for lower, upper in zip(edges, edges[1:]):
        if upper == edges[-1]:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        bucket_specs.append((f"{lower:.2f}-{upper:.2f}", mask))
    bucket_specs.append((f">{edges[-1]:.2f}", scores > edges[-1]))

    for label, mask in bucket_specs:
        count = int(mask.sum().item())
        pct_data = 100.0 * count / n if n else float("nan")
        if count:
            bucket_scores = scores[mask]
            bucket_targets = targets[mask]
            actual_rate = float(bucket_targets.mean().item())
            mean_pred = float(bucket_scores.mean().item())
            accuracy = float(
                ((bucket_scores > 0.5) == bucket_targets).float().mean().item()
            )
            logloss = binary_logloss(bucket_scores, bucket_targets)
            gap = actual_rate - mean_pred
        else:
            mean_pred = float("nan")
            actual_rate = float("nan")
            gap = float("nan")
            accuracy = float("nan")
            logloss = float("nan")

        rows.append(
            {
                "bucket": label,
                "count": count,
                "pct_data": pct_data,
                "mean_pred": mean_pred,
                "actual_rate": actual_rate,
                "gap": gap,
                "accuracy": accuracy,
                "logloss": logloss,
            }
        )
    return rows


def _headline_central_fields(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """auc / logloss / calibration on the 0.475-0.525 decision band."""
    lower, upper = HEADLINE_CENTRAL_RANGE
    label = HEADLINE_CENTRAL_BAND
    mask = (scores >= lower) & (scores <= upper)
    if int(mask.sum().item()) == 0:
        return {
            f"pred_central_{label}_auc": float("nan"),
            f"pred_central_{label}_logloss": float("nan"),
            f"pred_central_{label}_calibration": float("nan"),
        }
    central_scores = scores[mask]
    central_targets = targets[mask]
    return {
        f"pred_central_{label}_auc": binary_auc(central_scores, central_targets),
        f"pred_central_{label}_logloss": binary_logloss(
            central_scores, central_targets
        ),
        f"pred_central_{label}_calibration": binary_ece(
            central_scores, central_targets
        ),
    }


def prediction_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    total_loss: float,
    total_correct: int,
    total: int,
    *,
    bucket_table: bool = False,
) -> dict[str, object]:
    """Prediction-quality metrics.

    Always returns the core scalars + the headline 0.475-0.525 central band.
    `mean_pred`, `blue_win_rate`, and `baseline_logloss` are kept available
    for the final-test report; per-epoch logging deliberately excludes them.
    `bucket_table=True` attaches `prediction_bucket_table` rows for the
    heavy-cadence console log; the graduated band table is owned by the
    caller (see `prediction_band_diagnostics`).
    """
    metrics: dict[str, object] = {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "auc": binary_auc(scores, targets),
        "brier": torch.mean((scores - targets) ** 2).item(),
        "ece": binary_ece(scores, targets),
        "mean_pred": scores.mean().item(),
        "blue_win_rate": targets.mean().item(),
        "baseline_logloss": baseline_logloss(targets),
        "n": total,
    }
    metrics.update(_headline_central_fields(scores, targets))
    if bucket_table:
        metrics["prediction_bucket_table"] = prediction_bucket_rows(scores, targets)
    return metrics


def _format_metric_cell(value: object, decimals: int = 3) -> str:
    scalar = metric_scalar(value)
    if scalar is None:
        return "-"
    return f"{float(scalar):.{decimals}f}"


def _format_prediction_bucket_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "bucket       count    pct_data   mean_pred   actual_rate   gap      accuracy   logloss"
    ]
    for row in rows:
        pct_data = row.get("pct_data")
        pct_scalar = metric_scalar(pct_data)
        pct_text = f"{float(pct_scalar):.1f}%" if pct_scalar is not None else "-"
        count_scalar = metric_scalar(row.get("count")) or 0
        lines.append(
            f"{str(row['bucket']):<11} "
            f"{int(count_scalar):>8,}  "
            f"{pct_text:>8}   "
            f"{_format_metric_cell(row.get('mean_pred')):>9}   "
            f"{_format_metric_cell(row.get('actual_rate')):>11}   "
            f"{_format_metric_cell(row.get('gap')):>6}   "
            f"{_format_metric_cell(row.get('accuracy')):>8}   "
            f"{_format_metric_cell(row.get('logloss')):>7}"
        )
    return "\n".join(lines)


# Per-mille integer edges (then /1000) avoid float drift.
# 0-30% and 70-100%: 5% slices; 30-40% and 60-70%: 1% slices; 40-60%: 0.1% slices.
_BAND_EDGES: tuple[float, ...] = tuple(
    x / 1000
    for x in (
        *range(50, 250, 50),    # 5-25 in 5% bins
        *range(250, 400, 25),  # 25-40 in 2.5% bins
        *range(400, 600, 5),   # 40-60 in 0.5% bins
        *range(600, 750, 25),  # 60-75 in 2.5% bins
        *range(750, 950, 50),  # 75-95 in 5% bins
        1000,
    )
)


def prediction_band_diagnostics(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> list[dict[str, object]]:
    """Graduated band table emitted on heavy epochs + final test."""
    n_bins = len(_BAND_EDGES) - 1
    rows: list[dict[str, object]] = []
    for i, (lo, hi) in enumerate(zip(_BAND_EDGES, _BAND_EDGES[1:])):
        mask = (scores >= lo) & (scores <= hi if i == n_bins - 1 else scores < hi)
        count = int(mask.sum().item())
        if count > 0:
            bin_preds = scores[mask]
            bin_targets = targets[mask]
            accuracy_pct = 100.0 * float(
                ((bin_preds > 0.5) == bin_targets).float().mean().item()
            )
        else:
            accuracy_pct = float("nan")
        rows.append({"band": f"{lo:.1%}-{hi:.1%}", "count": count, "accuracy_pct": accuracy_pct})
    return rows


def format_prediction_band_table(rows: list[dict[str, object]]) -> str:
    section_headers = {
        "0.0%-5.0%": "--- 5% bins (0-30%) ---",
        "30.0%-31.0%": "--- 1% bins (30-40%) ---",
        "40.0%-40.1%": "--- 0.1% bins (40-60%) ---",
        "60.0%-61.0%": "--- 1% bins (60-70%) ---",
        "70.0%-75.0%": "--- 5% bins (70-100%) ---",
    }
    lines: list[str] = ["band             count    accuracy"]
    for row in rows:
        band = str(row["band"])
        if band in section_headers:
            lines.append(section_headers[band])
        count = int(row["count"])  # type: ignore[arg-type]
        acc = row["accuracy_pct"]
        acc_str = f"{float(acc):.1f}%" if isinstance(acc, float) and not (acc != acc) else "-"
        lines.append(f"{band:<16} {count:>8,}  {acc_str:>8}")
    return "\n".join(lines)


def log_prediction_diagnostics(
    logger: logging.Logger,
    split_name: str,
    metrics: dict[str, object],
) -> None:
    rows = metrics.get("prediction_bucket_table")
    if isinstance(rows, list):
        logger.info(
            "%s prediction buckets:\n%s",
            split_name,
            _format_prediction_bucket_table(rows),
        )
