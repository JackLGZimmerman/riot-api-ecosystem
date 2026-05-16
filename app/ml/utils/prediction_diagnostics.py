from __future__ import annotations

import logging

import torch

from app.ml.utils.metrics import metric_scalar

ECE_BINS = 15
PREDICTION_BUCKET_EDGES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
CENTRAL_PREDICTION_RANGES = ((0.475, 0.525), (0.45, 0.55), (0.40, 0.60))

# The 0.475-0.525 band is the critical decision region for this model. Its
# metrics are computed every epoch for both train and validation (cheap); the
# wider/narrower bands plus the bucket/distribution tables are computed only on
# the heavy-diagnostic cadence (prediction_metrics(..., full=True)).
HEADLINE_CENTRAL_RANGE = (0.475, 0.525)
_NON_HEADLINE_CENTRAL_RANGES = tuple(
    r for r in CENTRAL_PREDICTION_RANGES if r != HEADLINE_CENTRAL_RANGE
)
CENTRAL_BAND_KEYS = ("auc", "logloss", "accuracy", "brier", "calibration", "pct_data")


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


def prediction_summary_from_metrics(
    metrics: dict[str, object],
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {}
    for key, value in metrics.items():
        scalar = metric_scalar(value)
        if key.startswith("pred_") and scalar is not None:
            summary[key] = scalar
    return summary


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

    def _gap(name: str, train_key: str, held_out_key: str, sign: float) -> None:
        train_value = metric_scalar(train_metrics.get(train_key))
        held_out_value = metric_scalar(held_out_metrics.get(held_out_key))
        if train_value is not None and held_out_value is not None:
            gaps[name] = float(sign * (float(train_value) - float(held_out_value)))

    _gap("gen_loss_gap", "loss", "loss", -1.0)
    _gap("gen_accuracy_gap", "accuracy", "accuracy", 1.0)
    _gap("gen_auc_gap", "auc", "auc", 1.0)
    _gap("gen_brier_gap", "brier", "brier", -1.0)
    _gap("gen_ece_gap", "ece", "ece", -1.0)
    band = HEADLINE_CENTRAL_BAND
    _gap(
        f"gen_central_{band}_logloss_gap",
        f"pred_central_{band}_logloss",
        f"pred_central_{band}_logloss",
        -1.0,
    )
    _gap(
        f"gen_central_{band}_auc_gap",
        f"pred_central_{band}_auc",
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


def _prediction_distribution_fields(scores: torch.Tensor) -> dict[str, float]:
    if scores.numel() == 0:
        return {}
    quantiles = torch.quantile(
        scores.double(),
        torch.tensor(
            [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99],
            dtype=torch.float64,
            device=scores.device,
        ),
    )
    return {
        "pred_std": float(scores.std(unbiased=False).item()),
        "pred_p01": float(quantiles[0].item()),
        "pred_p05": float(quantiles[1].item()),
        "pred_p10": float(quantiles[2].item()),
        "pred_p50": float(quantiles[3].item()),
        "pred_p90": float(quantiles[4].item()),
        "pred_p95": float(quantiles[5].item()),
        "pred_p99": float(quantiles[6].item()),
    }


def _confidence_bucket_fields(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    fields: dict[str, float | int] = {}
    for threshold in (0.55, 0.60, 0.65):
        mask = scores > threshold
        count = int(mask.sum().item())
        label = int(round(threshold * 100))
        fields[f"pred_gt_{label}_count"] = count
        fields[f"pred_gt_{label}_accuracy"] = (
            float(targets[mask].mean().item()) if count else float("nan")
        )
    for threshold in (0.45, 0.40, 0.35):
        mask = scores < threshold
        count = int(mask.sum().item())
        label = int(round(threshold * 100))
        fields[f"pred_lt_{label}_count"] = count
        fields[f"pred_lt_{label}_accuracy"] = (
            float((1.0 - targets[mask]).mean().item()) if count else float("nan")
        )
    return fields


def prediction_bucket_diagnostics(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    n = scores.numel()
    rows: list[dict[str, object]] = []
    fields: dict[str, float | int] = {}
    edges = PREDICTION_BUCKET_EDGES
    bucket_specs: list[tuple[str, str, torch.Tensor]] = [
        (
            f"<{edges[0]:.2f}",
            f"lt_{int(round(edges[0] * 100))}",
            scores < edges[0],
        )
    ]
    for lower, upper in zip(edges, edges[1:]):
        lower_label = int(round(lower * 100))
        upper_label = int(round(upper * 100))
        if upper == edges[-1]:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        bucket_specs.append(
            (
                f"{lower:.2f}-{upper:.2f}",
                f"{lower_label}_{upper_label}",
                mask,
            )
        )
    bucket_specs.append(
        (
            f">{edges[-1]:.2f}",
            f"gt_{int(round(edges[-1] * 100))}",
            scores > edges[-1],
        )
    )

    for label, key, mask in bucket_specs:
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

        row = {
            "bucket": label,
            "count": count,
            "pct_data": pct_data,
            "mean_pred": mean_pred,
            "actual_rate": actual_rate,
            "gap": gap,
            "accuracy": accuracy,
            "logloss": logloss,
        }
        rows.append(row)
        for metric, value in row.items():
            if metric == "bucket":
                continue
            fields[f"pred_bucket_{key}_{metric}"] = value
    return rows, fields


def _central_prediction_fields(
    scores: torch.Tensor,
    targets: torch.Tensor,
    ranges: tuple[tuple[float, float], ...] = CENTRAL_PREDICTION_RANGES,
) -> dict[str, float | int]:
    fields: dict[str, float | int] = {}
    n = scores.numel()
    for lower, upper in ranges:
        label = central_range_label(lower, upper)
        mask = (scores >= lower) & (scores <= upper)
        count = int(mask.sum().item())
        fields[f"pred_central_{label}_count"] = count
        fields[f"pred_central_{label}_pct_data"] = (
            100.0 * count / n if n else float("nan")
        )
        if count:
            central_scores = scores[mask]
            central_targets = targets[mask]
            fields[f"pred_central_{label}_auc"] = binary_auc(
                central_scores, central_targets
            )
            fields[f"pred_central_{label}_calibration"] = binary_ece(
                central_scores, central_targets
            )
            fields[f"pred_central_{label}_accuracy"] = float(
                ((central_scores > 0.5) == central_targets).float().mean().item()
            )
            fields[f"pred_central_{label}_logloss"] = binary_logloss(
                central_scores, central_targets
            )
            fields[f"pred_central_{label}_brier"] = float(
                torch.mean((central_scores - central_targets) ** 2).item()
            )
        else:
            fields[f"pred_central_{label}_auc"] = float("nan")
            fields[f"pred_central_{label}_calibration"] = float("nan")
            fields[f"pred_central_{label}_accuracy"] = float("nan")
            fields[f"pred_central_{label}_logloss"] = float("nan")
            fields[f"pred_central_{label}_brier"] = float("nan")
    return fields


def prediction_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    total_loss: float,
    total_correct: int,
    total: int,
    *,
    full: bool = True,
) -> dict[str, object]:
    """Prediction-quality metrics.

    `full=False` computes only the core scalars plus the headline 0.475-0.525
    central band - everything needed every epoch. `full=True` adds the
    distribution quantiles, confidence buckets, the bucket table, and the
    remaining central bands (the heavy-diagnostic cadence).
    """
    mean_pred = scores.mean()
    blue_win_rate = targets.mean()
    metrics: dict[str, object] = {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "auc": binary_auc(scores, targets),
        "brier": torch.mean((scores - targets) ** 2).item(),
        "ece": binary_ece(scores, targets),
        "mean_pred": mean_pred.item(),
        "blue_win_rate": blue_win_rate.item(),
        "baseline_logloss": baseline_logloss(targets),
        "n": total,
    }
    metrics.update(
        _central_prediction_fields(scores, targets, (HEADLINE_CENTRAL_RANGE,))
    )
    if full:
        metrics.update(_prediction_distribution_fields(scores))
        metrics.update(_confidence_bucket_fields(scores, targets))
        bucket_rows, bucket_fields = prediction_bucket_diagnostics(scores, targets)
        metrics["prediction_bucket_table"] = bucket_rows
        metrics.update(bucket_fields)
        metrics.update(
            _central_prediction_fields(scores, targets, _NON_HEADLINE_CENTRAL_RANGES)
        )
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


def _format_central_prediction_table(metrics: dict[str, object]) -> str:
    lines = [
        "range    count    pct_data   auc      logloss   brier    calibration   accuracy"
    ]
    for lower, upper in CENTRAL_PREDICTION_RANGES:
        label = central_range_label(lower, upper)
        pct_data = metrics.get(f"pred_central_{label}_pct_data")
        pct_scalar = metric_scalar(pct_data)
        pct_text = f"{float(pct_scalar):.1f}%" if pct_scalar is not None else "-"
        count_scalar = metric_scalar(metrics.get(f"pred_central_{label}_count")) or 0
        lines.append(
            f"{lower:.3f}-{upper:.3f} "
            f"{int(count_scalar):>8,}  "
            f"{pct_text:>8}   "
            f"{_format_metric_cell(metrics.get(f'pred_central_{label}_auc')):>6}   "
            f"{_format_metric_cell(metrics.get(f'pred_central_{label}_logloss')):>7}   "
            f"{_format_metric_cell(metrics.get(f'pred_central_{label}_brier')):>6}   "
            f"{_format_metric_cell(metrics.get(f'pred_central_{label}_calibration')):>11}   "
            f"{_format_metric_cell(metrics.get(f'pred_central_{label}_accuracy')):>8}"
        )
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
    logger.info(
        "%s central prediction metrics:\n%s",
        split_name,
        _format_central_prediction_table(metrics),
    )
