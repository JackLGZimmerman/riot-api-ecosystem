from __future__ import annotations

import logging

import torch

ECE_BINS = 15


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


def prediction_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    total_loss: float,
    total_correct: int,
    total: int,
) -> dict[str, object]:
    """Headline scalars: loss, accuracy, auc, brier, ece (+ n)."""
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "auc": binary_auc(scores, targets),
        "brier": torch.mean((scores - targets) ** 2).item(),
        "ece": binary_ece(scores, targets),
        "n": total,
    }


# Per-mille integer edges (then /1000) avoid float drift.
# 0-30% and 70-100%: 5% slices; 30-40% and 60-70%: 2.5% slices; 40-60%: 0.5% slices.
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
    """Graduated band table over `_BAND_EDGES`."""
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


def log_prediction_bands(
    logger: logging.Logger,
    split_name: str,
    rows: list[dict[str, object]],
) -> None:
    logger.info(
        "%s graduated prediction bands:\n%s",
        split_name,
        format_prediction_band_table(rows),
    )
