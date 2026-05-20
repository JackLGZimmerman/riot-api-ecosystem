from __future__ import annotations

import logging
from typing import cast

import torch
from torch.nn import functional as F

from app.ml.utils.prediction_diagnostics import binary_auc, binary_ece

_CENTRAL_EDGES: tuple[float, ...] = (
    0.400,
    0.425,
    0.450,
    0.475,
    0.500,
    0.525,
    0.550,
    0.575,
    0.600,
)
_FOLDED_CONFIDENCE_EDGES: tuple[float, ...] = (
    0.500,
    0.525,
    0.550,
    0.575,
    0.600,
    1.000,
)
_MATCHED_BAND_EDGES: tuple[float, ...] = (
    0.000,
    0.300,
    *_CENTRAL_EDGES,
    0.700,
    1.000,
)


def _band_mask(values: torch.Tensor, lo: float, hi: float, last: bool) -> torch.Tensor:
    return (values >= lo) & (values <= hi if last else values < hi)


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:.1%}-{hi:.1%}"


def _metric_set(
    scores: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    if scores.numel() == 0:
        return {
            "bce": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "auc": float("nan"),
            "accuracy": float("nan"),
        }
    return {
        "bce": F.binary_cross_entropy_with_logits(logits, targets).item(),
        "brier": ((scores - targets) ** 2).mean().item(),
        "ece": binary_ece(scores, targets),
        "auc": binary_auc(scores, targets),
        "accuracy": ((scores > 0.5) == targets).float().mean().item(),
    }


def _metric_comparison(
    mask: torch.Tensor,
    baseline_scores: torch.Tensor,
    baseline_logits: torch.Tensor,
    final_scores: torch.Tensor,
    final_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object]:
    baseline = _metric_set(baseline_scores[mask], baseline_logits[mask], targets[mask])
    final = _metric_set(final_scores[mask], final_logits[mask], targets[mask])
    return {
        "baseline": baseline,
        "final": final,
        "delta": {key: final[key] - baseline[key] for key in baseline},
    }


def _mean_or_nan(values: torch.Tensor) -> float:
    return values.mean().item() if values.numel() else float("nan")


def _quantile_or_nan(values: torch.Tensor, q: float) -> float:
    return values.quantile(q).item() if values.numel() else float("nan")


def _top_margin(weights: torch.Tensor) -> torch.Tensor:
    if weights.shape[-1] == 1:
        return weights.squeeze(-1)
    top2 = weights.topk(2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def _expert_rows(
    weights: torch.Tensor,
    corrections: torch.Tensor,
    final_scores: torch.Tensor,
    targets: torch.Tensor,
) -> list[dict[str, object]]:
    selected = weights > 0.0
    hard_ok = ((final_scores > 0.5) == (targets > 0.5)).float()
    rows: list[dict[str, object]] = []
    for expert_id in range(weights.shape[-1]):
        mask = selected[:, expert_id]
        correction = corrections[mask, expert_id]
        abs_correction = correction.abs()
        selected_weight = weights[mask, expert_id]
        rows.append(
            {
                "expert": expert_id,
                "selected_share": mask.float().mean().item(),
                "mean_weight": weights[:, expert_id].mean().item(),
                "mean_selected_weight": _mean_or_nan(selected_weight),
                "correction_mean": _mean_or_nan(correction),
                "abs_correction_p50": _quantile_or_nan(abs_correction, 0.50),
                "abs_correction_p90": _quantile_or_nan(abs_correction, 0.90),
                "abs_correction_p99": _quantile_or_nan(abs_correction, 0.99),
                "accuracy": _mean_or_nan(hard_ok[mask]),
            }
        )
    return rows


def _orientation_route_summary(
    label: str,
    weights: torch.Tensor,
    probs: torch.Tensor,
    corrections: torch.Tensor,
    final_scores: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object]:
    probs = probs.clamp_min(1e-12)
    return {
        "orientation": label,
        "router_entropy_mean": (-(probs * probs.log()).sum(dim=-1)).mean().item(),
        "top_k_margin_mean": _top_margin(weights).mean().item(),
        "experts": _expert_rows(weights, corrections, final_scores, targets),
    }


def _combined_route_summary(
    bvr_weights: torch.Tensor,
    rvb_weights: torch.Tensor,
    bvr_probs: torch.Tensor,
    rvb_probs: torch.Tensor,
    bvr_corrections: torch.Tensor,
    rvb_corrections: torch.Tensor,
    final_scores: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object]:
    selected = (bvr_weights > 0.0) | (rvb_weights > 0.0)
    correction = bvr_corrections - rvb_corrections
    weight_delta = bvr_weights - rvb_weights
    hard_ok = ((final_scores > 0.5) == (targets > 0.5)).float()
    probs = torch.cat([bvr_probs.clamp_min(1e-12), rvb_probs.clamp_min(1e-12)])
    margins = torch.cat([_top_margin(bvr_weights), _top_margin(rvb_weights)])
    rows: list[dict[str, object]] = []
    for expert_id in range(bvr_weights.shape[-1]):
        mask = selected[:, expert_id]
        abs_correction = correction[mask, expert_id].abs()
        rows.append(
            {
                "expert": expert_id,
                "selected_share": mask.float().mean().item(),
                "mean_weight_delta": weight_delta[:, expert_id].mean().item(),
                "mean_abs_weight_delta": weight_delta[:, expert_id].abs().mean().item(),
                "correction_mean": _mean_or_nan(correction[mask, expert_id]),
                "abs_correction_p50": _quantile_or_nan(abs_correction, 0.50),
                "abs_correction_p90": _quantile_or_nan(abs_correction, 0.90),
                "abs_correction_p99": _quantile_or_nan(abs_correction, 0.99),
                "accuracy": _mean_or_nan(hard_ok[mask]),
            }
        )
    return {
        "orientation": "combined",
        "router_entropy_mean": (-(probs * probs.log()).sum(dim=-1)).mean().item(),
        "top_k_margin_mean": margins.mean().item(),
        "experts": rows,
    }


def _route_telemetry(
    route_tensors: dict[str, torch.Tensor],
    final_scores: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object] | None:
    required = (
        "bvr_expert_weights",
        "rvb_expert_weights",
        "bvr_router_probs",
        "rvb_router_probs",
        "bvr_expert_corrections",
        "rvb_expert_corrections",
    )
    if not all(key in route_tensors for key in required):
        return None
    bvr_weights = route_tensors["bvr_expert_weights"].float()
    rvb_weights = route_tensors["rvb_expert_weights"].float()
    bvr_probs = route_tensors["bvr_router_probs"].float()
    rvb_probs = route_tensors["rvb_router_probs"].float()
    bvr_corrections = route_tensors["bvr_expert_corrections"].float()
    rvb_corrections = route_tensors["rvb_expert_corrections"].float()
    return {
        "bvr": _orientation_route_summary(
            "m(b,r)", bvr_weights, bvr_probs, bvr_corrections, final_scores, targets
        ),
        "rvb": _orientation_route_summary(
            "m(r,b)", rvb_weights, rvb_probs, rvb_corrections, final_scores, targets
        ),
        "combined": _combined_route_summary(
            bvr_weights,
            rvb_weights,
            bvr_probs,
            rvb_probs,
            bvr_corrections,
            rvb_corrections,
            final_scores,
            targets,
        ),
    }


def matched_moe_diagnostics(
    baseline_logits: torch.Tensor,
    final_logits: torch.Tensor,
    targets: torch.Tensor,
    route_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    baseline_logits = baseline_logits.float()
    final_logits = final_logits.float()
    targets = targets.float()
    baseline_scores = baseline_logits.sigmoid()
    final_scores = final_logits.sigmoid()
    logit_delta = final_logits - baseline_logits
    score_delta = final_scores - baseline_scores
    route_tensors = route_tensors or {}

    central = (baseline_scores >= 0.40) & (baseline_scores <= 0.60)
    central_comparison = _metric_comparison(
        central,
        baseline_scores,
        baseline_logits,
        final_scores,
        final_logits,
        targets,
    )
    central_metrics = {
        "count": int(central.sum().item()),
        **central_comparison,
    }

    baseline_confidence = (baseline_scores - 0.5).abs() + 0.5
    final_confidence = (final_scores - 0.5).abs() + 0.5
    transition_rows: list[dict[str, object]] = []
    transition_edges = _FOLDED_CONFIDENCE_EDGES
    for i, (lo, hi) in enumerate(zip(transition_edges, transition_edges[1:])):
        base_mask = _band_mask(
            baseline_confidence, lo, hi, i == len(transition_edges) - 2
        )
        counts = []
        for j, (final_lo, final_hi) in enumerate(
            zip(transition_edges, transition_edges[1:])
        ):
            mask = base_mask & _band_mask(
                final_confidence, final_lo, final_hi, j == len(transition_edges) - 2
            )
            counts.append(
                {
                    "band": _band_label(final_lo, final_hi),
                    "count": int(mask.sum().item()),
                }
            )
        transition_rows.append(
            {
                "baseline_confidence": _band_label(lo, hi),
                "count": int(base_mask.sum().item()),
                "final_confidence_counts": counts,
            }
        )

    band_rows: list[dict[str, object]] = []
    bvr_weights = route_tensors.get("bvr_expert_weights")
    rvb_weights = route_tensors.get("rvb_expert_weights")
    for i, (lo, hi) in enumerate(zip(_MATCHED_BAND_EDGES, _MATCHED_BAND_EDGES[1:])):
        mask = _band_mask(
            baseline_scores, lo, hi, i == len(_MATCHED_BAND_EDGES) - 2
        )
        count = int(mask.sum().item())
        if count == 0:
            continue
        abs_logit_delta = logit_delta[mask].abs()
        target_direction = targets[mask] * 2.0 - 1.0
        row: dict[str, object] = {
            "baseline_band": _band_label(lo, hi),
            "count": count,
            "mean_logit_delta": logit_delta[mask].mean().item(),
            "mean_probability_delta": score_delta[mask].mean().item(),
            "abs_logit_delta_p50": abs_logit_delta.quantile(0.50).item(),
            "abs_logit_delta_p90": abs_logit_delta.quantile(0.90).item(),
            "abs_logit_delta_p99": abs_logit_delta.quantile(0.99).item(),
            "sign_agreement": ((logit_delta[mask] * target_direction) > 0.0)
            .float()
            .mean()
            .item(),
            **_metric_comparison(
                mask,
                baseline_scores,
                baseline_logits,
                final_scores,
                final_logits,
                targets,
            ),
        }
        if bvr_weights is not None and rvb_weights is not None:
            bvr = bvr_weights[mask].float()
            rvb = rvb_weights[mask].float()
            combined = (bvr + rvb) * 0.5
            row["bvr_avg_weight"] = bvr.mean(dim=0).tolist()
            row["rvb_avg_weight"] = rvb.mean(dim=0).tolist()
            row["avg_selected_weight"] = combined.mean(dim=0).tolist()
            row["expert_utilization"] = (
                ((bvr > 0.0) | (rvb > 0.0)).float().mean(dim=0).tolist()
            )
            row["mean_abs_weight_delta"] = (bvr - rvb).abs().mean(dim=0).tolist()
        band_rows.append(row)

    output: dict[str, object] = {
        "central": central_metrics,
        "folded_transition": transition_rows,
        "baseline_band_rows": band_rows,
    }
    telemetry = _route_telemetry(route_tensors, final_scores, targets)
    if telemetry is not None:
        output["route_telemetry"] = telemetry
    return output


def log_matched_moe_diagnostics(
    logger: logging.Logger,
    split_name: str,
    rows: dict[str, object],
) -> None:
    central = cast(dict[str, object], rows["central"])
    baseline = cast(dict[str, float], central["baseline"])
    final = cast(dict[str, float], central["final"])
    logger.info(
        "%s matched MoE central n=%d | BCE %.4e -> %.4e | Brier %.4e -> %.4e | ECE %.4e -> %.4e | AUC %.4e -> %.4e | acc %.4e -> %.4e",
        split_name,
        cast(int, central["count"]),
        baseline["bce"],
        final["bce"],
        baseline["brier"],
        final["brier"],
        baseline["ece"],
        final["ece"],
        baseline["auc"],
        final["auc"],
        baseline["accuracy"],
        final["accuracy"],
    )
