# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false

"""Audit how much the HGNN uses 1v1/2vx relationship priors.

Run:
    uv run python -m app.ml.experiments.hgnn_relationship_audit --split val
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from app.ml.cache_layout import ARRAY_FILES
from app.ml.config import CACHE_DIR, DatasetConfig, TrainConfig
from app.ml.dataset import SplitData, load_splits
from app.ml.hgnn_model import (
    LOGIT_EPS,
    HGNNWinModel,
    TEAM_PAIRS,
    _logit,
    build_hgnn_inputs,
    load_hgnn_model,
    relationship_logit_features,
    resolve_device,
    swap_hgnn_inputs,
)
from app.ml.train import (
    RawTensorSplit,
    _binary_auc,
    _cache_raw_tensor_split,
    _ece,
    _nll,
    _predict_hgnn,
)


DEFAULT_OUT_DIR = Path("app/ml/data/experiments/hgnn_relationship_audit")
RELATIONSHIP_GRAD_KEYS: tuple[str, ...] = (
    "delta_logit_1v1",
    "delta_logit_2vx",
)
PATH_GRADIENT_PATHS: tuple[str, ...] = (
    "full",
    "core_main",
    "residual_slot",
    "prior_shortcut",
)
MODULE_GRADIENT_PREFIXES: dict[str, str] = {
    "phi_1vx": "phi.1vx.",
    "residual_head": "residual_head.",
    "prior_shortcut": "prior_shortcut.",
    "head": "head.",
    "identity": "identity.",
    "node_init": "node_init.",
}
COUNT_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("zero_count", 0.0, 0.0),
    ("sparse_1_4", 1.0, 4.0),
    ("low_5_19", 5.0, 19.0),
    ("mid_20_49", 20.0, 49.0),
    ("confident_50_plus", 50.0, math.inf),
)


@dataclass(frozen=True)
class AuditConfig:
    split: str = "val"
    cache_dir: Path = CACHE_DIR
    model_path: Path = TrainConfig().model_path
    out_dir: Path = DEFAULT_OUT_DIR
    device: str = "auto"
    batch_size: int = 8192
    max_games: int | None = None
    examples: int = 12
    seed: int = 0
    include_path_grads: bool = False
    grad_batch_size: int = 512
    grad_split: str = "train"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _limit_split(split: SplitData, max_games: int | None) -> SplitData:
    if max_games is None or split.blue_win.size <= max_games:
        return split
    n = int(max_games)
    return SplitData(
        win_rate=split.win_rate[:n],
        matchup_1v1=split.matchup_1v1[:n],
        synergy_2vx=split.synergy_2vx[:n],
        p1_cnt=split.p1_cnt[:n],
        m1v1_cnt=split.m1v1_cnt[:n],
        s2vx_cnt=split.s2vx_cnt[:n],
        m1v1_eff_n=split.m1v1_eff_n[:n] if split.m1v1_eff_n is not None else None,
        s2vx_eff_n=split.s2vx_eff_n[:n] if split.s2vx_eff_n is not None else None,
        blue_win=split.blue_win[:n],
        champion_id=split.champion_id[:n] if split.champion_id is not None else None,
        build_id=split.build_id[:n] if split.build_id is not None else None,
    )


def _composite_rates(win_rate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blue, red = win_rate[:, :5], win_rate[:, 5:]
    matchup = (0.5 + (blue[:, :, None] - red[:, None, :]) / 2.0).reshape(-1, 25)
    blue_pairs = torch.stack(
        [0.5 * (blue[:, a] + blue[:, b]) for a, b in TEAM_PAIRS],
        dim=1,
    )
    red_pairs = torch.stack(
        [0.5 * (red[:, a] + red[:, b]) for a, b in TEAM_PAIRS],
        dim=1,
    )
    return matchup, torch.cat([blue_pairs, red_pairs], dim=1)


def _neutralized(raw: RawTensorSplit, *, onev1: bool, twovx: bool, missing: bool) -> RawTensorSplit:
    matchup, synergy = _composite_rates(raw.win_rate)
    zero_1v1 = torch.zeros_like(raw.m1v1_cnt)
    zero_2vx = torch.zeros_like(raw.s2vx_cnt)
    return replace(
        raw,
        matchup_1v1=matchup if onev1 else raw.matchup_1v1,
        synergy_2vx=synergy if twovx else raw.synergy_2vx,
        m1v1_cnt=zero_1v1 if onev1 and missing else raw.m1v1_cnt,
        s2vx_cnt=zero_2vx if twovx and missing else raw.s2vx_cnt,
    )


def _logit_np(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob.astype(np.float64, copy=False), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p / (1.0 - p))


def _metric_suite(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "n": int(targets.size),
        "acc": float(np.mean((scores >= 0.5) == (targets > 0.5))),
        "auc": _binary_auc(scores, targets),
        "nll": _nll(scores, targets),
        "ece": _ece(scores, targets),
    }
    if baseline is None:
        out["delta_nll"] = 0.0
        out["mean_abs_effect"] = 0.0
        out["p95_abs_effect"] = 0.0
    else:
        effect = _logit_np(scores) - _logit_np(baseline)
        out["delta_nll"] = float(out["nll"]) - _nll(baseline, targets)
        out["mean_abs_effect"] = float(np.mean(np.abs(effect)))
        out["p95_abs_effect"] = float(np.quantile(np.abs(effect), 0.95))
    return out


def _confidence(count: torch.Tensor, strength: float) -> torch.Tensor:
    count_f = count.clamp_min(0.0)
    return count_f / (count_f + float(strength)).clamp_min(1.0e-12)


def _residual_scores(raw: RawTensorSplit, strength: float) -> dict[str, np.ndarray]:
    features = relationship_logit_features(
        mu_1vx=raw.win_rate,
        mu_2vx=raw.synergy_2vx,
        mu_1v1=raw.matchup_1v1,
    )
    conf_1v1 = _confidence(raw.m1v1_cnt, strength)
    conf_2vx = _confidence(raw.s2vx_cnt, strength)
    onev1 = features["delta_logit_1v1"]
    twovx = features["delta_logit_2vx"]
    signed_2vx = torch.cat([twovx[:, :10], -twovx[:, 10:]], dim=1)
    matchup_adv = (onev1 * conf_1v1).sum(dim=1) / conf_1v1.sum(dim=1).clamp_min(1.0)
    synergy_adv = (signed_2vx * conf_2vx).sum(dim=1) / conf_2vx.sum(dim=1).clamp_min(1.0)
    abs_residual = torch.cat([onev1.abs(), twovx.abs()], dim=1).mean(dim=1)
    rel_conf = torch.cat([conf_1v1, conf_2vx], dim=1).mean(dim=1)
    return {
        "matchup_adv": matchup_adv.detach().cpu().numpy().astype(np.float64),
        "synergy_adv": synergy_adv.detach().cpu().numpy().astype(np.float64),
        "combined_adv": (matchup_adv + synergy_adv).detach().cpu().numpy().astype(np.float64),
        "abs_residual": abs_residual.detach().cpu().numpy().astype(np.float64),
        "relationship_confidence": rel_conf.detach().cpu().numpy().astype(np.float64),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _quantile_buckets(
    name: str,
    values: np.ndarray,
    targets: np.ndarray,
    original: np.ndarray,
    neutral: np.ndarray,
    effect: np.ndarray,
    *,
    n_buckets: int = 5,
) -> list[dict[str, float | int | str]]:
    qs = np.quantile(values, np.linspace(0.0, 1.0, n_buckets + 1))
    rows: list[dict[str, float | int | str]] = []
    for i in range(n_buckets):
        lo, hi = qs[i], qs[i + 1]
        mask = values >= lo if i == n_buckets - 1 else (values >= lo) & (values < hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bucket": f"{name}_q{i + 1}",
                "n": int(mask.sum()),
                "lo": float(lo),
                "hi": float(hi),
                "blue_win_rate": float(targets[mask].mean()),
                "mean_pred": float(original[mask].mean()),
                "mean_neutral_pred": float(neutral[mask].mean()),
                "mean_model_rel_effect_logit": float(effect[mask].mean()),
                "nll": _nll(original[mask], targets[mask]),
                "neutral_nll": _nll(neutral[mask], targets[mask]),
            }
        )
    return rows


def _residual_summary(
    raw: RawTensorSplit,
    targets: np.ndarray,
    original: np.ndarray,
    neutral: np.ndarray,
    *,
    strength: float,
) -> dict[str, Any]:
    scores = _residual_scores(raw, strength)
    effect = _logit_np(original) - _logit_np(neutral)
    combined = scores["combined_adv"]
    abs_residual = scores["abs_residual"]
    rel_conf = scores["relationship_confidence"]
    return {
        "scores": {
            "matchup_adv_auc": _binary_auc(scores["matchup_adv"], targets),
            "synergy_adv_auc": _binary_auc(scores["synergy_adv"], targets),
            "combined_adv_auc": _binary_auc(combined, targets),
            "abs_residual_auc": _binary_auc(abs_residual, targets),
            "relationship_confidence_auc": _binary_auc(rel_conf, targets),
            "model_effect_auc": _binary_auc(effect, targets),
            "corr_combined_adv_vs_model_effect": _safe_corr(combined, effect),
            "corr_abs_residual_vs_abs_model_effect": _safe_corr(abs_residual, np.abs(effect)),
        },
        "combined_adv_buckets": _quantile_buckets(
            "combined_adv", combined, targets, original, neutral, effect
        ),
        "confidence_buckets": _quantile_buckets(
            "rel_conf", rel_conf, targets, original, neutral, effect
        ),
    }


def _edge_inputs(raw: RawTensorSplit, strength: float, device: str) -> dict[str, torch.Tensor]:
    if raw.champion_id is None or raw.build_id is None:
        raise ValueError("HGNN audit requires champion_id/build_id; rebuild the cache.")
    return build_hgnn_inputs(
        champion_id=raw.champion_id,
        build_id=raw.build_id,
        win_rate=raw.win_rate,
        matchup_1v1=raw.matchup_1v1,
        synergy_2vx=raw.synergy_2vx,
        p1_cnt=raw.p1_cnt,
        m1v1_cnt=raw.m1v1_cnt,
        s2vx_cnt=raw.s2vx_cnt,
        strength=strength,
        device=device,
    )


def relationship_contract(cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Report which relationship families exist in the cache/model contract."""

    cache_arrays = set(ARRAY_FILES)

    def cache_present(*names: str) -> bool:
        return all(name in cache_arrays and (cache_dir / ARRAY_FILES[name]).exists() for name in names)

    return {
        "1v1": {
            "present_in_cache": cache_present("matchup_1v1", "m1v1_cnt"),
            "present_in_forward": True,
            "cache_arrays": ["matchup_1v1", "m1v1_cnt"],
            "forward_inputs": ["mu_1v1", "delta_logit_1v1", "conf_1v1", "missing_1v1"],
        },
        "2vx": {
            "present_in_cache": cache_present("synergy_2vx", "s2vx_cnt"),
            "present_in_forward": True,
            "cache_arrays": ["synergy_2vx", "s2vx_cnt"],
            "forward_inputs": ["mu_2vx", "delta_logit_2vx", "conf_2vx", "missing_2vx"],
        },
    }


def _clone_inputs_for_gradients(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        item = value.detach().clone()
        if key in RELATIONSHIP_GRAD_KEYS:
            item.requires_grad_(True)
        cloned[key] = item
    return cloned


def _gradient_stats(inputs: dict[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for key in RELATIONSHIP_GRAD_KEYS:
        tensor = inputs[key]
        grad = tensor.grad
        if grad is None:
            rows[key] = {
                "abs_mean": 0.0,
                "l2": 0.0,
                "max_abs": 0.0,
                "zero_fraction": 1.0,
                "n": int(tensor.numel()),
            }
            continue
        abs_grad = grad.detach().abs()
        rows[key] = {
            "abs_mean": float(abs_grad.mean().cpu()),
            "l2": float(grad.detach().norm().cpu()),
            "max_abs": float(abs_grad.max().cpu()),
            "zero_fraction": float((abs_grad == 0.0).to(torch.float32).mean().cpu()),
            "n": int(grad.numel()),
        }
    return rows


def _core_readout_parts(
    model: HGNNWinModel,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    c = model.config
    h0 = model.identity(inputs["champion_id"], inputs["build_id"])
    phi_node = model.phi["1vx"](
        _logit(inputs["mu_1vx"], c.logit_clip),
        inputs["var_1vx"],
        inputs.get("conf_1vx"),
        inputs.get("log_count_1vx"),
        inputs.get("missing_1vx"),
    )
    h = model.node_norm(model.node_init(torch.cat([h0, phi_node], dim=-1)))
    return model._readout(h[:, :5]), model._readout(h[:, 5:])


def _path_logits(
    model: HGNNWinModel,
    inputs: dict[str, torch.Tensor],
    path: str,
) -> torch.Tensor:
    if path == "full":
        return model(**inputs)["final_logit"]
    if path == "prior_shortcut":
        return model._prior_shortcut_logit(
            mu_1vx=inputs["mu_1vx"],
            delta_logit_2vx=inputs["delta_logit_2vx"],
            delta_logit_1v1=inputs["delta_logit_1v1"],
            conf_2vx=inputs.get("conf_2vx"),
            conf_1v1=inputs.get("conf_1v1"),
        )

    a, b = _core_readout_parts(model, inputs)
    core_parts = [a, b, a - b, a * b]
    if path == "core_main":
        core_parts.append(torch.zeros_like(a))
        return model.head(torch.cat(core_parts, dim=-1)).squeeze(-1)
    if path == "residual_slot":
        residual = model._residual_readout(
            delta_logit_2vx=inputs["delta_logit_2vx"],
            delta_logit_1v1=inputs["delta_logit_1v1"],
            conf_2vx=inputs.get("conf_2vx"),
            conf_1v1=inputs.get("conf_1v1"),
            missing_2vx=inputs.get("missing_2vx"),
            missing_1v1=inputs.get("missing_1v1"),
        )
        detached_core = [part.detach() for part in core_parts]
        head_parts = [*detached_core, residual]
        return model.head(torch.cat(head_parts, dim=-1)).squeeze(-1)
    raise ValueError(f"Unsupported path gradient target: {path}")


def path_gradient_diagnostics(
    model: HGNNWinModel,
    inputs: dict[str, torch.Tensor],
    *,
    paths: tuple[str, ...] = PATH_GRADIENT_PATHS,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Measure final-logit sensitivity to explicit relationship logit tensors."""

    was_training = model.training
    model.eval()
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for path in paths:
        grad_inputs = _clone_inputs_for_gradients(inputs)
        model.zero_grad(set_to_none=True)
        _path_logits(model, grad_inputs, path).mean().backward()
        out[path] = _gradient_stats(grad_inputs)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return out


def module_gradient_diagnostics(
    model: HGNNWinModel,
    inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    seed: int = 0,
) -> dict[str, dict[str, float | int]]:
    """Measure parameter-gradient pressure from one training-style BCE batch."""

    groups = {
        name: {"l2sq": 0.0, "max_abs": 0.0, "params": 0, "grad_params": 0}
        for name in MODULE_GRADIENT_PREFIXES
    }
    groups["other"] = {"l2sq": 0.0, "max_abs": 0.0, "params": 0, "grad_params": 0}
    was_training = model.training
    torch.manual_seed(seed)
    model.train()
    model.zero_grad(set_to_none=True)
    loss_fn = nn.BCEWithLogitsLoss()
    direct_loss = loss_fn(model(**inputs)["final_logit"], labels)
    (0.5 * direct_loss).backward()
    swapped_loss = loss_fn(model(**swap_hgnn_inputs(inputs))["final_logit"], 1.0 - labels)
    (0.5 * swapped_loss).backward()
    for param_name, param in model.named_parameters():
        group_name = "other"
        for candidate, prefix in MODULE_GRADIENT_PREFIXES.items():
            if param_name.startswith(prefix):
                group_name = candidate
                break
        row = groups[group_name]
        row["params"] += int(param.numel())
        if param.grad is None:
            continue
        grad = param.grad.detach()
        row["grad_params"] += int(param.numel())
        row["l2sq"] += float((grad * grad).sum().cpu())
        row["max_abs"] = max(float(row["max_abs"]), float(grad.abs().max().cpu()))
    out: dict[str, dict[str, float | int]] = {}
    for group_name, row in groups.items():
        out[group_name] = {
            "l2": math.sqrt(float(row["l2sq"])),
            "max_abs": float(row["max_abs"]),
            "params": int(row["params"]),
            "grad_params": int(row["grad_params"]),
        }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return out


def _gradient_raw_batch(
    *,
    split_name: str,
    dataset_cfg: DatasetConfig,
    max_games: int | None,
    batch_size: int,
    device: str,
) -> RawTensorSplit:
    split = _limit_split(load_splits(dataset_cfg, require_counts=True)[split_name], max_games)
    n = min(int(batch_size), int(split.blue_win.size))
    if n <= 0:
        raise ValueError(f"Gradient split {split_name!r} is empty")
    return _cache_raw_tensor_split(split_name, _limit_split(split, n), device=device)


def _phi_gate_tensor(
    model: HGNNWinModel,
    key: str,
    var: torch.Tensor,
    count: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    precision = 1.0 / (1.0 + var)
    conf = _confidence(count, strength)
    gate_input = torch.stack(
        [precision, conf, torch.log1p(count.clamp_min(0.0)), (count <= 0.0).to(count.dtype)],
        dim=-1,
    )
    phi = cast(Any, model.phi[key])
    return torch.sigmoid(phi.gate(gate_input)).mean(dim=-1)


def _bucket_stats(
    *,
    name: str,
    count: torch.Tensor,
    eff_n: torch.Tensor,
    raw_logit: torch.Tensor,
    gate: torch.Tensor,
    phi_norm: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    count_np = count.detach().cpu().numpy().reshape(-1)
    eff_np = eff_n.detach().cpu().numpy().reshape(-1)
    raw_np = raw_logit.detach().cpu().numpy().reshape(-1)
    gate_np = gate.detach().cpu().numpy().reshape(-1)
    norm_np = phi_norm.detach().cpu().numpy().reshape(-1)
    for bucket, lo, hi in COUNT_BUCKETS:
        mask = count_np >= lo if math.isinf(hi) else (count_np >= lo) & (count_np <= hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bucket": bucket,
                "n_edges": int(mask.sum()),
                "mean_count": float(count_np[mask].mean()),
                "mean_eff_n": float(eff_np[mask].mean()),
                "mean_raw_logit": float(raw_np[mask].mean()),
                "mean_gate": float(gate_np[mask].mean()),
                "mean_phi_norm": float(norm_np[mask].mean()),
                "edge_type": name,
            }
        )
    return rows


def _phi_diagnostics(
    model: HGNNWinModel,
    raw: RawTensorSplit,
    *,
    strength: float,
    device: str,
    batch_size: int,
) -> dict[str, list[dict[str, float | int | str]]]:
    accum: dict[str, list[dict[str, np.ndarray]]] = {"1vx": []}
    n_rows = raw.blue_win.numel()
    model.eval()
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            part = RawTensorSplit(
                win_rate=raw.win_rate[start : start + batch_size],
                matchup_1v1=raw.matchup_1v1[start : start + batch_size],
                synergy_2vx=raw.synergy_2vx[start : start + batch_size],
                p1_cnt=raw.p1_cnt[start : start + batch_size],
                m1v1_cnt=raw.m1v1_cnt[start : start + batch_size],
                s2vx_cnt=raw.s2vx_cnt[start : start + batch_size],
                blue_win=raw.blue_win[start : start + batch_size],
                champion_id=raw.champion_id[start : start + batch_size] if raw.champion_id is not None else None,
                build_id=raw.build_id[start : start + batch_size] if raw.build_id is not None else None,
            )
            inputs = _edge_inputs(part, strength, device)
            phi_1vx = model.phi["1vx"](
                torch.logit(inputs["mu_1vx"].clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)),
                inputs["var_1vx"],
                inputs.get("conf_1vx"),
                inputs.get("log_count_1vx"),
                inputs.get("missing_1vx"),
            )
            accum["1vx"].append(
                {
                    "count": inputs["log_count_1vx"].expm1().detach().cpu().numpy(),
                    "eff_n": inputs["log_count_1vx"].expm1().detach().cpu().numpy(),
                    "raw": torch.logit(inputs["mu_1vx"].clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)).detach().cpu().numpy(),
                    "gate": _phi_gate_tensor(
                        model,
                        "1vx",
                        inputs["var_1vx"],
                        part.p1_cnt,
                        strength=strength,
                    ).detach().cpu().numpy(),
                    "norm": phi_1vx.norm(dim=-1).detach().cpu().numpy(),
                }
            )
    out: dict[str, list[dict[str, float | int | str]]] = {}
    for name, chunks in accum.items():
        out[name] = _bucket_stats(
            name=name,
            count=torch.as_tensor(np.concatenate([c["count"] for c in chunks], axis=0)),
            eff_n=torch.as_tensor(np.concatenate([c["eff_n"] for c in chunks], axis=0)),
            raw_logit=torch.as_tensor(np.concatenate([c["raw"] for c in chunks], axis=0)),
            gate=torch.as_tensor(np.concatenate([c["gate"] for c in chunks], axis=0)),
            phi_norm=torch.as_tensor(np.concatenate([c["norm"] for c in chunks], axis=0)),
        )
    return out


def _examples(
    raw: RawTensorSplit,
    targets: np.ndarray,
    original: np.ndarray,
    neutral: np.ndarray,
    *,
    n: int,
) -> list[dict[str, float | int]]:
    effect = _logit_np(original) - _logit_np(neutral)
    order = np.argsort(-np.abs(effect))[:n]
    return [
        {
            "split_index": int(i),
            "blue_win": int(targets[i]),
            "original_pred": float(original[i]),
            "neutral_all_pred": float(neutral[i]),
            "relationship_effect_logit": float(effect[i]),
        }
        for i in order
    ]


def _write_report_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# HGNN Relationship Audit",
        "",
        f"Split: `{report['config']['split']}`",
        f"Rows: `{report['n_rows']}`",
        f"Device: `{report['device']}`",
        "",
        "## Variant Metrics",
        "",
        "| variant | acc | auc | nll | delta_nll | mean_abs_effect | p95_abs_effect |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["variant_metrics"].items():
        lines.append(
            f"| {name} | {row['acc']:.4f} | {row['auc']:.4f} | {row['nll']:.4f} | "
            f"{row['delta_nll']:.4f} | {row['mean_abs_effect']:.4f} | {row['p95_abs_effect']:.4f} |"
        )
    lines.extend(["", "## Residual Signal", "", "| metric | value |", "| --- | ---: |"])
    for key, value in report["residual_summary"]["scores"].items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend(["", "## Phi Gate Buckets"])
    for edge, rows in report["phi_diagnostics"].items():
        lines.extend(
            [
                "",
                f"### {edge}",
                "",
                "| bucket | n_edges | mean_count | mean_eff_n | mean_gate | mean_phi_norm |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['bucket']} | {row['n_edges']} | {row['mean_count']:.4f} | "
                f"{row['mean_eff_n']:.4f} | {row['mean_gate']:.4f} | {row['mean_phi_norm']:.4f} |"
            )
    lines.extend(["", "## Relationship Contract", "", "| family | cache | forward |", "| --- | ---: | ---: |"])
    for family, row in report["relationship_contract"].items():
        lines.append(
            f"| {family} | {str(row['present_in_cache']).lower()} | "
            f"{str(row['present_in_forward']).lower()} |"
        )
    if report.get("path_gradients"):
        lines.extend(["", "## Path Gradients"])
        for path_name, rows in report["path_gradients"].items():
            lines.extend(
                [
                    "",
                    f"### {path_name}",
                    "",
                    "| input | abs_mean | l2 | max_abs | zero_fraction |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for key in RELATIONSHIP_GRAD_KEYS:
                row = rows[key]
                lines.append(
                    f"| {key} | {row['abs_mean']:.3e} | {row['l2']:.3e} | "
                    f"{row['max_abs']:.3e} | {row['zero_fraction']:.4f} |"
                )
    if report.get("module_gradients"):
        lines.extend(
            [
                "",
                "## Module Gradients",
                "",
                "| module | l2 | max_abs | grad_params | params |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for module, row in report["module_gradients"].items():
            lines.append(
                f"| {module} | {row['l2']:.3e} | {row['max_abs']:.3e} | "
                f"{row['grad_params']} | {row['params']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(config: AuditConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    dataset_cfg = DatasetConfig(cache_dir=config.cache_dir)
    split = _limit_split(load_splits(dataset_cfg, require_counts=True)[config.split], config.max_games)
    model, model_config, strength = load_hgnn_model(config.model_path, device=device)
    raw = _cache_raw_tensor_split(config.split, split, device=device)
    targets = split.blue_win.astype(np.float64, copy=False)
    grad_raw = None
    grad_inputs = None
    if config.include_path_grads:
        grad_raw = _gradient_raw_batch(
            split_name=config.grad_split,
            dataset_cfg=dataset_cfg,
            max_games=config.max_games,
            batch_size=config.grad_batch_size,
            device=device,
        )
        grad_inputs = _edge_inputs(grad_raw, strength, device)

    variants = {
        "original": raw,
        "neutral_1v1": _neutralized(raw, onev1=True, twovx=False, missing=False),
        "neutral_2vx": _neutralized(raw, onev1=False, twovx=True, missing=False),
        "neutral_all": _neutralized(raw, onev1=True, twovx=True, missing=False),
        "missing_1v1": _neutralized(raw, onev1=True, twovx=False, missing=True),
        "missing_2vx": _neutralized(raw, onev1=False, twovx=True, missing=True),
        "missing_all": _neutralized(raw, onev1=True, twovx=True, missing=True),
    }
    predictions = {
        name: _predict_hgnn(
            model,
            variant,
            batch_size=config.batch_size,
            strength=strength,
            device=device,
        )
        for name, variant in variants.items()
    }
    original = predictions["original"]
    variant_metrics = {
        name: _metric_suite(scores, targets, baseline=None if name == "original" else original)
        for name, scores in predictions.items()
    }

    report = {
        "confidence_strength": strength,
        "config": _jsonable(config),
        "device": device,
        "model_config": _jsonable(model_config),
        "n_rows": int(targets.size),
        "relationship_contract": relationship_contract(config.cache_dir),
        "variant_metrics": variant_metrics,
        "residual_summary": _residual_summary(
            raw,
            targets,
            original,
            predictions["neutral_all"],
            strength=strength,
        ),
        "phi_diagnostics": _phi_diagnostics(
            model,
            raw,
            strength=strength,
            device=device,
            batch_size=config.batch_size,
        ),
        "examples": _examples(
            raw,
            targets,
            original,
            predictions["neutral_all"],
            n=config.examples,
        ),
    }
    if config.include_path_grads:
        if grad_raw is None or grad_inputs is None:
            raise RuntimeError("Gradient inputs were not prepared")
        report["path_gradients"] = path_gradient_diagnostics(model, grad_inputs)
        report["module_gradients"] = module_gradient_diagnostics(
            model,
            grad_inputs,
            grad_raw.blue_win,
            seed=config.seed,
        )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report_md(report, config.out_dir / "report.md")
    return report


def _parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--model-path", type=Path, default=TrainConfig().model_path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-path-grads", action="store_true")
    parser.add_argument("--grad-batch-size", type=int, default=512)
    parser.add_argument("--grad-split", choices=("train", "val", "test"), default="train")
    return AuditConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run_audit(_parse_args())
    metrics = report["variant_metrics"]
    print(
        "HGNN relationship audit complete: "
        f"original_auc={metrics['original']['auc']:.4f} "
        f"neutral_all_auc={metrics['neutral_all']['auc']:.4f} "
        f"delta_nll={metrics['neutral_all']['delta_nll']:.4f}"
    )
    print(f"Wrote {report['config']['out_dir']}/report.json and report.md")


if __name__ == "__main__":
    main()
