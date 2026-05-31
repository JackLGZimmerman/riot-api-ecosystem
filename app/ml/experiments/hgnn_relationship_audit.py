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

from app.ml.config import CACHE_DIR, DatasetConfig, TrainConfig
from app.ml.dataset import SplitData, load_splits
from app.ml.hgnn_model import (
    LOGIT_EPS,
    TEAM_PAIRS,
    HGNNWinModel,
    build_hgnn_inputs,
    load_hgnn_model,
    relationship_logit_features,
    resolve_device,
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
        m1v1_eff_n=zero_1v1 if onev1 and missing else raw.m1v1_eff_n,
        s2vx_eff_n=zero_2vx if twovx and missing else raw.s2vx_eff_n,
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
        m1v1_eff_n=raw.m1v1_eff_n,
        s2vx_eff_n=raw.s2vx_eff_n,
        strength=strength,
        device=device,
    )


def _phi_gate_tensor(
    model: HGNNWinModel,
    key: str,
    var: torch.Tensor,
    count: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    precision = 1.0 / (1.0 + var)
    if model.config.use_count_features:
        conf = _confidence(count, strength)
        gate_input = torch.stack(
            [precision, conf, torch.log1p(count.clamp_min(0.0)), (count <= 0.0).to(count.dtype)],
            dim=-1,
        )
    else:
        gate_input = precision.unsqueeze(-1)
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
    accum: dict[str, list[dict[str, np.ndarray]]] = {"1vx": [], "1v1": [], "2vx": []}
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
                m1v1_eff_n=raw.m1v1_eff_n[start : start + batch_size],
                s2vx_eff_n=raw.s2vx_eff_n[start : start + batch_size],
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

            phi_1v1 = model._edge_phi(
                "onev1",
                inputs["mu_1v1"],
                inputs["var_1v1"],
                inputs.get("joint_logit_1v1"),
                inputs.get("expected_logit_1v1"),
                inputs.get("delta_logit_1v1"),
                inputs.get("conf_1v1"),
                inputs.get("log_count_1v1"),
                inputs.get("missing_1v1"),
            )
            accum["1v1"].append(
                {
                    "count": part.m1v1_cnt.detach().cpu().numpy(),
                    "eff_n": part.m1v1_eff_n.detach().cpu().numpy(),
                    "raw": inputs["delta_logit_1v1"].detach().cpu().numpy(),
                    "gate": _phi_gate_tensor(
                        model,
                        "onev1",
                        inputs["var_1v1"],
                        part.m1v1_cnt,
                        strength=strength,
                    ).detach().cpu().numpy(),
                    "norm": phi_1v1.norm(dim=-1).mean(dim=-1).detach().cpu().numpy(),
                }
            )

            phi_2vx = model._edge_phi(
                "twovx",
                inputs["mu_2vx"],
                inputs["var_2vx"],
                inputs.get("joint_logit_2vx"),
                inputs.get("expected_logit_2vx"),
                inputs.get("delta_logit_2vx"),
                inputs.get("conf_2vx"),
                inputs.get("log_count_2vx"),
                inputs.get("missing_2vx"),
            )
            accum["2vx"].append(
                {
                    "count": part.s2vx_cnt.detach().cpu().numpy(),
                    "eff_n": part.s2vx_eff_n.detach().cpu().numpy(),
                    "raw": inputs["delta_logit_2vx"].detach().cpu().numpy(),
                    "gate": _phi_gate_tensor(
                        model,
                        "twovx",
                        inputs["var_2vx"],
                        part.s2vx_cnt,
                        strength=strength,
                    ).detach().cpu().numpy(),
                    "norm": phi_2vx.norm(dim=-1).mean(dim=-1).detach().cpu().numpy(),
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(config: AuditConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    dataset_cfg = DatasetConfig(cache_dir=config.cache_dir)
    split = _limit_split(load_splits(dataset_cfg, require_counts=True)[config.split], config.max_games)
    model, model_config, strength = load_hgnn_model(config.model_path, device=device)
    raw = _cache_raw_tensor_split(config.split, split, device=device)
    targets = split.blue_win.astype(np.float64, copy=False)

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
