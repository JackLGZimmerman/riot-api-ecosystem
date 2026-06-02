"""Win-rate validation of the HGNN context-atlas head.

`context_probes.py` checks the *direction* of the context head on synthetic
drafts. This checks the *magnitude*: how much of the win-rate that the
draft-time context can explain does the head actually realize, on real
train/val/test games?

Method (residual-aware, the only honest framing):

1. ``base = final_logit - context_logit`` is the model with the context head
   removed (the head is an additive, zero-init residual, so this is exact).
2. ``ceiling`` = an independent draft-safe context extractor (linear + a small
   MLP) fit over the frozen ``base`` offset. It is the best win-rate a
   draft-time context signal can add on top of the context-free model.
3. The head is "good" when ``full ≈ ceiling`` (it realizes the achievable
   context signal) and when, binned by a context axis, its predicted WR
   correction matches the WR the base model actually misses.

The raw empirical WR swing across context bins (~18pp) is NOT the target: most
of it is owned by the win-rate prior. The target is the residual the base model
leaves on the table.

Run with:
    python -m app.ml.context_wr_validation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.classification.embeddings.config import (
    CONTEXT_ARMOR_INDEX,
    CONTEXT_DAMAGE_PRESSURE_INDEX,
    CONTEXT_HEAL_SHIELD_INDEX,
    CONTEXT_MR_INDEX,
    CONTEXT_TAKEN_INDEX,
)
from app.core.logging.logger import setup_logging_config
from app.ml.config import DatasetConfig, TrainConfig
from app.ml.dataset import SplitData, load_splits
from app.ml.hgnn_model import build_hgnn_inputs, load_hgnn_model, resolve_device

logger = logging.getLogger(__name__)

PHYS, MAGIC = 0, 1
ARMOR, MR = CONTEXT_ARMOR_INDEX, CONTEXT_MR_INDEX
DMG, TAKEN, HEAL = CONTEXT_DAMAGE_PRESSURE_INDEX, CONTEXT_TAKEN_INDEX, CONTEXT_HEAL_SHIELD_INDEX
INTERP = 14
SPLITS = ("train", "val", "test")


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _nll(logit: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(1.0 / (1.0 + np.exp(-logit)), 1e-9, 1.0 - 1e-9)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _team_summary(ctx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """mean, damage-weighted mean, per-axis max, per-axis variance over 5 players."""
    mean = ctx.mean(1)
    weight = np.clip(ctx[:, :, DMG : DMG + 1], 0.0, None)
    denom = np.clip(weight.sum(1), 1e-6, None)
    wmean = (ctx * weight).sum(1) / denom
    return mean, wmean, ctx.max(1), ctx.var(1)


def _draft_features(context: np.ndarray) -> np.ndarray:
    """Antisymmetric draft-safe context features (blue minus red).

    Mirrors the head's interpretable axes/products and adds the extreme-tail
    extraction (per-axis max/variance) the head's mean summaries cannot see, so
    the ceiling is a fair upper bound on draft-time context extraction.
    """
    blue, red = context[:, :5, :INTERP], context[:, 5:, :INTERP]
    bm, bw, bmx, bvar = _team_summary(blue)
    rm, rw, rmx, rvar = _team_summary(red)

    def prod(axis: int, enemy_axis: int) -> np.ndarray:
        return bm[:, axis] * rw[:, enemy_axis] - rm[:, axis] * bw[:, enemy_axis]

    products = np.stack(
        [
            prod(ARMOR, PHYS),
            prod(MR, MAGIC),
            bm[:, ARMOR] * (rw[:, PHYS] - rw[:, MAGIC]) - rm[:, ARMOR] * (bw[:, PHYS] - bw[:, MAGIC]),
            bm[:, MR] * (rw[:, MAGIC] - rw[:, PHYS]) - rm[:, MR] * (bw[:, MAGIC] - bw[:, PHYS]),
            prod(TAKEN, DMG),
            bm[:, DMG] * rw[:, HEAL] - rm[:, DMG] * bw[:, HEAL],
        ],
        axis=1,
    )
    tail = []
    for axis in (PHYS, MAGIC, HEAL, TAKEN, DMG):
        tail.append(bmx[:, axis] - rmx[:, axis])
        tail.append(bvar[:, axis] - rvar[:, axis])
    lowrank = context[:, :5, INTERP:].mean(1) - context[:, 5:, INTERP:].mean(1)
    return np.concatenate([bm - rm, products, np.stack(tail, 1), lowrank], 1).astype(np.float32)


def _predict(model, sp: SplitData, *, strength: float, device: str) -> tuple[np.ndarray, np.ndarray]:
    full, ctx_logit = [], []
    conditioned = getattr(model, "identity_conditioned_context_enabled", False)
    n = sp.blue_win.size
    for i in range(0, n, 32768):
        s = slice(i, i + 32768)
        raw = sp.identity_context_raw[s] if (conditioned and sp.identity_context_raw is not None) else None
        inputs = build_hgnn_inputs(
            champion_id=sp.champion_id[s], build_id=sp.build_id[s],
            win_rate=sp.win_rate[s], matchup_1v1=sp.matchup_1v1[s], synergy_2vx=sp.synergy_2vx[s],
            p1_cnt=sp.p1_cnt[s], m1v1_cnt=sp.m1v1_cnt[s], s2vx_cnt=sp.s2vx_cnt[s], strength=strength,
            identity_context=sp.identity_context[s],
            identity_context_support=sp.identity_context_support[s],
            identity_context_raw=raw, device=device,
        )
        with torch.no_grad():
            full.append(model(**inputs)["final_logit"].cpu().numpy())
            # base = full - context term is exact (both heads are additive zero-init residuals).
            if conditioned:
                dense = None
                if model.identity_conditioned_context.dense_dim > 0 and "identity_context" in inputs:
                    dense = inputs["identity_context"][..., model.config.context_interpretable_dim :]
                c = model.identity_conditioned_context(
                    inputs["identity_context_raw"], inputs["identity_context_support"],
                    inputs["champion_id"], inputs["build_id"], dense,
                )
            else:
                c = model._context_logit(inputs["identity_context"], inputs["identity_context_support"])
            ctx_logit.append(c.cpu().numpy())
    return np.concatenate(full), np.concatenate(ctx_logit)


def _fit_ceiling(
    feats: dict[str, np.ndarray],
    base: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    *,
    device: str,
    nonlinear: bool,
) -> dict[str, np.ndarray]:
    """Fit a draft-safe context model over the frozen base offset; return logits."""
    mu = feats["train"].mean(0)
    sd = feats["train"].std(0) + 1e-6

    def std(name: str) -> torch.Tensor:
        return torch.tensor((feats[name] - mu) / sd, device=device)

    y_tr = torch.tensor(y["train"], dtype=torch.float32, device=device)
    off_tr = torch.tensor(base["train"], device=device)
    scale = torch.ones(1, device=device, requires_grad=True)
    if nonlinear:
        net = nn.Sequential(nn.Linear(feats["train"].shape[1], 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        params = list(net.parameters()) + [scale]
        head = lambda x: net(x).squeeze(-1)
    else:
        w = torch.zeros(feats["train"].shape[1], device=device, requires_grad=True)
        params = [w, scale]
        head = lambda x: x @ w
    opt = torch.optim.Adam(params, lr=3e-3 if nonlinear else 0.05, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    x_tr = std("train")
    n = x_tr.shape[0]
    best_val, best_state, best_scale = np.inf, None, 1.0
    for _ in range(40 if nonlinear else 1):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 65536):
            idx = perm[i : i + 65536]
            opt.zero_grad()
            loss = loss_fn(scale * off_tr[idx] + head(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        if not nonlinear:
            for _ in range(599):
                opt.zero_grad()
                loss = loss_fn(scale * off_tr + head(x_tr), y_tr) + 1e-4 * (head(x_tr) ** 2).mean()
                loss.backward()
                opt.step()
            break
        with torch.no_grad():
            val_logit = (scale * torch.tensor(base["val"], device=device) + head(std("val"))).cpu().numpy()
        v = _nll(val_logit, y["val"])
        if v < best_val:
            best_val = v
            best_scale = float(scale)
            if nonlinear:
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
    if nonlinear and best_state is not None:
        net.load_state_dict(best_state)
        scale = torch.tensor([best_scale], device=device)
    out = {}
    with torch.no_grad():
        for name in SPLITS:
            out[name] = (scale * torch.tensor(base[name], device=device) + head(std(name))).cpu().numpy()
    return out


def _decile_calibration(
    axis: np.ndarray, base: np.ndarray, full: np.ndarray, y: np.ndarray
) -> dict[str, object]:
    """Bin by a context axis; compare empirical WR, base WR, full(head) WR."""
    edges = np.quantile(axis, np.linspace(0, 1, 11))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    binid = np.clip(np.digitize(axis, edges) - 1, 0, 9)
    rows = []
    for d in range(10):
        m = binid == d
        rows.append(
            (
                float(y[m].mean()),
                float((1 / (1 + np.exp(-base[m]))).mean()),
                float((1 / (1 + np.exp(-full[m]))).mean()),
            )
        )
    emp_swing = rows[-1][0] - rows[0][0]
    full_swing = rows[-1][2] - rows[0][2]
    emp_full_errors = np.array([e - h for e, _b, h in rows], dtype=np.float64)
    miss_swing = (rows[-1][0] - rows[-1][1]) - (rows[0][0] - rows[0][1])
    head_swing = (rows[-1][2] - rows[-1][1]) - (rows[0][2] - rows[0][1])
    return {
        "rows": rows,
        "emp_swing_pp": emp_swing * 100,
        "actual_model_swing_pp": full_swing * 100,
        "emp_actual_swing_gap_pp": (emp_swing - full_swing) * 100,
        "emp_actual_mean_abs_error_pp": float(np.abs(emp_full_errors).mean() * 100),
        "emp_actual_max_abs_error_pp": float(np.abs(emp_full_errors).max() * 100),
        "emp_actual_mean_error_pp": float(emp_full_errors.mean() * 100),
        "base_miss_swing_pp": miss_swing * 100,
        "head_corr_swing_pp": head_swing * 100,
        "realized_of_missable_pct": (100 * head_swing / miss_swing) if miss_swing else float("nan"),
        "realized_of_raw_pct": (100 * head_swing / emp_swing) if emp_swing else float("nan"),
    }


def run(model, splits: dict[str, SplitData], *, strength: float, device: str) -> dict[str, object]:
    base, full, feats, y = {}, {}, {}, {}
    for name in SPLITS:
        sp = splits[name]
        f, c = _predict(model, sp, strength=strength, device=device)
        full[name] = f
        base[name] = f - c
        y[name] = sp.blue_win.astype(np.float64)
        feats[name] = _draft_features(np.asarray(sp.identity_context, dtype=np.float32))

    linear = _fit_ceiling(feats, base, y, device=device, nonlinear=False)
    mlp = _fit_ceiling(feats, base, y, device=device, nonlinear=True)
    # Context axis for calibration bins = the standalone linear context score.
    axis_logit = _fit_ceiling(
        feats, {k: np.zeros_like(v) for k, v in base.items()}, y, device=device, nonlinear=False
    )

    report: dict[str, object] = {"splits": {}}
    for name in SPLITS:
        report["splits"][name] = {
            "base": {"auc": _auc(base[name], y[name]), "nll": _nll(base[name], y[name])},
            "head_full": {"auc": _auc(full[name], y[name]), "nll": _nll(full[name], y[name])},
            "ceiling_linear": {"auc": _auc(linear[name], y[name]), "nll": _nll(linear[name], y[name])},
            "ceiling_mlp": {"auc": _auc(mlp[name], y[name]), "nll": _nll(mlp[name], y[name])},
            "calibration": _decile_calibration(axis_logit[name], base[name], full[name], y[name]),
        }
    return report


def _print(report: dict[str, object]) -> None:
    print(f"{'split':6s} {'base AUC':>9s} {'head AUC':>9s} {'ceil-lin':>9s} {'ceil-mlp':>9s}"
          f" | {'base NLL':>9s} {'head NLL':>9s} {'ceil-mlp':>9s}")
    print("-" * 84)
    for name in SPLITS:
        s = report["splits"][name]
        print(f"{name:6s} {s['base']['auc']:9.4f} {s['head_full']['auc']:9.4f}"
              f" {s['ceiling_linear']['auc']:9.4f} {s['ceiling_mlp']['auc']:9.4f}"
              f" | {s['base']['nll']:9.4f} {s['head_full']['nll']:9.4f} {s['ceiling_mlp']['nll']:9.4f}")
    print("\ncontext-axis decile calibration (empirical vs actual model):")
    print(f"{'split':6s} {'emp swing':>10s} {'actual':>10s} {'emp-actual':>12s}"
          f" {'MAE':>8s} {'maxAE':>8s}")
    for name in SPLITS:
        c = report["splits"][name]["calibration"]
        print(f"{name:6s} {c['emp_swing_pp']:+9.2f}pp {c['actual_model_swing_pp']:+9.2f}pp"
              f" {c['emp_actual_swing_gap_pp']:+11.2f}pp"
              f" {c['emp_actual_mean_abs_error_pp']:7.2f}pp"
              f" {c['emp_actual_max_abs_error_pp']:7.2f}pp")
    print("\ntest decile table (empWR / actualWR / baseWR, %):")
    for d, (e, b, h) in enumerate(report["splits"]["test"]["calibration"]["rows"]):
        print(f"  D{d+1:<2d} emp={e*100:5.2f} actual={h*100:5.2f} base={b*100:5.2f}"
              f"  (emp-actual={ (e-h)*100:+5.2f}  emp-base={ (e-b)*100:+5.2f}"
              f"  actual-base={ (h-b)*100:+5.2f})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None, help="HGNN checkpoint to validate")
    args = parser.parse_args()

    setup_logging_config()
    logging.getLogger().setLevel(logging.WARNING)
    device = resolve_device("auto")
    model_path = args.model_path or TrainConfig().model_path
    model, _, strength = load_hgnn_model(model_path, device=device)
    if not (
        getattr(model, "context_enabled", False)
        or getattr(model, "identity_conditioned_context_enabled", False)
    ):
        raise SystemExit("Loaded model has no context head (identity_context_dim=0).")
    splits = load_splits(DatasetConfig(), require_counts=True)
    report = run(model, splits, strength=strength, device=device)
    _print(report)
    out = Path("app/ml/data/context_wr_validation.json")
    out.write_text(json.dumps(report, indent=2))
    logger.warning("Wrote %s", out)


if __name__ == "__main__":
    main()
