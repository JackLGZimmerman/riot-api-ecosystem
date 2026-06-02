"""Automated identity-atlas audit: shared head vs identity-conditioned head.

For every high-support identity ``(championid, teamposition, build)`` we find the
enemy/ally context axis it is most empirically sensitive to (largest low->high
win-rate gradient), then compare how well each model tracks that gradient:

    empirical          actual win-rate low->high swing on that axis
    base               context-free model (shared.final - shared.context)
    shared atlas       the shared context-head model (the "before")
    conditioned        the identity-conditioned model (the "after")

The remaining gap ``|empirical - model|`` is the under-fit. Ranking by the
conditioned gap surfaces where headroom is left; the Malphite case appears as one
row, not a special case. Improvement is judged by the gap shrinking across many
identities, not one champion.

Run with:
    python -m app.ml.context_atlas_audit
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from app.classification.embeddings.config import (
    CONTEXT_DAMAGE_PRESSURE_INDEX,
    CONTEXT_HEAL_SHIELD_INDEX,
)
from app.core.logging.logger import setup_logging_config
from app.core.utils.common import POSITIONS
from app.ml.config import DatasetConfig, TrainConfig
from app.ml.dataset import identity_meta, load_splits
from app.ml.hgnn_model import HGNNWinModel, build_hgnn_inputs, load_hgnn_model, resolve_device

logger = logging.getLogger(__name__)

PHYS, MAGIC = 0, 1
DMG = CONTEXT_DAMAGE_PRESSURE_INDEX
HEAL = CONTEXT_HEAL_SHIELD_INDEX

# Candidate enemy/ally context axes (channel, mode, label). Damage-weighted means
# track "expected" enemy composition; ally damage tracks 2vX synergy.
AXES = (
    ("enemy_phys", PHYS, "enemy", True),
    ("enemy_magic", MAGIC, "enemy", True),
    ("enemy_damage", DMG, "enemy", False),
    ("enemy_heal_shield", HEAL, "enemy", False),
    ("ally_damage", DMG, "ally", False),
)
MIN_SUPPORT = 200.0  # identity historical matchups (atlas support)
MIN_OBS = 4000  # focus-side observations for a stable gradient
MIN_TAIL = 400  # observations required in each tertile tail


@dataclass
class AuditRow:
    championid: int
    teamposition: str
    build: str
    support: float
    n_obs: int
    axis: str
    emp_gradient: float
    base_gradient: float
    shared_gradient: float
    conditioned_gradient: float
    gap_before: float  # |emp - shared|
    gap_after: float  # |emp - conditioned|
    auc_before: float
    auc_after: float
    nll_before: float
    nll_after: float


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _nll(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(prob, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _weighted_team_axis(team_ctx: np.ndarray, channel: int, weighted: bool) -> np.ndarray:
    """team_ctx [N, 5, D] -> [N] team summary on `channel`."""
    if weighted:
        w = np.clip(team_ctx[:, :, DMG], 0.0, None)
        denom = np.clip(w.sum(axis=1), 1e-6, None)
        return (team_ctx[:, :, channel] * w).sum(axis=1) / denom
    return team_ctx[:, :, channel].mean(axis=1)


def _predict(model: HGNNWinModel, sp, *, strength: float, device: str, want_ctx: bool):
    """Return P(blue win) and (optional) the context residual logit, per game."""
    n = sp.blue_win.size
    probs: list[np.ndarray] = []
    ctxs: list[np.ndarray] = []
    raw = sp.identity_context_raw
    dense_src = (
        model.identity_conditioned_context.dense_dim > 0
        if model.identity_conditioned_context_enabled
        else False
    )
    for i in range(0, n, 32768):
        sl = slice(i, i + 32768)
        kwargs = dict(
            champion_id=sp.champion_id[sl],
            build_id=sp.build_id[sl],
            win_rate=sp.win_rate[sl],
            matchup_1v1=sp.matchup_1v1[sl],
            synergy_2vx=sp.synergy_2vx[sl],
            p1_cnt=sp.p1_cnt[sl],
            m1v1_cnt=sp.m1v1_cnt[sl],
            s2vx_cnt=sp.s2vx_cnt[sl],
            strength=strength,
            identity_context=sp.identity_context[sl] if sp.identity_context is not None else None,
            identity_context_support=sp.identity_context_support[sl],
            device=device,
        )
        if raw is not None:
            kwargs["identity_context_raw"] = raw[sl]
        inp = build_hgnn_inputs(**kwargs)
        with torch.no_grad():
            full = model(**inp)["final_logit"]
            probs.append(torch.sigmoid(full).cpu().numpy())
            if want_ctx:
                if model.identity_conditioned_context_enabled:
                    dense = inp["identity_context"][..., model.config.context_interpretable_dim :] if dense_src else None
                    ctx = model.identity_conditioned_context(
                        inp["identity_context_raw"],
                        inp["identity_context_support"],
                        inp["champion_id"],
                        inp["build_id"],
                        dense,
                    )
                else:
                    ctx = model._context_logit(inp["identity_context"], inp["identity_context_support"])
                ctxs.append(ctx.cpu().numpy())
    prob = np.concatenate(probs).astype(np.float64)
    ctx = np.concatenate(ctxs).astype(np.float64) if want_ctx else None
    return prob, ctx


def _tertile_gradient(value: np.ndarray, *series: np.ndarray):
    """low/high tertile high-minus-low for each series; returns (grads, lo_n, hi_n)."""
    lo_thr, hi_thr = np.quantile(value, [1 / 3, 2 / 3])
    lo = value <= lo_thr
    hi = value >= hi_thr
    lo_n, hi_n = int(lo.sum()), int(hi.sum())
    grads = [float(s[hi].mean() - s[lo].mean()) for s in series]
    return grads, lo_n, hi_n


def run_audit(
    *,
    baseline_path: Path,
    conditioned_path: Path,
    device: str,
) -> list[AuditRow]:
    dataset_cfg = DatasetConfig()
    splits = load_splits(dataset_cfg, require_counts=True)
    meta = identity_meta(dataset_cfg)
    build_vocab: list[str] = list(meta["build_vocab"])

    shared, _, strength = load_hgnn_model(baseline_path, device=device)
    cond, _, _ = load_hgnn_model(conditioned_path, device=device)
    if not cond.identity_conditioned_context_enabled:
        raise SystemExit(f"{conditioned_path} is not an identity-conditioned model")

    # Concatenate all splits once (only the fields prediction/audit need; RAM is
    # tight) for stable per-identity gradients. Both models are scored
    # identically so the before/after comparison is fair.
    merged = _merge(splits)
    champ = np.asarray(merged.champion_id).astype(np.int64)
    build = np.asarray(merged.build_id).astype(np.int64)
    ctx24 = np.asarray(merged.identity_context).astype(np.float32)
    support10 = np.asarray(merged.identity_context_support).astype(np.float32)
    blue_win = np.asarray(merged.blue_win).astype(np.float64)

    shared_prob, shared_ctx = _predict(shared, merged, strength=strength, device=device, want_ctx=True)
    base_logit = np.log(np.clip(shared_prob, 1e-9, 1 - 1e-9) / np.clip(1 - shared_prob, 1e-9, 1 - 1e-9)) - shared_ctx
    base_prob = 1.0 / (1.0 + np.exp(-base_logit))
    cond_prob, _ = _predict(cond, merged, strength=strength, device=device, want_ctx=False)

    # Per-game team axis summaries for both possible focus sides.
    blue_ctx, red_ctx = ctx24[:, :5], ctx24[:, 5:]
    enemy_for_blue = {a: _weighted_team_axis(red_ctx, ch, w) for a, ch, side, w in AXES if side == "enemy"}
    enemy_for_red = {a: _weighted_team_axis(blue_ctx, ch, w) for a, ch, side, w in AXES if side == "enemy"}
    ally_for_blue = {a: _weighted_team_axis(blue_ctx, ch, w) for a, ch, side, w in AXES if side == "ally"}
    ally_for_red = {a: _weighted_team_axis(red_ctx, ch, w) for a, ch, side, w in AXES if side == "ally"}

    # Enumerate identities present, keyed by (champ, role_idx, build_idx).
    rows: list[AuditRow] = []
    seen: set[tuple[int, int, int]] = set()
    for role_idx in range(5):
        for side_off in (0, 5):
            slot = role_idx + side_off
            keys = np.unique(np.stack([champ[:, slot], build[:, slot]], axis=1), axis=0)
            for c, b in keys.tolist():
                key = (int(c), role_idx, int(b))
                if key in seen or c < 0:
                    continue
                seen.add(key)
                row = _audit_identity(
                    key,
                    champ=champ,
                    build=build,
                    support10=support10,
                    blue_win=blue_win,
                    base_prob=base_prob,
                    shared_prob=shared_prob,
                    cond_prob=cond_prob,
                    enemy_for_blue=enemy_for_blue,
                    enemy_for_red=enemy_for_red,
                    ally_for_blue=ally_for_blue,
                    ally_for_red=ally_for_red,
                    build_vocab=build_vocab,
                )
                if row is not None:
                    rows.append(row)
    rows.sort(key=lambda r: abs(r.gap_after), reverse=True)
    return rows


# Only these fields are needed for prediction + audit; the wide unused
# classification arrays (semantic/profile/m1v1_detail) are left out to fit RAM.
_MERGE_FIELDS = (
    "champion_id",
    "build_id",
    "win_rate",
    "matchup_1v1",
    "synergy_2vx",
    "p1_cnt",
    "m1v1_cnt",
    "s2vx_cnt",
    "identity_context",
    "identity_context_support",
    "identity_context_raw",
    "blue_win",
)


def _merge(splits):
    """One concatenated SplitData across all splits (needed fields only)."""
    from app.ml.dataset import SplitData

    names = ("train", "val", "test")

    def cat(attr):
        if attr not in _MERGE_FIELDS:
            return None
        vals = [getattr(splits[s], attr) for s in names]
        if any(v is None for v in vals):
            return None
        return np.concatenate([np.asarray(v) for v in vals])

    return SplitData(**{f: cat(f) for f in SplitData.__dataclass_fields__})


def _audit_identity(
    key,
    *,
    champ,
    build,
    support10,
    blue_win,
    base_prob,
    shared_prob,
    cond_prob,
    enemy_for_blue,
    enemy_for_red,
    ally_for_blue,
    ally_for_red,
    build_vocab,
) -> AuditRow | None:
    c, role_idx, b = key
    blue_slot, red_slot = role_idx, role_idx + 5
    blue_mask = (champ[:, blue_slot] == c) & (build[:, blue_slot] == b)
    red_mask = (champ[:, red_slot] == c) & (build[:, red_slot] == b)
    n_obs = int(blue_mask.sum() + red_mask.sum())
    if n_obs < MIN_OBS:
        return None
    support = float(
        np.concatenate([support10[blue_mask, blue_slot], support10[red_mask, red_slot]]).mean()
    )
    if support < MIN_SUPPORT:
        return None

    # Focus-side observations: blue focus uses P(blue); red focus uses 1-P(blue).
    def focus(series_blue_prob: np.ndarray) -> np.ndarray:
        return np.concatenate([series_blue_prob[blue_mask], 1.0 - series_blue_prob[red_mask]])

    label = np.concatenate([blue_win[blue_mask], 1.0 - blue_win[red_mask]])
    base = focus(base_prob)
    shared = focus(shared_prob)
    cond = focus(cond_prob)

    best = None
    for axis_name, _, side, _ in AXES:
        if side == "enemy":
            value = np.concatenate([enemy_for_blue[axis_name][blue_mask], enemy_for_red[axis_name][red_mask]])
        else:
            value = np.concatenate([ally_for_blue[axis_name][blue_mask], ally_for_red[axis_name][red_mask]])
        (grads, lo_n, hi_n) = _tertile_gradient(value, label, base, shared, cond)
        if lo_n < MIN_TAIL or hi_n < MIN_TAIL:
            continue
        emp = grads[0]
        if best is None or abs(emp) > abs(best[1][0]):
            best = (axis_name, grads)
    if best is None:
        return None

    axis_name, (emp_g, base_g, shared_g, cond_g) = best
    return AuditRow(
        championid=int(c),
        teamposition=POSITIONS[role_idx],
        build=build_vocab[b] if 0 <= b < len(build_vocab) else "",
        support=round(support, 1),
        n_obs=n_obs,
        axis=axis_name,
        emp_gradient=round(emp_g, 4),
        base_gradient=round(base_g, 4),
        shared_gradient=round(shared_g, 4),
        conditioned_gradient=round(cond_g, 4),
        gap_before=round(abs(emp_g - shared_g), 4),
        gap_after=round(abs(emp_g - cond_g), 4),
        auc_before=round(_auc(shared, label), 4),
        auc_after=round(_auc(cond, label), 4),
        nll_before=round(_nll(shared, label), 4),
        nll_after=round(_nll(cond, label), 4),
    )


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.WARNING)
    device = resolve_device("auto")
    exp = TrainConfig().model_path.parent / "experiments" / "identity_conditioned"
    rows = run_audit(
        baseline_path=exp / "baseline_shared.pt",
        conditioned_path=exp / "cond_raw.pt",
        device=device,
    )

    gap_before = np.array([r.gap_before for r in rows])
    gap_after = np.array([r.gap_after for r in rows])
    auc_before = np.array([r.auc_before for r in rows])
    auc_after = np.array([r.auc_after for r in rows])
    nll_before = np.array([r.nll_before for r in rows])
    nll_after = np.array([r.nll_after for r in rows])
    improved = int((gap_after < gap_before).sum())
    # Per-identity AUC/NLL is the selection-bias-free signal (the max-selected
    # gradient gap is noisy); the gap closure is reported on the under-fit tail.
    top = np.argsort(gap_before)[::-1][:60]
    print(f"identities audited: {len(rows)}")
    print(
        f"per-identity AUC improved: {100*(auc_after>auc_before).mean():.1f}%  "
        f"(mean {auc_before.mean():.4f} -> {auc_after.mean():.4f}, Δ {+(auc_after-auc_before).mean():+.4f})"
    )
    print(
        f"per-identity NLL improved: {100*(nll_after<nll_before).mean():.1f}%  "
        f"(mean {nll_before.mean():.4f} -> {nll_after.mean():.4f}, Δ {(nll_after-nll_before).mean():+.4f})"
    )
    print(f"mean |gap| before: {gap_before.mean():.4f}  after: {gap_after.mean():.4f}")
    print(
        f"under-fit tail (top 60 by gap_before): {gap_before[top].mean():.4f} -> "
        f"{gap_after[top].mean():.4f}  closed {100*(gap_after[top]<gap_before[top]).mean():.1f}%"
    )
    print(f"all identities with smaller gap after: {improved}/{len(rows)} ({100*improved/len(rows):.1f}%)")
    print()
    header = (
        f"{'champ':>5} {'role':<7} {'build':<18} {'sup':>6} {'n':>7} {'axis':<16} "
        f"{'emp':>7} {'base':>7} {'shared':>7} {'cond':>7} {'gapB':>6} {'gapA':>6} {'aucB':>6} {'aucA':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows[:30]:
        print(
            f"{r.championid:>5} {r.teamposition:<7} {r.build:<18} {r.support:>6.0f} {r.n_obs:>7} {r.axis:<16} "
            f"{r.emp_gradient:>7.3f} {r.base_gradient:>7.3f} {r.shared_gradient:>7.3f} {r.conditioned_gradient:>7.3f} "
            f"{r.gap_before:>6.3f} {r.gap_after:>6.3f} {r.auc_before:>6.3f} {r.auc_after:>6.3f}"
        )

    out = Path("app/ml/data/context_atlas_audit.json")
    out.write_text(
        json.dumps(
            {
                "summary": {
                    "n_identities": len(rows),
                    "mean_gap_before": float(gap_before.mean()),
                    "mean_gap_after": float(gap_after.mean()),
                    "median_gap_before": float(np.median(gap_before)),
                    "median_gap_after": float(np.median(gap_after)),
                    "improved": improved,
                },
                "rows": [asdict(r) for r in rows],
            },
            indent=2,
        )
    )
    logger.warning("Wrote %s", out)


if __name__ == "__main__":
    main()
