"""Model-aligned context examples audit.

Reproduces every table in ``documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md``:
for each documented draft-time slice it reports the empirical focus-side win
rate (`Emp WR`, pure data) against the model's mean predicted focus-side win
probability (`Model WR`), so `Gap = Model WR - Emp WR` shows where the average
HGNN prediction diverges from the data.

`n` and `Emp WR` are model-independent; running this with the production model
must reproduce the committed doc, which is the fidelity check. Swap the model
with ``--model-path`` to audit a different checkpoint (for example the shared
context baseline); the context term is isolated for either head so the
context-free `base` WR is exact.

Run with:
    .venv/bin/python -m app.ml.context_examples_audit [--model-path PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.ml.cache_layout import (
    CONTEXT_DAMAGE_PRESSURE_INDEX,
    CONTEXT_HEAL_SHIELD_INDEX,
)
from app.core.logging.logger import setup_logging_config
from app.ml.config import DatasetConfig, TrainConfig
from app.ml.dataset import SplitData, identity_meta, load_splits
from app.ml.hgnn_model import build_hgnn_inputs, load_hgnn_model, resolve_device

logger = logging.getLogger(__name__)

PHYS, MAGIC = 0, 1
DMG, HEAL = CONTEXT_DAMAGE_PRESSURE_INDEX, CONTEXT_HEAL_SHIELD_INDEX
SPLITS = ("train", "val", "test")
QUINTILE_LABELS = ("0-20", "20-40", "40-60", "60-80", "80-100")

# Riot champion ids used by the documented examples.
CH = {
    "Malphite": 54, "Sion": 14, "DrMundo": 36, "Maokai": 57, "Ornn": 516,
    "Nautilus": 111, "Shen": 98, "Poppy": 78, "Galio": 3, "Chogath": 31,
    "Amumu": 32, "Sona": 37, "Karma": 43, "Lulu": 117, "Zilean": 26,
}
SKIRMISHERS = {887: "Gwen", 24: "Jax", 39: "Irelia", 114: "Fiora", 77: "Udyr", 5: "XinZhao"}


def _predict(model, sp: SplitData, *, strength: float, device: str, conditioned: bool):
    """Per-game blue-win prob and the additive context-term logit."""
    full, ctx = [], []
    n = sp.blue_win.size
    use_relationships = bool(model.config.use_relationship_integrations)
    for i in range(0, n, 32768):
        s = slice(i, i + 32768)
        raw = sp.identity_context_raw[s] if (conditioned and sp.identity_context_raw is not None) else None
        inp = build_hgnn_inputs(
            champion_id=sp.champion_id[s], build_id=sp.build_id[s], win_rate=sp.win_rate[s],
            p1_cnt=sp.p1_cnt[s], strength=strength,
            matchup_1v1=sp.matchup_1v1[s] if (use_relationships and sp.matchup_1v1 is not None) else None,
            synergy_2vx=sp.synergy_2vx[s] if (use_relationships and sp.synergy_2vx is not None) else None,
            m1v1_cnt=sp.m1v1_cnt[s] if (use_relationships and sp.m1v1_cnt is not None) else None,
            s2vx_cnt=sp.s2vx_cnt[s] if (use_relationships and sp.s2vx_cnt is not None) else None,
            include_relationship_features=use_relationships,
            identity_context=sp.identity_context[s],
            identity_context_support=sp.identity_context_support[s],
            identity_context_raw=raw, device=device,
        )
        with torch.no_grad():
            full.append(model(**inp)["final_logit"].cpu().numpy())
            if conditioned:
                dense = None
                if model.identity_conditioned_context.dense_dim > 0 and "identity_context" in inp:
                    dense = inp["identity_context"][..., model.config.context_interpretable_dim :]
                c = model.identity_conditioned_context(
                    inp["identity_context_raw"], inp["identity_context_support"],
                    inp["champion_id"], inp["build_id"], dense,
                )
            else:
                c = model._context_logit(inp["identity_context"], inp["identity_context_support"])
            ctx.append(c.cpu().numpy())
    return np.concatenate(full), np.concatenate(ctx)


def _weighted_share(ctx5: np.ndarray, axis: int) -> np.ndarray:
    """Damage-pressure-weighted mean of an offense-share axis over 5 players."""
    w = np.clip(ctx5[:, :, DMG], 0.0, None)
    den = np.clip(w.sum(1), 1e-6, None)
    return (ctx5[:, :, axis] * w).sum(1) / den


class SideRows:
    """Flat per-(game, focus-team) table with everything the examples need."""

    def __init__(self) -> None:
        self.cols: dict[str, list] = {k: [] for k in (
            "emp", "model", "base", "champ", "build", "enemy_phys", "enemy_magic",
            "enemy_dmg", "enemy_heal", "focus_dmg", "focus_heal", "ally_skirmish",
            "mean_support", "min_support", "zero_support_players",
        )}

    def add_split(self, sp: SplitData, full: np.ndarray, ctxl: np.ndarray) -> None:
        cid = np.asarray(sp.champion_id)
        bid = np.asarray(sp.build_id)
        ctx = np.asarray(sp.identity_context, dtype=np.float32)
        sup = (
            np.asarray(sp.identity_context_support, dtype=np.float32)
            if sp.identity_context_support is not None
            else np.zeros((sp.blue_win.size, 10), dtype=np.float32)
        )
        bw = np.asarray(sp.blue_win).astype(np.float64)
        pblue = 1.0 / (1.0 + np.exp(-full))
        base_blue = 1.0 / (1.0 + np.exp(-(full - ctxl)))
        skirmish = np.isin(cid, list(SKIRMISHERS))  # [N, 10]
        for focus, enemy in ((slice(0, 5), slice(5, 10)), (slice(5, 10), slice(0, 5))):
            fctx, ectx = ctx[:, focus, :], ctx[:, enemy, :]
            fsup = sup[:, focus]
            is_blue = focus.start == 0
            self.cols["emp"].append(bw if is_blue else 1.0 - bw)
            self.cols["model"].append(pblue if is_blue else 1.0 - pblue)
            self.cols["base"].append(base_blue if is_blue else 1.0 - base_blue)
            self.cols["champ"].append(cid[:, focus])
            self.cols["build"].append(bid[:, focus])
            self.cols["enemy_phys"].append(_weighted_share(ectx, PHYS))
            self.cols["enemy_magic"].append(_weighted_share(ectx, MAGIC))
            self.cols["enemy_dmg"].append(ectx[:, :, DMG].mean(1))
            self.cols["enemy_heal"].append(ectx[:, :, HEAL].mean(1))
            self.cols["focus_dmg"].append(fctx[:, :, DMG].mean(1))
            self.cols["focus_heal"].append(fctx[:, :, HEAL].mean(1))
            self.cols["ally_skirmish"].append(skirmish[:, focus].sum(1))
            self.cols["mean_support"].append(fsup.mean(1))
            self.cols["min_support"].append(np.where(fsup > 0.0, fsup, np.inf).min(1))
            self.cols["zero_support_players"].append((fsup <= 0.0).sum(1))

    def finalize(self) -> None:
        self.d = {k: np.concatenate(v) for k, v in self.cols.items()}
        self.d["min_support"] = np.where(np.isinf(self.d["min_support"]), 0.0, self.d["min_support"])

    def has_identity(self, champ: int, build: int | None) -> np.ndarray:
        """Focus rows containing `champ` (optionally with `build`) in any role."""
        m = self.d["champ"] == champ
        if build is not None:
            m = m & (self.d["build"] == build)
        return m.any(1)

    def has_identity_role(self, champ: int, role: int, build: int | None) -> np.ndarray:
        m = self.d["champ"][:, role] == champ
        if build is not None:
            m = m & (self.d["build"][:, role] == build)
        return m


@dataclass(frozen=True)
class Predicate:
    label: str
    mask: np.ndarray
    family: str
    parent_mask: np.ndarray | None = None


def _slice_suggestion(family: str) -> str:
    if family.startswith("identity"):
        return "identity_conditioning/backoff_ablation"
    if family.startswith("context"):
        return "context_set_encoder_or_semantic_feature_ablation"
    if family.startswith("support"):
        return "support_bucket_calibration_or_smoothing_ablation"
    if family.startswith("pair"):
        return "interaction_slice_data_or_model_ablation"
    return "calibration_audit"


def rank_residual_slices(
    predicates: list[Predicate],
    model_prob: np.ndarray,
    labels: np.ndarray,
    *,
    min_support: int = 1000,
    shrink_strength: float = 1000.0,
    top: int = 50,
    bootstrap: int = 0,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Rank validation-only residual slices with sparse-slice guardrails."""
    p = np.asarray(model_prob, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    residual = p - y
    global_gap = float(residual.mean()) if residual.size else 0.0
    rows: list[dict[str, object]] = []
    for pred in predicates:
        mask = np.asarray(pred.mask, dtype=bool)
        n = int(mask.sum())
        if n < min_support:
            continue
        values = residual[mask]
        raw_gap = float(values.mean())
        parent_gap = global_gap
        if pred.parent_mask is not None:
            parent = np.asarray(pred.parent_mask, dtype=bool)
            if int(parent.sum()) >= min_support:
                parent_gap = float(residual[parent].mean())
        shrunk_gap = float((n * raw_gap + shrink_strength * parent_gap) / (n + shrink_strength))
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        ci_half = float(1.96 * std / math.sqrt(max(n, 1)))
        rows.append(
            {
                "label": pred.label,
                "family": pred.family,
                "n": n,
                "model_mean": float(p[mask].mean()),
                "label_mean": float(y[mask].mean()),
                "raw_gap": raw_gap,
                "parent_gap": parent_gap,
                "shrunk_gap": shrunk_gap,
                "ci_half_width": ci_half,
                "ci_low": shrunk_gap - ci_half,
                "ci_high": shrunk_gap + ci_half,
                "score": max(0.0, abs(shrunk_gap) - ci_half),
                "suggestion": _slice_suggestion(pred.family),
                "bootstrap_low": None,
                "bootstrap_high": None,
            }
        )
    rows.sort(key=lambda r: (float(r["score"]), abs(float(r["shrunk_gap"])), int(r["n"])), reverse=True)
    if bootstrap > 0 and rows:
        rng = np.random.default_rng(seed)
        mask_by_label = {pred.label: np.asarray(pred.mask, dtype=bool) for pred in predicates}
        for row in rows[:top]:
            values = residual[mask_by_label[str(row["label"])]]
            draws = rng.choice(values, size=(int(bootstrap), values.size), replace=True).mean(axis=1)
            row["bootstrap_low"] = float(np.quantile(draws, 0.025))
            row["bootstrap_high"] = float(np.quantile(draws, 0.975))
    return rows[:top]


def multiaccuracy_residual_report(
    predicates: list[Predicate],
    model_prob: np.ndarray,
    labels: np.ndarray,
    *,
    min_support: int = 1000,
    shrink_strength: float = 1000.0,
    top: int = 25,
) -> list[dict[str, object]]:
    rows = rank_residual_slices(
        predicates,
        model_prob,
        labels,
        min_support=min_support,
        shrink_strength=shrink_strength,
        top=top,
    )
    for row in rows:
        row["residual_direction"] = (
            "model_high" if float(row["shrunk_gap"]) > 0.0 else "model_low"
        )
    return rows


def _label_build(build_vocab: dict[str, int], build_id: int) -> str:
    labels = {idx: label for label, idx in build_vocab.items()}
    return labels.get(int(build_id), str(int(build_id)))


def _append_if_supported(out: list[Predicate], pred: Predicate, *, min_support: int) -> None:
    if int(np.asarray(pred.mask, dtype=bool).sum()) >= min_support:
        out.append(pred)


def _axis_band_predicates(d: dict[str, np.ndarray], *, min_support: int) -> list[Predicate]:
    out: list[Predicate] = []
    axes = (
        "enemy_phys", "enemy_magic", "enemy_dmg", "enemy_heal",
        "focus_dmg", "focus_heal", "ally_skirmish",
    )
    for name in axes:
        values = np.asarray(d[name], dtype=np.float64)
        if values.size == 0:
            continue
        edges = np.unique(np.quantile(values, [0.20, 0.40, 0.60, 0.80]))
        band = np.searchsorted(edges, values, side="right")
        for idx in range(edges.size + 1):
            mask = band == idx
            label = QUINTILE_LABELS[idx] if edges.size == 4 else f"band{idx}"
            _append_if_supported(
                out,
                Predicate(f"context:{name}:{label}", mask, "context_axis"),
                min_support=min_support,
            )
    return out


def _support_predicates(d: dict[str, np.ndarray], *, min_support: int) -> list[Predicate]:
    out: list[Predicate] = []
    mean_support = np.asarray(d["mean_support"], dtype=np.float64)
    min_seen = np.asarray(d["min_support"], dtype=np.float64)
    zero_players = np.asarray(d["zero_support_players"], dtype=np.float64)
    candidates = (
        ("support:zero_player", zero_players > 0.0),
        ("support:min_zero", min_seen <= 0.0),
        ("support:mean_low_<30", mean_support < 30.0),
        ("support:mean_mid_30_199", (mean_support >= 30.0) & (mean_support < 200.0)),
        ("support:mean_high_200_plus", mean_support >= 200.0),
    )
    for label, mask in candidates:
        _append_if_supported(out, Predicate(label, mask, "support"), min_support=min_support)
    return out


def _identity_predicates(
    d: dict[str, np.ndarray],
    build_vocab: dict[str, int],
    *,
    min_support: int,
) -> list[Predicate]:
    out: list[Predicate] = []
    champ = np.asarray(d["champ"])
    build = np.asarray(d["build"])
    for role in range(champ.shape[1]):
        role_parent = np.ones(champ.shape[0], dtype=bool)
        for champ_id in np.unique(champ[:, role]):
            mask = champ[:, role] == champ_id
            _append_if_supported(
                out,
                Predicate(f"identity:role{role}:champ:{int(champ_id)}", mask, "identity_champion", role_parent),
                min_support=min_support,
            )
        for build_id in np.unique(build[:, role]):
            mask = build[:, role] == build_id
            label = _label_build(build_vocab, int(build_id))
            _append_if_supported(
                out,
                Predicate(f"identity:role{role}:build:{label}", mask, "identity_build", role_parent),
                min_support=min_support,
            )
    for champ_id in np.unique(champ):
        mask = (champ == champ_id).any(axis=1)
        _append_if_supported(
            out,
            Predicate(f"identity:any_role:champ:{int(champ_id)}", mask, "identity_champion"),
            min_support=min_support,
        )
    return out


def _pair_predicates(base: list[Predicate], *, min_support: int, limit: int = 200) -> list[Predicate]:
    out: list[Predicate] = []
    left = [p for p in base if p.family in {"context_axis", "support"}]
    right = [p for p in base if p.family.startswith("identity")]
    for a in left[:40]:
        for b in right[:80]:
            mask = np.asarray(a.mask, dtype=bool) & np.asarray(b.mask, dtype=bool)
            if int(mask.sum()) >= min_support:
                out.append(Predicate(f"pair:{a.label} && {b.label}", mask, "pair_context_identity"))
                if len(out) >= limit:
                    return out
    return out


def discover_slice_predicates(
    d: dict[str, np.ndarray],
    build_vocab: dict[str, int],
    *,
    min_support: int = 1000,
) -> list[Predicate]:
    base: list[Predicate] = []
    base.extend(_axis_band_predicates(d, min_support=min_support))
    base.extend(_support_predicates(d, min_support=min_support))
    base.extend(_identity_predicates(d, build_vocab, min_support=min_support))
    return base + _pair_predicates(base, min_support=min_support)


def _print_ranked_rows(title: str, rows: list[dict[str, object]]) -> None:
    print(f"\n## {title}")
    print(
        f"{'slice':>58s} {'n':>8s} {'gap':>9s} {'shrunk':>9s} "
        f"{'CI':>9s} {'score':>9s} {'suggestion':>36s}"
    )
    for row in rows:
        print(
            f"{str(row['label'])[:58]:>58s} {int(row['n']):8d} "
            f"{100*float(row['raw_gap']):+8.3f} {100*float(row['shrunk_gap']):+8.3f} "
            f"{100*float(row['ci_half_width']):8.3f} {100*float(row['score']):8.3f} "
            f"{str(row['suggestion'])[:36]:>36s}"
        )


def run_discovery(
    sr: SideRows,
    build: dict[str, int],
    *,
    min_support: int = 1000,
    shrink_strength: float = 1000.0,
    top: int = 50,
    bootstrap: int = 0,
) -> None:
    d = sr.d
    predicates = discover_slice_predicates(d, build, min_support=min_support)
    print(f"discovered predicates: {len(predicates):,}   min_support={min_support:,}")
    rows = rank_residual_slices(
        predicates,
        d["model"],
        d["emp"],
        min_support=min_support,
        shrink_strength=shrink_strength,
        top=top,
        bootstrap=bootstrap,
    )
    _print_ranked_rows("Automated Residual Slice Discovery", rows)
    ma = multiaccuracy_residual_report(
        predicates,
        d["model"],
        d["emp"],
        min_support=min_support,
        shrink_strength=shrink_strength,
        top=min(top, 25),
    )
    _print_ranked_rows("Multiaccuracy-Style Residual Audit", ma)


def _row(label: str, mask: np.ndarray, sr: SideRows) -> dict:
    d = sr.d
    n = int(mask.sum())
    emp = float(d["emp"][mask].mean()) if n else float("nan")
    mod = float(d["model"][mask].mean()) if n else float("nan")
    base = float(d["base"][mask].mean()) if n else float("nan")
    return {"label": label, "n": n, "emp": emp, "model": mod, "base": base}


def _print_bins(title: str, rows: list[dict]) -> None:
    print(f"\n## {title}")
    print(f"{'bin':>22s} {'n':>8s} {'Emp WR':>8s} {'Model WR':>9s} {'Base WR':>8s} {'Gap':>7s}")
    for r in rows:
        if r["n"] == 0:
            print(f"{r['label']:>22s} {0:8d} {'-':>8s} {'-':>9s} {'-':>8s} {'-':>7s}")
            continue
        print(f"{r['label']:>22s} {r['n']:8d} {100*r['emp']:7.2f}% {100*r['model']:8.2f}% "
              f"{100*r['base']:7.2f}% {100*(r['model']-r['emp']):+6.2f}")


def _effect_summary(
    *,
    label: str,
    axis: str,
    rows: list[dict],
    low_label: str,
    high_label: str,
    direction: str = "high-low",
) -> dict[str, object]:
    by_label = {str(row["label"]): row for row in rows}
    if low_label not in by_label or high_label not in by_label:
        raise ValueError(f"Missing semantic effect rows for {label}: {low_label}, {high_label}")
    low = by_label[low_label]
    high = by_label[high_label]
    if direction == "high-low":
        emp_effect = float(high["emp"]) - float(low["emp"])
        model_effect = float(high["model"]) - float(low["model"])
        base_effect = float(high["base"]) - float(low["base"])
    elif direction == "low-high":
        emp_effect = float(low["emp"]) - float(high["emp"])
        model_effect = float(low["model"]) - float(high["model"])
        base_effect = float(low["base"]) - float(high["base"])
    else:
        raise ValueError("direction must be high-low or low-high")
    low_gap = float(low["model"]) - float(low["emp"])
    high_gap = float(high["model"]) - float(high["emp"])
    return {
        "label": label,
        "axis": axis,
        "direction": direction,
        "low_label": low_label,
        "high_label": high_label,
        "low_n": int(low["n"]),
        "high_n": int(high["n"]),
        "emp_effect": emp_effect,
        "model_effect": model_effect,
        "base_effect": base_effect,
        "delta_gap": model_effect - emp_effect,
        "low_gap": low_gap,
        "high_gap": high_gap,
        "max_abs_endpoint_gap": max(abs(low_gap), abs(high_gap)),
    }


def _semantic_summary(sr: SideRows, build: dict[str, int]) -> dict[str, object]:
    d = sr.d
    AR, ADOFF, MR = build["ar_tank"], build["ad_off_tank"], build["mr_tank"]
    UTIL = (build["utility_enchanter"], build["utility_protection"])

    rows: list[dict[str, object]] = []
    mal = sr.has_identity(CH["Malphite"], AR)
    mal_top = sr.has_identity_role(CH["Malphite"], 0, AR)
    mal_rows = [*_quintile_rows(values=d["enemy_phys"], mask=mal, sr=sr, prefix="all")]
    mal_rows += [*_quintile_rows(values=d["enemy_phys"], mask=mal_top, sr=sr, prefix="TOP")]
    rows.append(
        _effect_summary(
            label="Malphite ar_tank",
            axis="enemy physical share",
            rows=mal_rows,
            low_label="all 0-20",
            high_label="all 80-100",
        )
    )
    rows.append(
        _effect_summary(
            label="Malphite TOP ar_tank",
            axis="enemy physical share",
            rows=mal_rows,
            low_label="TOP 0-20",
            high_label="TOP 80-100",
        )
    )

    dmg_band = _quintile_ids(d["focus_dmg"])
    heal_band = _quintile_ids(d["enemy_heal"])
    damage_heal_rows = []
    for di, dl in enumerate(QUINTILE_LABELS):
        for hi, hl in enumerate(QUINTILE_LABELS):
            damage_heal_rows.append(
                _row(f"dmg{dl} heal{hl}", (dmg_band == di) & (heal_band == hi), sr)
            )
    rows.append(
        _effect_summary(
            label="Low own damage into enemy heal/shield",
            axis="enemy heal/shield pressure",
            rows=damage_heal_rows,
            low_label="dmg0-20 heal0-20",
            high_label="dmg0-20 heal80-100",
        )
    )

    sion_rows = list(
        _quintile_rows(
            values=d["enemy_dmg"],
            mask=sr.has_identity_role(CH["Sion"], 0, ADOFF),
            sr=sr,
            prefix="dmg",
        )
    )
    rows.append(
        _effect_summary(
            label="Sion TOP ad_off_tank",
            axis="enemy damage pressure",
            rows=sion_rows,
            low_label="dmg 0-20",
            high_label="dmg 80-100",
            direction="low-high",
        )
    )

    mundo_rows = []
    for blab, bid in (("ad_off_tank", ADOFF), ("mr_tank", MR)):
        mundo_rows += list(
            _quintile_rows(
                values=d["enemy_magic"],
                mask=sr.has_identity(CH["DrMundo"], bid),
                sr=sr,
                prefix=blab,
            )
        )
    for blab in ("ad_off_tank", "mr_tank"):
        rows.append(
            _effect_summary(
                label=f"DrMundo {blab}",
                axis="enemy magic share",
                rows=mundo_rows,
                low_label=f"{blab} 0-20",
                high_label=f"{blab} 80-100",
            )
        )

    enchanters = ("Sona", "Karma", "Lulu", "Zilean")
    selected = np.zeros(d["emp"].shape[0], dtype=bool)
    for name in enchanters:
        for bid in UTIL:
            selected |= sr.has_identity_role(CH[name], 4, bid)
    skirmish = d["ally_skirmish"]
    enchanter_rows = [
        _row("selected enchanters 0", selected & (skirmish == 0), sr),
        _row("selected enchanters 1", selected & (skirmish == 1), sr),
        _row("selected enchanters 2+", selected & (skirmish >= 2), sr),
    ]
    rows.append(
        _effect_summary(
            label="Selected enchanters",
            axis="skirmish allies",
            rows=enchanter_rows,
            low_label="selected enchanters 0",
            high_label="selected enchanters 2+",
        )
    )

    delta_abs = [abs(float(row["delta_gap"])) for row in rows]
    endpoint_abs = [float(row["max_abs_endpoint_gap"]) for row in rows]
    aggregate = {
        "n_effects": len(rows),
        "mean_abs_delta_gap": float(np.mean(delta_abs)) if delta_abs else float("nan"),
        "max_abs_delta_gap": float(np.max(delta_abs)) if delta_abs else float("nan"),
        "mean_abs_endpoint_gap": float(np.mean(endpoint_abs)) if endpoint_abs else float("nan"),
        "max_abs_endpoint_gap": float(np.max(endpoint_abs)) if endpoint_abs else float("nan"),
    }
    return {"aggregate": aggregate, "effects": rows}


def _write_summary_json(
    path: Path,
    *,
    sr: SideRows,
    build: dict[str, int],
    model_path: Path,
    conditioned: bool,
    active_splits: tuple[str, ...],
) -> dict[str, object]:
    summary = _semantic_summary(sr, build)
    payload = {
        "model_path": str(model_path),
        "conditioned": conditioned,
        "side_rows": int(sr.d["emp"].size),
        "splits": list(active_splits),
        "semantic_summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _resolve_active_splits(
    *,
    discover_slices: bool,
    split: str,
    splits: tuple[str, ...] | None,
) -> tuple[str, ...]:
    active = splits if splits is not None else ((split,) if discover_slices else SPLITS)
    if not active:
        raise ValueError("At least one split must be selected.")
    unknown = [name for name in active if name not in SPLITS]
    if unknown:
        raise ValueError(f"Unknown splits: {', '.join(unknown)}")
    if len(set(active)) != len(active):
        raise ValueError("Duplicate splits are not allowed.")
    if discover_slices and active != ("val",):
        raise ValueError("Slice discovery is validation-only; use --splits val or omit --splits.")
    return active


def write_semantic_summary(
    model_path: Path | str,
    output_path: Path | str,
    *,
    dataset_config: DatasetConfig | None = None,
    splits: tuple[str, ...] = ("val",),
    device: str = "auto",
) -> dict[str, object]:
    """Write semantic effect gaps for a checkpoint and return the JSON payload."""
    dataset_config = dataset_config or DatasetConfig()
    active_splits = _resolve_active_splits(
        discover_slices=False,
        split="val",
        splits=splits,
    )
    resolved_device = resolve_device(device)
    model_path = Path(model_path)
    model, _, strength = load_hgnn_model(model_path, device=resolved_device)
    conditioned = getattr(model, "identity_conditioned_context_enabled", False)
    if not (getattr(model, "context_enabled", False) or conditioned):
        raise ValueError("Loaded model has no context head (identity_context_dim=0).")

    build = {b: i for i, b in enumerate(identity_meta(dataset_config)["build_vocab"])}
    loaded_splits = load_splits(dataset_config, require_counts=True)
    sr = SideRows()
    for name in active_splits:
        sp = loaded_splits[name]
        full, ctxl = _predict(
            model,
            sp,
            strength=strength,
            device=resolved_device,
            conditioned=conditioned,
        )
        sr.add_split(sp, full, ctxl)
    sr.finalize()
    return _write_summary_json(
        Path(output_path),
        sr=sr,
        build=build,
        model_path=model_path,
        conditioned=conditioned,
        active_splits=active_splits,
    )


def _quintile_ids(values: np.ndarray) -> np.ndarray:
    """Global side-row quintile id for a continuous context axis."""
    edges = np.quantile(values, [0.20, 0.40, 0.60, 0.80])
    return np.searchsorted(edges, values, side="right")


def _quintile_rows(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    sr: SideRows,
    prefix: str = "",
) -> list[dict]:
    band = _quintile_ids(values)
    label_prefix = f"{prefix} " if prefix else ""
    return [
        _row(f"{label_prefix}{label}", mask & (band == idx), sr)
        for idx, label in enumerate(QUINTILE_LABELS)
    ]


def run(sr: SideRows, build: dict[str, int]) -> None:
    d = sr.d
    AR, ADOFF, MR = build["ar_tank"], build["ad_off_tank"], build["mr_tank"]
    UTIL = (build["utility_enchanter"], build["utility_protection"])

    # Malphite ar_tank vs expected enemy physical (all roles + TOP only).
    mal = sr.has_identity(CH["Malphite"], AR)
    mal_top = sr.has_identity_role(CH["Malphite"], 0, AR)
    rows = [*_quintile_rows(values=d["enemy_phys"], mask=mal, sr=sr, prefix="all")]
    rows += [*_quintile_rows(values=d["enemy_phys"], mask=mal_top, sr=sr, prefix="TOP")]
    _print_bins("Malphite ar_tank Vs Expected Enemy Physical", rows)

    # Own damage into enemy heal/shield (global quintiles over all side rows).
    dmg_band = _quintile_ids(d["focus_dmg"])
    heal_band = _quintile_ids(d["enemy_heal"])
    cells = []
    for di, dl in enumerate(QUINTILE_LABELS):
        for hi, hl in enumerate(QUINTILE_LABELS):
            dmask = dmg_band == di
            hmask = heal_band == hi
            cells.append(_row(f"dmg{dl} heal{hl}", dmask & hmask, sr))
    _print_bins("Own Damage Into Enemy Heal/Shield", cells)

    # Sion TOP ad_off_tank vs expected enemy damage (global percentiles).
    sion = sr.has_identity_role(CH["Sion"], 0, ADOFF)
    srows = list(_quintile_rows(values=d["enemy_dmg"], mask=sion, sr=sr, prefix="dmg"))
    _print_bins("Sion TOP ad_off_tank Vs Expected Enemy Damage", srows)

    # DrMundo vs expected enemy magic, by build.
    rows = []
    for blab, bid in (("ad_off_tank", ADOFF), ("mr_tank", MR)):
        m = sr.has_identity(CH["DrMundo"], bid)
        rows += list(_quintile_rows(values=d["enemy_magic"], mask=m, sr=sr, prefix=blab))
    _print_bins("DrMundo Vs Expected Enemy Magic", rows)

    # Enchanters with skirmish-heavy allies (0 / 1 / 2+).
    enchanters = ("Sona", "Karma", "Lulu", "Zilean")
    sel = np.zeros(d["emp"].shape[0], dtype=bool)
    for name in enchanters:
        for b in UTIL:
            sel |= sr.has_identity_role(CH[name], 4, b)
    other = np.zeros_like(sel)
    for b in UTIL:
        other |= (d["build"][:, 4] == b)
    other &= ~sel
    sk = d["ally_skirmish"]
    rows = []
    for lab, grp in (("selected enchanters", sel), ("other utility supports", other)):
        for cnt, cmask in (("0", sk == 0), ("1", sk == 1), ("2+", sk >= 2)):
            rows.append(_row(f"{lab} {cnt}", grp & cmask, sr))
    _print_bins("Enchanters With Skirmish-Heavy Allies", rows)
    perrows = []
    for name in enchanters:
        em = np.zeros_like(sel)
        for b in UTIL:
            em |= sr.has_identity_role(CH[name], 4, b)
        for cnt, cmask in (("0", sk == 0), ("1", sk == 1), ("2+", sk >= 2)):
            perrows.append(_row(f"{name} {cnt}", em & cmask, sr))
    _print_bins("Per-Enchanter With Skirmish-Heavy Allies", perrows)

    # Armor tanks sweep: global quintiles (build ar_tank, all roles).
    _print_sweep("Armor Tanks Into Expected Enemy Physical", sr,
                 ["Maokai", "Malphite", "Sion", "Ornn", "Nautilus", "Shen", "Poppy"],
                 AR, d["enemy_phys"])
    # MR tanks sweep: global quintiles (build mr_tank, all roles).
    _print_sweep("MR Tanks Into Expected Enemy Magic", sr,
                 ["Galio", "Sion", "Ornn", "Chogath", "Shen", "Nautilus", "DrMundo", "Amumu", "Maokai", "Malphite"],
                 MR, d["enemy_magic"])


def _print_sweep(title, sr, champs, build_id, axis):
    print(f"\n## {title} (global quintile bands)")
    print(f"{'Champion':>10s} {'bin':>8s} {'n':>8s} {'Emp WR':>8s} "
          f"{'Model WR':>9s} {'Base WR':>8s} {'Gap':>7s}")
    band = _quintile_ids(axis)
    for name in champs:
        m = sr.has_identity(CH[name], build_id)
        for idx, label in enumerate(QUINTILE_LABELS):
            r = _row(label, m & (band == idx), sr)
            if r["n"] == 0:
                print(f"{name:>10s} {label:>8s} {0:8d} {'-':>8s} {'-':>9s} {'-':>8s} {'-':>7s}")
                continue
            print(f"{name:>10s} {label:>8s} {r['n']:8d} {100*r['emp']:7.2f}% "
                  f"{100*r['model']:8.2f}% {100*r['base']:7.2f}% {100*(r['model']-r['emp']):+6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None, help="HGNN checkpoint to audit")
    parser.add_argument(
        "--discover-slices",
        action="store_true",
        help="Run validation-only automated residual slice discovery.",
    )
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=None,
        help=(
            "Splits to include in the printed audit and summary JSON. Defaults to val "
            "for --discover-slices, otherwise train val test."
        ),
    )
    parser.add_argument("--min-support", type=int, default=1000)
    parser.add_argument("--shrink-strength", type=float, default=1000.0)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write machine-readable semantic effect gaps for the audited model.",
    )
    args = parser.parse_args()

    setup_logging_config()
    logging.getLogger().setLevel(logging.WARNING)
    device = resolve_device("auto")
    model_path = args.model_path or TrainConfig().model_path
    model, _, strength = load_hgnn_model(model_path, device=device)
    conditioned = getattr(model, "identity_conditioned_context_enabled", False)
    if not (getattr(model, "context_enabled", False) or conditioned):
        raise SystemExit("Loaded model has no context head (identity_context_dim=0).")

    build = {b: i for i, b in enumerate(identity_meta(DatasetConfig())["build_vocab"])}
    splits = load_splits(DatasetConfig(), require_counts=True)
    sr = SideRows()
    try:
        active_splits = _resolve_active_splits(
            discover_slices=args.discover_slices,
            split=args.split,
            splits=tuple(args.splits) if args.splits is not None else None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for name in active_splits:
        sp = splits[name]
        full, ctxl = _predict(model, sp, strength=strength, device=device, conditioned=conditioned)
        sr.add_split(sp, full, ctxl)
    sr.finalize()
    print(f"model: {Path(model_path).name}   conditioned={conditioned}   "
          f"side rows ({'+'.join(active_splits)}): {sr.d['emp'].size:,}")
    if args.discover_slices:
        run_discovery(
            sr,
            build,
            min_support=args.min_support,
            shrink_strength=args.shrink_strength,
            top=args.top,
            bootstrap=args.bootstrap,
        )
    else:
        run(sr, build)
    if args.summary_json is not None:
        _write_summary_json(
            args.summary_json,
            sr=sr,
            build=build,
            model_path=Path(model_path),
            conditioned=conditioned,
            active_splits=tuple(active_splits),
        )


if __name__ == "__main__":
    main()
