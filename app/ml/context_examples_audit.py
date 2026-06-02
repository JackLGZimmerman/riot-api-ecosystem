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
import logging
from pathlib import Path

import numpy as np
import torch

from app.classification.embeddings.config import (
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
    for i in range(0, n, 32768):
        s = slice(i, i + 32768)
        raw = sp.identity_context_raw[s] if (conditioned and sp.identity_context_raw is not None) else None
        inp = build_hgnn_inputs(
            champion_id=sp.champion_id[s], build_id=sp.build_id[s], win_rate=sp.win_rate[s],
            matchup_1v1=sp.matchup_1v1[s], synergy_2vx=sp.synergy_2vx[s], p1_cnt=sp.p1_cnt[s],
            m1v1_cnt=sp.m1v1_cnt[s], s2vx_cnt=sp.s2vx_cnt[s], strength=strength,
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
        )}

    def add_split(self, sp: SplitData, full: np.ndarray, ctxl: np.ndarray) -> None:
        cid = np.asarray(sp.champion_id)
        bid = np.asarray(sp.build_id)
        ctx = np.asarray(sp.identity_context, dtype=np.float32)
        bw = np.asarray(sp.blue_win).astype(np.float64)
        pblue = 1.0 / (1.0 + np.exp(-full))
        base_blue = 1.0 / (1.0 + np.exp(-(full - ctxl)))
        skirmish = np.isin(cid, list(SKIRMISHERS))  # [N, 10]
        for focus, enemy in ((slice(0, 5), slice(5, 10)), (slice(5, 10), slice(0, 5))):
            fctx, ectx = ctx[:, focus, :], ctx[:, enemy, :]
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

    def finalize(self) -> None:
        self.d = {k: np.concatenate(v) for k, v in self.cols.items()}

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
    for name in SPLITS:
        sp = splits[name]
        full, ctxl = _predict(model, sp, strength=strength, device=device, conditioned=conditioned)
        sr.add_split(sp, full, ctxl)
    sr.finalize()
    print(f"model: {Path(model_path).name}   conditioned={conditioned}   "
          f"side rows (all splits): {sr.d['emp'].size:,}")
    run(sr, build)


if __name__ == "__main__":
    main()
