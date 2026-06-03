"""Regenerate the HGNN context examples audit with model prediction gaps."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.classification.embeddings.static_champion import load_static_by_id
from app.ml.config import DatasetConfig
from app.ml.dataset import SplitData, identity_meta, load_splits
from app.ml.hgnn_model import HGNNWinModel, build_hgnn_inputs, load_hgnn_model, resolve_device
from app.ml.train import _SidecarGatherer, _build_sidecar_gatherer, _model_uses_sidecar

DEFAULT_CONTEXT_CACHE_DIR = Path("app/ml/data/cache")
DEFAULT_MODEL_CACHE_DIR = Path("app/ml/data/experiments/semantic_context_compact_cache")
DEFAULT_MODEL_PATH = Path("app/ml/data/experiments/semantic_context_compact_run/model.pt")
DEFAULT_OUTPUT_PATH = Path("app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md")
DEFAULT_PREDICTION_CACHE = Path(
    "app/ml/data/experiments/semantic_context_compact_run/audit_final_blue_probability.npy"
)

POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
CONTEXT_AXIS_INDEX = {
    "physical": 0,
    "magic": 1,
    "damage": 5,
    "damage_taken": 9,
    "heal_shield": 10,
    "cc": 11,
    "siege": 12,
    "scaling": 13,
}
TANK_BUILD_LABELS = frozenset({"ar_tank", "mr_tank", "ad_off_tank", "ap_off_tank"})
SKIRMISH_CHAMPIONS = frozenset({887, 24, 39, 114, 77, 5})  # Gwen, Jax, Irelia, Fiora, Udyr, XinZhao.
SELECTED_ENCHANTERS = frozenset({37, 43, 117, 26})  # Sona, Karma, Lulu, Zilean.
SELECTED_ENCHANTER_BUILDS = ("utility_enchanter", "utility_protection")


@dataclass(frozen=True)
class BinSpec:
    label: str
    predicate: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class AuditSpec:
    section: str
    title: str
    read: str
    axis: str
    bins: tuple[BinSpec, ...]
    champions: tuple[int, ...] = ()
    positions: tuple[str, ...] = ()
    builds: tuple[str, ...] = ()
    focus_condition: str | None = None


@dataclass(frozen=True)
class AuditBin:
    label: str
    n: int
    empirical_wr: float
    hgnn_wr: float
    gap: float


@dataclass(frozen=True)
class AuditRow:
    spec: AuditSpec
    bins: tuple[AuditBin, ...]

    @property
    def endpoint_effect(self) -> float:
        populated = [row for row in self.bins if row.n > 0]
        if len(populated) < 2:
            return float("nan")
        return populated[-1].empirical_wr - populated[0].empirical_wr

    @property
    def hgnn_endpoint_effect(self) -> float:
        populated = [row for row in self.bins if row.n > 0]
        if len(populated) < 2:
            return float("nan")
        return populated[-1].hgnn_wr - populated[0].hgnn_wr


def le(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: axis <= float(value)


def lt(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: axis < float(value)


def eq(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: axis == float(value)


def ge(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: axis >= float(value)


def gt(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: axis > float(value)


def between(lower: float, upper: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: (axis > float(lower)) & (axis < float(upper))


def count_bins() -> tuple[BinSpec, ...]:
    return (
        BinSpec("0", eq(0)),
        BinSpec("1", eq(1)),
        BinSpec("2", eq(2)),
        BinSpec(">= 3", ge(3)),
    )


def range_count_bins() -> tuple[BinSpec, ...]:
    return (
        BinSpec("<= 1", le(1)),
        BinSpec("2", eq(2)),
        BinSpec("3", eq(3)),
        BinSpec(">= 4", ge(4)),
    )


def continuous_bins(a: float, b: float, c: float, d: float) -> tuple[BinSpec, ...]:
    return (
        BinSpec(f"<= {a:.3f}", le(a)),
        BinSpec(f"{a:.3f}-{b:.3f}", between(a, b)),
        BinSpec(f"{b:.3f}-{c:.3f}", between(b, c)),
        BinSpec(f"{c:.3f}-{d:.3f}", between(c, d)),
        BinSpec(f">= {d:.3f}", ge(d)),
    )


PHYSICAL_BINS = continuous_bins(0.387, 0.448, 0.508, 0.557)
MAGIC_BINS = continuous_bins(0.373, 0.423, 0.486, 0.549)
DAMAGE_BINS = continuous_bins(0.739, 0.764, 0.785, 0.813)
TAKEN_BINS = continuous_bins(0.639, 0.667, 0.692, 0.721)
HEAL_BINS = continuous_bins(0.028, 0.077, 0.200, 0.202)
CC_BINS = continuous_bins(0.374, 0.429, 0.479, 0.539)
SIEGE_BINS = continuous_bins(0.441, 0.471, 0.499, 0.530)
SCALING_BINS = continuous_bins(0.829, 0.841, 0.852, 0.863)


def audit_specs() -> tuple[AuditSpec, ...]:
    headline = "Headline Trajectory Audit Tables"
    richer = "Richer Composition Trajectory Tables"
    retained = "Retained Prior And User-Requested Trajectory Tables"
    lower = "Inspected Lower-Signal Trajectory Tables"
    return (
        AuditSpec(headline, "Yone TOP `on_hit` vs enemy siege", "Melee carry into siege and poke.", "enemy_siege", SIEGE_BINS, champions=(777,), positions=("TOP",), builds=("on_hit",)),
        AuditSpec(headline, "Graves JUNGLE `lethality` vs enemy damage", "Burst jungler into high enemy damage.", "enemy_damage", DAMAGE_BINS, champions=(104,), positions=("JUNGLE",), builds=("lethality",)),
        AuditSpec(headline, "Yone MIDDLE `on_hit` vs enemy siege", "Same melee-carry pattern across lane.", "enemy_siege", SIEGE_BINS, champions=(777,), positions=("MIDDLE",), builds=("on_hit",)),
        AuditSpec(headline, "Swain UTILITY `ap_off_tank` vs enemy scaling", "Drain support into scaling enemies.", "enemy_scaling", SCALING_BINS, champions=(50,), positions=("UTILITY",), builds=("ap_off_tank",)),
        AuditSpec(headline, "Nautilus UTILITY `mr_tank` with ally damage", "Engage support with damage behind it.", "ally_damage", DAMAGE_BINS, champions=(111,), positions=("UTILITY",), builds=("mr_tank",)),
        AuditSpec(headline, "Galio MIDDLE `mr_tank` vs enemy magic", "Anti-magic tank itemization.", "enemy_magic", MAGIC_BINS, champions=(3,), positions=("MIDDLE",), builds=("mr_tank",)),
        AuditSpec(headline, "Malphite TOP `ar_tank` vs enemy physical", "Armor tank into AD-heavy enemies.", "enemy_physical", PHYSICAL_BINS, champions=(54,), positions=("TOP",), builds=("ar_tank",)),
        AuditSpec(headline, "Swain MIDDLE any build vs enemy range", "Static range pressure on short-range battlemage.", "enemy_ranged_count", range_count_bins(), champions=(50,), positions=("MIDDLE",)),
        AuditSpec(headline, "Nilah BOTTOM any build vs enemy range", "Melee bot lane into range-heavy teams.", "enemy_ranged_count", range_count_bins(), champions=(895,), positions=("BOTTOM",)),
        AuditSpec(richer, "Swain BOTTOM `ability_power` vs enemy frontline count", "Swain gets better as enemies add durable targets.", "enemy_frontline_count", count_bins(), champions=(50,), positions=("BOTTOM",), builds=("ability_power",)),
        AuditSpec(richer, "Swain MIDDLE any build vs enemy frontline count", "Same Swain anti-frontline pattern mid.", "enemy_frontline_count", count_bins(), champions=(50,), positions=("MIDDLE",)),
        AuditSpec(richer, "Swain UTILITY any build vs enemy frontline count", "Support Swain also improves into frontline-heavy teams.", "enemy_frontline_count", count_bins(), champions=(50,), positions=("UTILITY",)),
        AuditSpec(richer, "Lillia JUNGLE `ap_off_tank` vs enemy frontline count", "Sustained AP skirmisher into beefy teams.", "enemy_frontline_count", count_bins(), champions=(876,), positions=("JUNGLE",), builds=("ap_off_tank",)),
        AuditSpec(richer, "Morgana UTILITY `ability_power` vs enemy frontline count", "Zone and control support benefits when enemies walk into space.", "enemy_frontline_count", count_bins(), champions=(25,), positions=("UTILITY",), builds=("ability_power",)),
        AuditSpec(richer, "Vayne BOTTOM `on_hit` vs enemy frontline count", "Classic anti-tank marksman pattern.", "enemy_frontline_count", count_bins(), champions=(67,), positions=("BOTTOM",), builds=("on_hit",)),
        AuditSpec(richer, "Alistar UTILITY `ar_tank` vs enemy burst count", "Durable engage support punished by multiple burst threats.", "enemy_burst_count", count_bins(), champions=(12,), positions=("UTILITY",), builds=("ar_tank",)),
        AuditSpec(richer, "Sion TOP `mr_tank` vs enemy burst count", "High-HP tank loses into concentrated burst threats.", "enemy_burst_count", count_bins(), champions=(14,), positions=("TOP",), builds=("mr_tank",)),
        AuditSpec(richer, "Qiyana JUNGLE `lethality` vs enemy burst count", "Assassin jungler into enemy burst stacking.", "enemy_burst_count", count_bins(), champions=(246,), positions=("JUNGLE",), builds=("lethality",)),
        AuditSpec(richer, "Rell UTILITY `utility_protection` vs enemy burst count", "All-in support punished by burst-heavy enemies.", "enemy_burst_count", count_bins(), champions=(526,), positions=("UTILITY",), builds=("utility_protection",)),
        AuditSpec(richer, "Corki BOTTOM `crit` vs enemy burst count", "Fragile carry into burst-heavy enemies.", "enemy_burst_count", count_bins(), champions=(42,), positions=("BOTTOM",), builds=("crit",)),
        AuditSpec(richer, "Malphite TOP `ar_tank` vs heavy damage-taken count", "Armor tank loses into teams with multiple high-soak targets.", "enemy_heavy_taken_count", count_bins(), champions=(54,), positions=("TOP",), builds=("ar_tank",)),
        AuditSpec(richer, "Poppy JUNGLE any build vs enemy high-HP count", "Anti-dash/control jungler into high-HP enemy teams.", "enemy_high_hp_count", count_bins(), champions=(78,), positions=("JUNGLE",)),
        AuditSpec(retained, "Malphite all roles `ar_tank` vs enemy physical", "Original armor-stack audit, retained beyond TOP-only.", "enemy_physical", PHYSICAL_BINS, champions=(54,), builds=("ar_tank",)),
        AuditSpec(retained, "Galio all roles `mr_tank` vs enemy magic", "Original anti-magic tank family, broader than MIDDLE-only.", "enemy_magic", MAGIC_BINS, champions=(3,), builds=("mr_tank",)),
        AuditSpec(retained, "Chogath all roles `mr_tank` vs enemy magic", "Smaller support, but unique scaling-tank anti-magic case.", "enemy_magic", MAGIC_BINS, champions=(31,), builds=("mr_tank",)),
        AuditSpec(retained, "Nautilus all roles `ar_tank` vs enemy physical", "Physical-heavy enemy teams remain a support-tank check.", "enemy_physical", PHYSICAL_BINS, champions=(111,), builds=("ar_tank",)),
        AuditSpec(retained, "Darius TOP any build vs enemy range count", "Static team range pressure, stronger than lane-only range.", "enemy_ranged_count", range_count_bins(), champions=(122,), positions=("TOP",)),
        AuditSpec(retained, "Darius TOP any build vs same-role range", "User-requested static melee/ranged lane audit.", "same_role_range", (BinSpec("<= 250", le(250)), BinSpec("> 250", gt(250))), champions=(122,), positions=("TOP",)),
        AuditSpec(retained, "MasterYi JUNGLE any build vs enemy hard CC", "User-requested low-CC audit; unique even though gap is modest.", "enemy_hard_cc_count", count_bins(), champions=(11,), positions=("JUNGLE",)),
        AuditSpec(retained, "Selected enchanters UTILITY with skirmish allies", "Original enchanter-with-skirmishers synergy probe.", "ally_skirmish_count", (BinSpec("0", eq(0)), BinSpec("1", eq(1)), BinSpec(">= 2", ge(2))), positions=("UTILITY",), focus_condition="selected_enchanter"),
        AuditSpec(retained, "Low own-damage teams vs enemy heal/shield", "Original low-damage into sustain audit.", "enemy_heal_shield", HEAL_BINS, focus_condition="low_own_damage"),
        AuditSpec(retained, "Sion TOP `ad_off_tank` vs enemy damage", "Retained as a tank-into-damage pressure sanity check.", "enemy_damage", DAMAGE_BINS, champions=(14,), positions=("TOP",), builds=("ad_off_tank",)),
        AuditSpec(retained, "DrMundo all roles `ad_off_tank` vs enemy magic", "Original Mundo magic-share probe, low gap but distinct champion.", "enemy_magic", MAGIC_BINS, champions=(36,), builds=("ad_off_tank",)),
        AuditSpec(retained, "DrMundo all roles `mr_tank` vs enemy magic", "Retained to compare MR-tank Mundo against Galio/Chogath.", "enemy_magic", MAGIC_BINS, champions=(36,), builds=("mr_tank",)),
        AuditSpec(lower, "Focus HP `<= 2309` vs enemy burst count", "Broad HP-vs-burst check; useful but lower signal than champion-specific rows.", "enemy_burst_count", count_bins(), focus_condition="focus_hp_low"),
        AuditSpec(lower, "Focus HP `>= 2478` vs enemy burst count", "High-HP slots also drop into burst stacks, so champion/build specificity matters.", "enemy_burst_count", count_bins(), focus_condition="focus_hp_high"),
        AuditSpec(lower, "Swain MIDDLE any build vs heavy damage-taken count", "Swain into heavy damage-taken count was inspected; tank/frontline count is much stronger.", "enemy_heavy_taken_count", count_bins(), champions=(50,), positions=("MIDDLE",)),
        AuditSpec(lower, "Swain BOTTOM `ability_power` vs heavy damage-taken count", "Same result bot: tank/frontline count is the better Swain audit.", "enemy_heavy_taken_count", count_bins(), champions=(50,), positions=("BOTTOM",), builds=("ability_power",)),
    )


class AuditData:
    def __init__(
        self,
        *,
        context_cache_dir: Path,
        blue_probability: np.ndarray,
    ) -> None:
        self.cache_dir = context_cache_dir
        meta = json.loads((context_cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
        self.n_games = int(meta["n_games"])
        if blue_probability.shape != (self.n_games,):
            raise ValueError("blue probability array must match the context cache n_games")
        self.build_vocab = tuple(meta["identity"]["build_vocab"])
        self.build_to_idx = {label: idx for idx, label in enumerate(self.build_vocab)}
        self.blue_win = np.load(context_cache_dir / "blue_win.npy", mmap_mode="r")[: self.n_games]
        self.champion_id = np.load(context_cache_dir / "champion_id.npy", mmap_mode="r")[: self.n_games]
        self.build_id = np.load(context_cache_dir / "build_id.npy", mmap_mode="r")[: self.n_games]
        self.context_raw = np.load(context_cache_dir / "identity_context_raw.npy", mmap_mode="r")[: self.n_games]
        self.blue_probability = np.asarray(blue_probability, dtype=np.float64)
        self._axis_cache: dict[str, np.ndarray] = {}
        self._hp_lookup, self._range_lookup = _static_lookups()
        self._slot_hp_cache: np.ndarray | None = None
        self._slot_range_cache: np.ndarray | None = None

    @property
    def labels(self) -> np.ndarray:
        return np.concatenate(
            [
                np.repeat(self.blue_win[:, None].astype(np.float64), 5, axis=1),
                np.repeat((1.0 - self.blue_win[:, None]).astype(np.float64), 5, axis=1),
            ],
            axis=1,
        )

    @property
    def predictions(self) -> np.ndarray:
        p = self.blue_probability[:, None]
        return np.concatenate(
            [
                np.repeat(p, 5, axis=1),
                np.repeat(1.0 - p, 5, axis=1),
            ],
            axis=1,
        )

    @property
    def slot_hp(self) -> np.ndarray:
        if self._slot_hp_cache is None:
            self._slot_hp_cache = self._hp_lookup[self.champion_id]
        return self._slot_hp_cache

    @property
    def slot_range(self) -> np.ndarray:
        if self._slot_range_cache is None:
            self._slot_range_cache = self._range_lookup[self.champion_id]
        return self._slot_range_cache

    def axis(self, name: str) -> np.ndarray:
        if name not in self._axis_cache:
            self._axis_cache[name] = self._build_axis(name)
        return self._axis_cache[name]

    def _build_axis(self, name: str) -> np.ndarray:
        if name.startswith("enemy_") and name.removeprefix("enemy_") in CONTEXT_AXIS_INDEX:
            return self._team_context(CONTEXT_AXIS_INDEX[name.removeprefix("enemy_")], enemy=True)
        if name.startswith("ally_") and name.removeprefix("ally_") in CONTEXT_AXIS_INDEX:
            return self._team_context(CONTEXT_AXIS_INDEX[name.removeprefix("ally_")], enemy=False)
        if name == "enemy_burst_count":
            non_tank = ~np.isin(self.build_id, [self.build_to_idx[label] for label in TANK_BUILD_LABELS])
            burst_slot = (self.context_raw[:, :, CONTEXT_AXIS_INDEX["damage"]] >= 0.952) & non_tank
            return self._enemy_count(burst_slot)
        if name == "enemy_hard_cc_count":
            return self._enemy_count(self.context_raw[:, :, CONTEXT_AXIS_INDEX["cc"]] >= 0.696)
        if name == "enemy_frontline_count":
            tank_ids = [self.build_to_idx[label] for label in TANK_BUILD_LABELS]
            return self._enemy_count(np.isin(self.build_id, tank_ids))
        if name == "enemy_heavy_taken_count":
            return self._enemy_count(self.context_raw[:, :, CONTEXT_AXIS_INDEX["damage_taken"]] >= 0.822)
        if name == "enemy_high_hp_count":
            return self._enemy_count(self.slot_hp >= 2478.5)
        if name == "enemy_ranged_count":
            return self._enemy_count(self.slot_range > 250.0)
        if name == "same_role_range":
            return np.concatenate([self.slot_range[:, 5:], self.slot_range[:, :5]], axis=1)
        if name == "ally_skirmish_count":
            return self._ally_count(np.isin(self.champion_id, list(SKIRMISH_CHAMPIONS)))
        raise ValueError(f"unknown audit axis: {name}")

    def _team_context(self, dim: int, *, enemy: bool) -> np.ndarray:
        blue = self.context_raw[:, :5, dim].mean(axis=1)
        red = self.context_raw[:, 5:, dim].mean(axis=1)
        blue_focus = red if enemy else blue
        red_focus = blue if enemy else red
        return np.concatenate(
            [
                np.repeat(blue_focus[:, None], 5, axis=1),
                np.repeat(red_focus[:, None], 5, axis=1),
            ],
            axis=1,
        )

    @staticmethod
    def _side_count(slot_mask: np.ndarray, *, enemy: bool) -> np.ndarray:
        blue = slot_mask[:, :5].sum(axis=1).astype(np.float64)
        red = slot_mask[:, 5:].sum(axis=1).astype(np.float64)
        blue_focus = red if enemy else blue
        red_focus = blue if enemy else red
        return np.concatenate(
            [
                np.repeat(blue_focus[:, None], 5, axis=1),
                np.repeat(red_focus[:, None], 5, axis=1),
            ],
            axis=1,
        )

    def _enemy_count(self, slot_mask: np.ndarray) -> np.ndarray:
        return self._side_count(slot_mask, enemy=True)

    def _ally_count(self, slot_mask: np.ndarray) -> np.ndarray:
        return self._side_count(slot_mask, enemy=False)

    def focus_mask(self, spec: AuditSpec) -> np.ndarray:
        mask = np.ones((self.n_games, 10), dtype=bool)
        if spec.champions:
            mask &= np.isin(self.champion_id, list(spec.champions))
        if spec.positions:
            slot_mask = np.zeros(10, dtype=bool)
            for pos in spec.positions:
                idx = POSITIONS.index(pos)
                slot_mask[idx] = True
                slot_mask[idx + 5] = True
            mask &= slot_mask[None, :]
        if spec.builds:
            mask &= np.isin(self.build_id, [self.build_to_idx[label] for label in spec.builds])
        if spec.focus_condition == "low_own_damage":
            side_anchor = np.zeros(10, dtype=bool)
            side_anchor[[0, 5]] = True
            mask &= side_anchor[None, :]
            mask &= self.axis("ally_damage") <= 0.739
        elif spec.focus_condition == "focus_hp_low":
            mask &= self.slot_hp <= 2309.0
        elif spec.focus_condition == "focus_hp_high":
            mask &= self.slot_hp >= 2478.5
        elif spec.focus_condition == "selected_enchanter":
            mask &= np.isin(self.champion_id, list(SELECTED_ENCHANTERS))
            mask &= np.isin(
                self.build_id,
                [self.build_to_idx[label] for label in SELECTED_ENCHANTER_BUILDS],
            )
        elif spec.focus_condition is not None:
            raise ValueError(f"unknown focus condition: {spec.focus_condition}")
        return mask


def evaluate_specs(data: AuditData, specs: Sequence[AuditSpec]) -> tuple[AuditRow, ...]:
    labels = data.labels
    predictions = data.predictions
    rows: list[AuditRow] = []
    for spec in specs:
        focus = data.focus_mask(spec)
        axis = data.axis(spec.axis)
        bins: list[AuditBin] = []
        for bin_spec in spec.bins:
            mask = focus & bin_spec.predicate(axis)
            n = int(mask.sum())
            empirical = _mean_or_nan(labels[mask])
            hgnn = _mean_or_nan(predictions[mask])
            bins.append(
                AuditBin(
                    label=bin_spec.label,
                    n=n,
                    empirical_wr=empirical,
                    hgnn_wr=hgnn,
                    gap=hgnn - empirical,
                )
            )
        rows.append(AuditRow(spec=spec, bins=tuple(bins)))
    return tuple(rows)


def predict_blue_probabilities(
    *,
    model_path: Path,
    cache_dir: Path,
    batch_size: int,
    device: str,
) -> np.ndarray:
    device = resolve_device(device)
    model, config, strength = load_hgnn_model(model_path, device=device)
    model.eval()
    dataset_cfg = DatasetConfig(cache_dir=cache_dir)
    splits = load_splits(dataset_cfg, require_counts=True)
    gatherer = None
    if _model_uses_sidecar(config) and splits["train"].identity_static_sidecar is None:
        gatherer = _build_sidecar_gatherer(
            dataset_cfg,
            identity_meta(dataset_cfg),
            config,
            device=device,
        )
    outputs = [
        _predict_split(
            model,
            split,
            batch_size=batch_size,
            strength=strength,
            device=device,
            gatherer=gatherer,
        )
        for split in (splits["train"], splits["val"], splits["test"])
    ]
    return np.concatenate(outputs).astype(np.float64)


def _predict_split(
    model: HGNNWinModel,
    split: SplitData,
    *,
    batch_size: int,
    strength: float,
    device: str,
    gatherer: _SidecarGatherer | None = None,
) -> np.ndarray:
    out: list[np.ndarray] = []
    with torch.no_grad():
        n_rows = int(split.blue_win.size)
        for start in range(0, n_rows, batch_size):
            rows = slice(start, min(start + batch_size, n_rows))
            champion_id = split.champion_id[rows]
            build_id = split.build_id[rows]
            gathered_sidecar = (
                None
                if gatherer is None or split.identity_static_sidecar is not None
                else gatherer.gather(
                    torch.as_tensor(np.array(champion_id, copy=True), dtype=torch.long, device=device),
                    torch.as_tensor(np.array(build_id, copy=True), dtype=torch.long, device=device),
                )
            )
            if gathered_sidecar is None:
                identity_static_sidecar = (
                    None if split.identity_static_sidecar is None else split.identity_static_sidecar[rows]
                )
                identity_full_game_sidecar = (
                    None if split.identity_full_game_sidecar is None else split.identity_full_game_sidecar[rows]
                )
                identity_temporal_sidecar = (
                    None if split.identity_temporal_sidecar is None else split.identity_temporal_sidecar[rows]
                )
                identity_encoder_support = (
                    None if split.identity_encoder_support is None else split.identity_encoder_support[rows]
                )
            else:
                identity_static_sidecar = gathered_sidecar["identity_static_sidecar"]
                identity_full_game_sidecar = gathered_sidecar["identity_full_game_sidecar"]
                identity_temporal_sidecar = gathered_sidecar["identity_temporal_sidecar"]
                identity_encoder_support = gathered_sidecar["identity_encoder_support"]
            inputs = build_hgnn_inputs(
                champion_id=champion_id,
                build_id=build_id,
                win_rate=split.win_rate[rows],
                p1_cnt=split.p1_cnt[rows],
                strength=strength,
                matchup_1v1=split.matchup_1v1[rows],
                synergy_2vx=split.synergy_2vx[rows],
                m1v1_cnt=split.m1v1_cnt[rows],
                s2vx_cnt=split.s2vx_cnt[rows],
                identity_static_sidecar=identity_static_sidecar,
                identity_full_game_sidecar=identity_full_game_sidecar,
                identity_temporal_sidecar=identity_temporal_sidecar,
                identity_encoder_support=identity_encoder_support,
                include_relationship_features=bool(model.config.use_relationship_integrations),
                device=device,
            )
            logits = model(**inputs)["final_logit"]
            out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out)


def load_or_predict_blue_probabilities(
    *,
    model_path: Path,
    model_cache_dir: Path,
    prediction_cache: Path,
    n_games: int,
    refresh: bool,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if not refresh and prediction_cache.exists():
        cached = np.load(prediction_cache)
        if cached.shape == (n_games,):
            return np.asarray(cached, dtype=np.float64)
    probabilities = predict_blue_probabilities(
        model_path=model_path,
        cache_dir=model_cache_dir,
        batch_size=batch_size,
        device=device,
    )
    if probabilities.shape != (n_games,):
        raise ValueError("predicted probability count does not match context cache n_games")
    prediction_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_cache, probabilities.astype(np.float32))
    return probabilities


def render_audit(
    rows: Sequence[AuditRow],
    *,
    model_path: Path,
    model_cache_dir: Path,
    context_cache_dir: Path,
    updated: str | None = None,
) -> str:
    updated = updated or date.today().isoformat()
    by_section: dict[str, list[AuditRow]] = {}
    for row in rows:
        by_section.setdefault(row.spec.section, []).append(row)
    lines = [
        "# HGNN Context Examples Audit",
        "",
        f"Updated: {updated}.",
        "",
        "This audit joins the empirical focus-side context examples to the trained "
        "semantic HGNN predictions for the same cached games. Each bin reports "
        "`n / empirical WR / HGNN WR / gap`, where gap is "
        "`HGNN WR - empirical WR`. Zero gap is the target.",
        "",
        "## Scope And Threshold Definitions",
        "",
        f"- Context source: `{context_cache_dir}` side-row arrays, all splits combined.",
        f"- HGNN model: `{model_path}`.",
        f"- HGNN cache: `{model_cache_dir}`.",
        "- HGNN WR uses raw `final_logit` probabilities; report-only temperature scaling is not applied.",
        "- Side rows audited: 2,862,626.",
        "- Model-alignment rows score blue slots with `P(blue wins)` and red slots with `1 - P(blue wins)`.",
        "- Continuous thresholds are global side-row team-average percentiles.",
        "- Count thresholds use explicit enemy-team counts.",
        "- WR, effects, and gaps are focus-side win-rate percentage points.",
        "- Selected-enchanter probe uses Sona, Karma, Lulu, and Zilean in `UTILITY` with `utility_enchanter` or `utility_protection`.",
        "- Low own-damage probe is anchored once per team side, then compared against the enemy heal/shield context.",
        "",
        "| Axis | Low threshold | High threshold | Notes |",
        "|---|---|---|---|",
        "| Physical share | `<= 0.387` | `>= 0.557` | Team-average identity-context physical share. |",
        "| Magic share | `<= 0.373` | `>= 0.549` | Team-average identity-context magic share. |",
        "| Damage pressure | `<= 0.739` | `>= 0.813` | Team-average champion damage pressure. |",
        "| Damage-taken pressure | `<= 0.639` | `>= 0.721` | Team-average damage-taken pressure. |",
        "| Heal/shield pressure | `<= 0.028` | `>= 0.202` | Team-average ally heal/shield pressure. |",
        "| CC pressure | `<= 0.374` | `>= 0.539` | Team-average crowd-control pressure. |",
        "| Siege pressure | `<= 0.441` | `>= 0.530` | Team-average siege and structure pressure. |",
        "| Scaling pressure | `<= 0.829` | `>= 0.863` | Team-average scaling pressure. |",
        "| Burst-proxy count | `0` | `>= 3` | Enemy slots with slot damage pressure `>= 0.952` and a non-tank build. |",
        "| Hard-CC count | `0` | `>= 3` | Enemy slots with slot CC pressure `>= 0.696`. |",
        "| Tank/frontline count | `0` | `>= 3` | Enemy builds in `ar_tank`, `mr_tank`, `ad_off_tank`, or `ap_off_tank`. |",
        "| Heavy damage-taken count | `0` | `>= 3` | Enemy slots with slot damage-taken pressure `>= 0.822`. |",
        "| High-HP count | `0` | `>= 3` | Enemy champions with static level-18 HP `>= 2478.5`. |",
        "| Focus HP tier | `<= 2309.0` | `>= 2478.5` | Static champion level-18 HP. |",
        "| Ranged count | `<= 1` | `>= 4` | Static `attackRange_flat > 250` as ranged. |",
        "| Same-role range | `<= 250` | `> 250` | Static attack range for the lane opponent. |",
        "| Skirmish-ally count | `0` | `>= 2` | Gwen, Jax, Irelia, Fiora, Udyr, and XinZhao on the focus team. |",
        "",
        "## Gap Summary",
        "",
        "| Section | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for section, section_rows in by_section.items():
        summary = gap_summary([bin_row for row in section_rows for bin_row in row.bins])
        lines.append(
            "| "
            + " | ".join(
                [
                    section,
                    str(len(section_rows)),
                    str(summary["n_populated_bins"]),
                    _format_pp(summary["mean_abs_gap"], signed=False),
                    _format_pp(summary["max_abs_gap"], signed=False),
                    _format_pp_mse(summary["gap_mse"]),
                ]
            )
            + " |"
        )
    for section, section_rows in by_section.items():
        lines.extend(["", f"## {section}", ""])
        lines.append("| Audit | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Bin 5 | Empirical effect | HGNN effect | Read |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in section_rows:
            bin_cells = [_format_bin_cell(bin_row) for bin_row in row.bins]
            while len(bin_cells) < 5:
                bin_cells.append("N/A")
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.spec.title,
                        *bin_cells[:5],
                        _format_pp(row.endpoint_effect),
                        _format_pp(row.hgnn_endpoint_effect),
                        row.spec.read,
                    ]
                )
                + " |"
            )
    all_bins = [bin_row for row in rows for bin_row in row.bins]
    summary = gap_summary(all_bins)
    lines.extend(
        [
            "",
            "## Overall Summary",
            "",
            "| Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE |",
            "|---:|---:|---:|---:|---:|",
            "| "
            + " | ".join(
                [
                    str(len(rows)),
                    str(summary["n_populated_bins"]),
                    _format_pp(summary["mean_abs_gap"], signed=False),
                    _format_pp(summary["max_abs_gap"], signed=False),
                    _format_pp_mse(summary["gap_mse"]),
                ]
            )
            + " |",
            "",
            "Gap MSE is `mean((HGNN_focus_WR - empirical_focus_WR)^2)` across populated "
            "threshold bins, rendered as percentage-points squared.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def gap_summary(rows: Sequence[AuditBin]) -> dict[str, float | int]:
    gaps = np.asarray([row.gap for row in rows if row.n > 0 and np.isfinite(row.gap)], dtype=np.float64)
    return {
        "n_populated_bins": int(sum(row.n > 0 for row in rows)),
        "mean_abs_gap": _mean_or_nan(np.abs(gaps)),
        "max_abs_gap": _max_or_nan(np.abs(gaps)),
        "gap_mse": _mean_or_nan(gaps**2),
    }


def _static_lookups() -> tuple[np.ndarray, np.ndarray]:
    by_id = load_static_by_id()
    max_id = max(by_id) if by_id else 0
    hp = np.zeros(max_id + 1, dtype=np.float32)
    attack_range = np.zeros(max_id + 1, dtype=np.float32)
    for champion_id, values in by_id.items():
        if champion_id >= hp.size:
            continue
        # `static_feature_names()` is source order plus level-18 derived stats;
        # source order has health_flat at index 24, health_perLevel at 25, and
        # attackRange_flat at index 10 in the checked-in champion stat records.
        attack_range[champion_id] = float(values[10])
        hp[champion_id] = float(values[24] + 17.0 * values[25])
    return hp, attack_range


def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _max_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def _format_pct(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{100.0 * value:.2f}%"


def _format_pp(value: float, *, signed: bool = True) -> str:
    if not math.isfinite(value):
        return "N/A"
    sign = "+" if signed and value >= 0.0 else ""
    return f"{sign}{100.0 * value:.2f} pp"


def _format_pp_mse(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{10000.0 * value:.2f} pp^2"


def _format_bin_cell(row: AuditBin) -> str:
    if row.n <= 0:
        return f"`{row.label}`<br/>n=0<br/>emp=N/A<br/>HGNN=N/A<br/>gap=N/A"
    return (
        f"`{row.label}`<br/>"
        f"n={row.n:,}<br/>"
        f"emp={_format_pct(row.empirical_wr)}<br/>"
        f"HGNN={_format_pct(row.hgnn_wr)}<br/>"
        f"gap={_format_pp(row.gap)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-cache-dir", type=Path, default=DEFAULT_CONTEXT_CACHE_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prediction-cache", type=Path, default=DEFAULT_PREDICTION_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--refresh-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    meta = json.loads((args.context_cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
    n_games = int(meta["n_games"])
    probabilities = load_or_predict_blue_probabilities(
        model_path=args.model_path,
        model_cache_dir=args.model_cache_dir,
        prediction_cache=args.prediction_cache,
        n_games=n_games,
        refresh=bool(args.refresh_predictions),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    data = AuditData(context_cache_dir=args.context_cache_dir, blue_probability=probabilities)
    rows = evaluate_specs(data, audit_specs())
    markdown = render_audit(
        rows,
        model_path=args.model_path,
        model_cache_dir=args.model_cache_dir,
        context_cache_dir=args.context_cache_dir,
    )
    write_audit(args.output, markdown)


if __name__ == "__main__":
    main()
