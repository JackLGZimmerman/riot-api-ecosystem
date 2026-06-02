"""Behavioral probes for the HGNN context-atlas head.

These are post-training sanity / regression checks against the trained model. For
each documented example in `documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md` we feed
the model synthetic drafts that sweep one enemy/ally context axis and measure the
context head's marginal contribution (the learned contextual residual) and the
implied win-probability movement, then compare the *direction* to the audit.

Banks/prototypes are data-derived: each axis prototype is the support-weighted
mean of the real identities that score highest on that interpretable axis (not a
hand-authored specialist label). Nothing here feeds training, feature
engineering, or thresholds; the thresholds below are only for pass/weak/fail
*reporting*.

Run with:
    python -m app.ml.context_probes
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from app.classification.embeddings.config import (
    CONTEXT_CC_INDEX,
    CONTEXT_DAMAGE_PRESSURE_INDEX,
    CONTEXT_HEAL_SHIELD_INDEX,
    CONTEXT_TAKEN_INDEX,
)
from app.classification.embeddings.runtime import IdentityContextLookup
from app.core.logging.logger import setup_logging_config
from app.core.utils.common import POSITIONS
from app.ml.config import TrainConfig
from app.ml.hgnn_model import HGNNWinModel, load_hgnn_model, resolve_device

logger = logging.getLogger(__name__)

PHYS_INDEX = 0
MAGIC_INDEX = 1
MEANINGFUL = 0.02  # |Δlogit| above which a directionally-correct probe is "pass"
DEFAULT_SUPPORT = 300.0


@dataclass
class ProbeResult:
    name: str
    axis: str
    expected: str  # "increase" | "decrease"
    low_label: str
    high_label: str
    low_marginal_logit: float
    high_marginal_logit: float
    delta_logit: float
    delta_winprob: float
    verdict: str


class ContextBank:
    """Data-derived axis prototypes from the identity_context descriptor."""

    def __init__(self, lookup: IdentityContextLookup, *, min_support: float = 200.0) -> None:
        self.lookup = lookup
        keys = list(lookup.values)
        self.keys = keys
        self.vectors = np.stack([lookup.values[k] for k in keys], axis=0).astype(np.float64)
        self.support = np.array([lookup.support[k] for k in keys], dtype=np.float64)
        self.dim = lookup.dim
        self.mask = self.support >= min_support
        # Support-weighted global mean = the "neutral / average" identity.
        w = np.where(self.mask, self.support, 0.0)
        self.neutral = (self.vectors * w[:, None]).sum(0) / max(w.sum(), 1e-9)

    def prototype(self, axis_index: int, *, weight_index: int | None = None, top_frac: float = 0.10) -> np.ndarray:
        """Support-weighted mean of the top identities on `axis_index`.

        `weight_index` (e.g. damage pressure) up-weights identities that actually
        express the axis with volume, so e.g. the "physical" prototype is a real
        physical-damage carry rather than a low-volume oddity.
        """
        score = self.vectors[:, axis_index].copy()
        if weight_index is not None:
            score = score * self.vectors[:, weight_index]
        score = np.where(self.mask, score, -np.inf)
        k = max(int(self.mask.sum() * top_frac), 20)
        top = np.argsort(score)[-k:]
        w = self.support[top]
        proto = (self.vectors[top] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
        return proto.astype(np.float64)


def _winprob_delta(low_logit: float, high_logit: float) -> float:
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    return float(sig(high_logit) - sig(low_logit))


class ContextProber:
    def __init__(self, model: HGNNWinModel, lookup: IdentityContextLookup) -> None:
        self.model = model.eval()
        self.lookup = lookup
        self.bank = ContextBank(lookup)
        self.dim = model.config.identity_context_dim

    def _marginal(
        self,
        *,
        focus_vec: np.ndarray,
        focus_support: float,
        focus_role_idx: int,
        enemy_vec: np.ndarray,
        ally_fill_vec: np.ndarray,
    ) -> float:
        """Team-level context logit added by placing `focus` (vs a neutral filler)
        on blue, against an enemy team of `enemy_vec` and neutral allies."""

        def ctx_logit(blue_focus: np.ndarray, focus_sup: float) -> float:
            ctx = np.zeros((1, 10, self.dim), dtype=np.float32)
            sup = np.full((1, 10), DEFAULT_SUPPORT, dtype=np.float32)
            for i in range(5):
                ctx[0, i] = ally_fill_vec
                ctx[0, 5 + i] = enemy_vec
            ctx[0, focus_role_idx] = blue_focus
            sup[0, focus_role_idx] = focus_sup
            with torch.no_grad():
                out = self.model._context_logit(
                    torch.from_numpy(ctx), torch.from_numpy(sup)
                )
            return float(out.item())

        return ctx_logit(focus_vec.astype(np.float32), focus_support) - ctx_logit(
            ally_fill_vec.astype(np.float32), DEFAULT_SUPPORT
        )

    def sweep_enemy(
        self,
        *,
        name: str,
        axis: str,
        expected: str,
        focus_key: tuple[int, str, str],
        low_proto: np.ndarray,
        high_proto: np.ndarray,
        readout_index: int,
        ally_fill_vec: np.ndarray | None = None,
    ) -> ProbeResult | None:
        focus_vec = self.lookup.values.get(focus_key)
        if focus_vec is None:
            logger.warning("probe %s: focus identity %s missing from lookup", name, focus_key)
            return None
        focus_support = self.lookup.support.get(focus_key, DEFAULT_SUPPORT)
        focus_role_idx = POSITIONS.index(focus_key[1])
        fill = self.bank.neutral if ally_fill_vec is None else ally_fill_vec

        def label(vec: np.ndarray) -> str:
            return f"{vec[readout_index]:.2f}"

        low = self._marginal(
            focus_vec=focus_vec,
            focus_support=focus_support,
            focus_role_idx=focus_role_idx,
            enemy_vec=low_proto,
            ally_fill_vec=fill,
        )
        high = self._marginal(
            focus_vec=focus_vec,
            focus_support=focus_support,
            focus_role_idx=focus_role_idx,
            enemy_vec=high_proto,
            ally_fill_vec=fill,
        )
        delta = high - low
        want_up = expected == "increase"
        ok_dir = (delta > 0) == want_up
        verdict = "pass" if (ok_dir and abs(delta) >= MEANINGFUL) else ("weak" if ok_dir else "fail")
        return ProbeResult(
            name=name,
            axis=axis,
            expected=expected,
            low_label=label(low_proto),
            high_label=label(high_proto),
            low_marginal_logit=round(low, 4),
            high_marginal_logit=round(high, 4),
            delta_logit=round(delta, 4),
            delta_winprob=round(_winprob_delta(low, high), 4),
            verdict=verdict,
        )

    def sweep_ally(
        self,
        *,
        name: str,
        axis: str,
        expected: str,
        focus_key: tuple[int, str, str],
        low_ally: np.ndarray,
        high_ally: np.ndarray,
        readout_index: int,
    ) -> ProbeResult | None:
        """Like sweep_enemy but varies the ALLY fill (2vX synergy), enemy neutral."""
        focus_vec = self.lookup.values.get(focus_key)
        if focus_vec is None:
            logger.warning("probe %s: focus identity %s missing from lookup", name, focus_key)
            return None
        focus_support = self.lookup.support.get(focus_key, DEFAULT_SUPPORT)
        focus_role_idx = POSITIONS.index(focus_key[1])
        enemy = self.bank.neutral

        low = self._marginal(
            focus_vec=focus_vec, focus_support=focus_support, focus_role_idx=focus_role_idx,
            enemy_vec=enemy, ally_fill_vec=low_ally,
        )
        high = self._marginal(
            focus_vec=focus_vec, focus_support=focus_support, focus_role_idx=focus_role_idx,
            enemy_vec=enemy, ally_fill_vec=high_ally,
        )
        delta = high - low
        want_up = expected == "increase"
        ok_dir = (delta > 0) == want_up
        verdict = "pass" if (ok_dir and abs(delta) >= MEANINGFUL) else ("weak" if ok_dir else "fail")
        return ProbeResult(
            name=name, axis=axis, expected=expected,
            low_label=f"{low_ally[readout_index]:.2f}", high_label=f"{high_ally[readout_index]:.2f}",
            low_marginal_logit=round(low, 4), high_marginal_logit=round(high, 4),
            delta_logit=round(delta, 4), delta_winprob=round(_winprob_delta(low, high), 4),
            verdict=verdict,
        )


def run_probes(model: HGNNWinModel, lookup: IdentityContextLookup) -> list[ProbeResult]:
    p = ContextProber(model, lookup)
    b = p.bank
    phys = b.prototype(PHYS_INDEX, weight_index=CONTEXT_DAMAGE_PRESSURE_INDEX)
    magic = b.prototype(MAGIC_INDEX, weight_index=CONTEXT_DAMAGE_PRESSURE_INDEX)
    heal = b.prototype(CONTEXT_HEAL_SHIELD_INDEX)
    low_heal = b.neutral.copy()
    low_heal[CONTEXT_HEAL_SHIELD_INDEX] = 0.0
    hi_damage = b.prototype(CONTEXT_DAMAGE_PRESSURE_INDEX, weight_index=CONTEXT_DAMAGE_PRESSURE_INDEX)
    lo_damage = b.neutral.copy()
    lo_damage[CONTEXT_DAMAGE_PRESSURE_INDEX] *= 0.5
    carry = b.prototype(CONTEXT_DAMAGE_PRESSURE_INDEX, weight_index=PHYS_INDEX)

    results: list[ProbeResult] = []
    specs = [
        # Malphite ar_tank gains as enemy physical share rises (kept as ONE case).
        dict(name="malphite_artank_vs_enemy_physical", axis="enemy phys_offense_share",
             expected="increase", focus_key=(54, "TOP", "ar_tank"),
             low_proto=magic, high_proto=phys, readout_index=PHYS_INDEX),
        # Dr. Mundo mr_tank gains as enemy magic share rises.
        dict(name="drmundo_mrtank_vs_enemy_magic", axis="enemy magic_offense_share",
             expected="increase", focus_key=(36, "TOP", "mr_tank"),
             low_proto=phys, high_proto=magic, readout_index=MAGIC_INDEX),
        # Galio mr_tank, the cleanest mr-tank-into-magic example in the audit.
        dict(name="galio_mrtank_vs_enemy_magic", axis="enemy magic_offense_share",
             expected="increase", focus_key=(3, "MIDDLE", "mr_tank"),
             low_proto=phys, high_proto=magic, readout_index=MAGIC_INDEX),
        # Low-damage identity into rising enemy heal/shield should weaken.
        dict(name="low_damage_vs_enemy_heal_shield", axis="enemy heal_shield_pressure",
             expected="decrease", focus_key=(117, "UTILITY", "utility_enchanter"),
             low_proto=low_heal, high_proto=heal, readout_index=CONTEXT_HEAL_SHIELD_INDEX),
        # Sion TOP ad_off_tank performs better into LOW enemy damage than HIGH, so
        # rising enemy damage output should weaken the durability identity's edge.
        dict(name="tank_vs_enemy_damage_output", axis="enemy champion_damage_pressure",
             expected="decrease", focus_key=(14, "TOP", "ad_off_tank"),
             low_proto=lo_damage, high_proto=hi_damage, readout_index=CONTEXT_DAMAGE_PRESSURE_INDEX),
    ]
    for spec in specs:
        r = p.sweep_enemy(**spec)
        if r is not None:
            results.append(r)

    # Enchanter with skirmish/carry allies (ally 2vX synergy) should gain.
    r = p.sweep_ally(
        name="enchanter_with_carry_allies", axis="ally champion_damage_pressure",
        expected="increase", focus_key=(117, "UTILITY", "utility_enchanter"),
        low_ally=lo_damage, high_ally=carry, readout_index=CONTEXT_DAMAGE_PRESSURE_INDEX,
    )
    if r is not None:
        results.append(r)
    return results


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.INFO)
    device = resolve_device("auto")
    model, config, _ = load_hgnn_model(TrainConfig().model_path, device="cpu")
    if not getattr(model, "context_enabled", False):
        raise SystemExit("Loaded model has no context head (identity_context_dim=0).")
    lookup = IdentityContextLookup.load()
    results = run_probes(model, lookup)

    header = f"{'probe':38s} {'expected':9s} {'low->high':>14s} {'Δlogit':>8s} {'Δwinp':>7s}  verdict"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:38s} {r.expected:9s} "
            f"{r.low_label+'->'+r.high_label:>14s} {r.delta_logit:>8.4f} {r.delta_winprob:>7.4f}  {r.verdict}"
        )
    out = Path("app/ml/data/context_probes.json")
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
