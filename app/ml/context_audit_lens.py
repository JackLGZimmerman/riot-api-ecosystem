from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.ml.context_audit_specs import AuditSpec, POSITIONS
from app.ml.semantic_group_features import (
    BURST_DAMAGE_THRESHOLD,
    CONTEXT_AXIS_INDEX,
    FOCUS_HP_LOW_THRESHOLD,
    HARD_CC_THRESHOLD,
    HEAVY_TAKEN_THRESHOLD,
    HIGH_HP_THRESHOLD,
    LOW_OWN_DAMAGE_THRESHOLD,
    RANGED_ATTACK_RANGE_THRESHOLD,
    SELECTED_ENCHANTER_BUILDS,
    SELECTED_ENCHANTERS,
    SKIRMISH_CHAMPIONS,
    TANK_BUILD_LABELS,
    static_hp_range_lookups,
)


class AuditLens:
    """Hard audit-bin lens shared by reports and train-time checkpoint metrics."""

    def __init__(
        self,
        *,
        champion_id: np.ndarray,
        build_id: np.ndarray,
        context_raw: np.ndarray,
        build_vocab: Sequence[str],
        hp_lookup: np.ndarray | None = None,
        range_lookup: np.ndarray | None = None,
    ) -> None:
        self.champion_id = np.asarray(champion_id)
        self.build_id = np.asarray(build_id)
        self.context_raw = np.asarray(context_raw)
        if self.champion_id.shape != self.build_id.shape:
            raise ValueError("champion_id and build_id must have matching shapes")
        if self.champion_id.ndim != 2 or self.champion_id.shape[1] != 10:
            raise ValueError("champion_id/build_id must have shape [games, 10]")
        if self.context_raw.ndim != 3 or self.context_raw.shape[:2] != self.champion_id.shape:
            raise ValueError("context_raw must have shape [games, 10, context_dim]")
        required_context_dim = max(CONTEXT_AXIS_INDEX.values()) + 1
        if self.context_raw.shape[2] < required_context_dim:
            raise ValueError(
                f"context_raw must have at least {required_context_dim} axes"
            )
        self.build_vocab = tuple(str(label) for label in build_vocab)
        self.build_to_idx = {label: idx for idx, label in enumerate(self.build_vocab)}
        self._hp_lookup, self._range_lookup = (
            static_hp_range_lookups()
            if hp_lookup is None or range_lookup is None
            else (hp_lookup, range_lookup)
        )
        self._slot_hp_cache: np.ndarray | None = None
        self._slot_range_cache: np.ndarray | None = None
        self._axis_cache: dict[str, np.ndarray] = {}

    @property
    def n_games(self) -> int:
        return int(self.champion_id.shape[0])

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

    def build_ids(self, labels: Sequence[str]) -> list[int]:
        return [self.build_to_idx[label] for label in labels if label in self.build_to_idx]

    def axis(self, name: str) -> np.ndarray:
        if name not in self._axis_cache:
            self._axis_cache[name] = self._build_axis(name)
        return self._axis_cache[name]

    def _build_axis(self, name: str) -> np.ndarray:
        if name.startswith("enemy_") and name.removeprefix("enemy_") in CONTEXT_AXIS_INDEX:
            return self._team_context(
                CONTEXT_AXIS_INDEX[name.removeprefix("enemy_")],
                enemy=True,
            )
        if name.startswith("ally_") and name.removeprefix("ally_") in CONTEXT_AXIS_INDEX:
            return self._team_context(
                CONTEXT_AXIS_INDEX[name.removeprefix("ally_")],
                enemy=False,
            )
        if name == "enemy_burst_count":
            non_tank = ~np.isin(self.build_id, self.build_ids(TANK_BUILD_LABELS))
            burst_slot = (
                self.context_raw[:, :, CONTEXT_AXIS_INDEX["damage"]]
                >= BURST_DAMAGE_THRESHOLD
            ) & non_tank
            return self._enemy_count(burst_slot)
        if name == "enemy_hard_cc_count":
            return self._enemy_count(
                self.context_raw[:, :, CONTEXT_AXIS_INDEX["cc"]] >= HARD_CC_THRESHOLD
            )
        if name == "enemy_frontline_count":
            return self._enemy_count(np.isin(self.build_id, self.build_ids(TANK_BUILD_LABELS)))
        if name == "enemy_heavy_taken_count":
            return self._enemy_count(
                self.context_raw[:, :, CONTEXT_AXIS_INDEX["damage_taken"]]
                >= HEAVY_TAKEN_THRESHOLD
            )
        if name == "enemy_high_hp_count":
            return self._enemy_count(self.slot_hp >= HIGH_HP_THRESHOLD)
        if name == "enemy_ranged_count":
            return self._enemy_count(self.slot_range > RANGED_ATTACK_RANGE_THRESHOLD)
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
        mask = np.ones(self.champion_id.shape, dtype=bool)
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
            mask &= np.isin(self.build_id, self.build_ids(spec.builds))
        if spec.focus_condition == "low_own_damage":
            side_anchor = np.zeros(10, dtype=bool)
            side_anchor[[0, 5]] = True
            mask &= side_anchor[None, :]
            mask &= self.axis("ally_damage") <= LOW_OWN_DAMAGE_THRESHOLD
        elif spec.focus_condition == "focus_hp_low":
            mask &= self.slot_hp <= FOCUS_HP_LOW_THRESHOLD
        elif spec.focus_condition == "focus_hp_high":
            mask &= self.slot_hp >= HIGH_HP_THRESHOLD
        elif spec.focus_condition == "selected_enchanter":
            mask &= np.isin(self.champion_id, list(SELECTED_ENCHANTERS))
            mask &= np.isin(self.build_id, self.build_ids(SELECTED_ENCHANTER_BUILDS))
        elif spec.focus_condition is not None:
            raise ValueError(f"unknown focus condition: {spec.focus_condition}")
        return mask


__all__ = ["AuditLens"]
