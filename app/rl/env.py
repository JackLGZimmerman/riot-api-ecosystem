"""Gymnasium environment for tournament-style LoL champion drafting.

Internal action space is `Discrete(len(champion_ids))` — i.e. positional
indices into the `champion_ids` tuple. Real champion IDs (which are sparse
in the DB) are only resolved at the predictor boundary.

Internal state is a single `int8` ownership vector of shape (n_champions,)
with codes {AVAILABLE, BLUE_BAN, RED_BAN, BLUE_PICK, RED_PICK}. The dict
observation shape is unchanged — the four pick/ban index arrays and the
available mask are derived from the ownership vector on each `_obs` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.rl.draft import DRAFT_SEQUENCE, ActionType, DraftStep, Side
from app.rl.reward import (
    OptimizationResult,
    Predictor,
    RewardMode,
    RoleBuildOptimizer,
    RoleBuildSampler,
    resolve_rewards,
)

# Ownership codes packed into a single int8 vector.
_AVAILABLE: int = 0
_BLUE_BAN: int = 1
_RED_BAN: int = 2
_BLUE_PICK: int = 3
_RED_PICK: int = 4


@dataclass(kw_only=True)
class DraftEnvConfig:
    top_k_build_configs: int  # required; cap on configs per team per terminal
    champion_ids: tuple[int, ...] = field(default_factory=tuple)
    agent_side: Literal["blue", "red", "self_play"] = "self_play"
    reward_mode: RewardMode = "expected_value"
    risk_lambda: float = 0.5
    random_start_steps: int = 0

    @property
    def n_champions(self) -> int:
        return len(self.champion_ids)


class DraftEnv(gym.Env):
    """LoL tournament-draft RL environment (index-space action, real-id reward)."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        predictor: Predictor,
        config: DraftEnvConfig,
        *,
        sampler: RoleBuildSampler | None = None,
        optimizer: RoleBuildOptimizer | None = None,
    ) -> None:
        super().__init__()
        n = config.n_champions
        if n < 20:
            raise ValueError(
                "champion_ids must have at least 20 entries (10 bans + 10 picks)."
            )
        if not (0 <= config.random_start_steps < len(DRAFT_SEQUENCE)):
            raise ValueError("random_start_steps must be in [0, len(DRAFT_SEQUENCE)).")
        if sampler is None and optimizer is None:
            raise ValueError(
                "DraftEnv requires either a sampler or an optimizer; "
                "the previous pick-order default was unsafe and has been removed."
            )
        self.cfg = config
        self._champ_ids = np.asarray(config.champion_ids, dtype=np.int64)
        self._predictor = predictor
        self._sampler = sampler
        self._optimizer = optimizer

        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict(
            {
                "blue_picks": spaces.Box(
                    low=-1, high=n - 1, shape=(5,), dtype=np.int32
                ),
                "red_picks": spaces.Box(low=-1, high=n - 1, shape=(5,), dtype=np.int32),
                "blue_bans": spaces.Box(low=-1, high=n - 1, shape=(5,), dtype=np.int32),
                "red_bans": spaces.Box(low=-1, high=n - 1, shape=(5,), dtype=np.int32),
                "available_mask": spaces.MultiBinary(n),
                "step": spaces.Discrete(len(DRAFT_SEQUENCE) + 1),
                "acting_side": spaces.Discrete(2),
                "action_type": spaces.Discrete(2),
            }
        )

        self._ownership = np.zeros(n, dtype=np.int8)
        self._blue_team: list[int] = []
        self._red_team: list[int] = []
        self._step_idx = 0

    def current_step(self) -> DraftStep | None:
        if self._step_idx >= len(DRAFT_SEQUENCE):
            return None
        return DRAFT_SEQUENCE[self._step_idx]

    def get_action_mask(self) -> np.ndarray:
        return self._ownership == _AVAILABLE

    def _pad_indices(self, code: int) -> np.ndarray:
        out = np.full(5, -1, dtype=np.int32)
        idx = np.flatnonzero(self._ownership == code)
        out[: idx.size] = idx
        return out

    def _obs(self) -> dict[str, Any]:
        step = self.current_step()
        acting_side = 0 if step is None else int(step.side)
        action_type = 0 if step is None else int(step.action_type)
        return {
            "blue_picks": self._pad_indices(_BLUE_PICK),
            "red_picks": self._pad_indices(_RED_PICK),
            "blue_bans": self._pad_indices(_BLUE_BAN),
            "red_bans": self._pad_indices(_RED_BAN),
            "available_mask": self.get_action_mask(),
            "step": int(self._step_idx),
            "acting_side": acting_side,
            "action_type": action_type,
        }

    def _info(self) -> dict[str, Any]:
        return {
            "draft_step": self.current_step(),
            "action_mask": self.get_action_mask(),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._ownership.fill(_AVAILABLE)
        self._blue_team.clear()
        self._red_team.clear()
        self._step_idx = 0

        # Pre-fill K steps with random legal actions so episodes start
        # from diverse mid-draft states. Uses the env's own np_random.
        for _ in range(self.cfg.random_start_steps):
            legal = np.flatnonzero(self._ownership == _AVAILABLE)
            action = int(self.np_random.choice(legal))
            self._apply(action)
        return self._obs(), self._info()

    def _apply(self, action: int) -> Side:
        """Commit an already-validated action; return the side that acted."""
        step = DRAFT_SEQUENCE[self._step_idx]
        if step.action_type == ActionType.BAN:
            code = _BLUE_BAN if step.side == Side.BLUE else _RED_BAN
        else:
            code = _BLUE_PICK if step.side == Side.BLUE else _RED_PICK
            team = self._blue_team if step.side == Side.BLUE else self._red_team
            team.append(int(self._champ_ids[action]))
        self._ownership[action] = code
        self._step_idx += 1
        return step.side

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = int(action)
        if self._step_idx >= len(DRAFT_SEQUENCE):
            raise RuntimeError(
                "Episode already terminated; call reset() before stepping."
            )
        assert self._ownership[action] == _AVAILABLE, (
            f"Illegal action {action}; policy must mask via get_action_mask()."
        )

        just_acted_side = self._apply(action)
        terminated = self._step_idx >= len(DRAFT_SEQUENCE)
        reward = 0.0
        info = self._info()

        if terminated:
            if self._optimizer is not None:
                result: OptimizationResult = self._optimizer(
                    self._blue_team,
                    self._red_team,
                    self._predictor,
                    self.cfg.reward_mode,
                )
            else:
                assert self._sampler is not None  # checked in __init__
                result = resolve_rewards(
                    self._blue_team,
                    self._red_team,
                    self._predictor,
                    self._sampler,
                    self.cfg.reward_mode,
                    self.cfg.risk_lambda,
                )

            info.update(
                {
                    "blue_reward": result.blue_reward,
                    "red_reward": result.red_reward,
                    "p_blue_win_for_blue": result.p_blue_win_for_blue,
                    "p_blue_win_for_red": result.p_blue_win_for_red,
                    "win_matrix": result.win_matrix,
                    "blue_configs": result.blue_configs,
                    "red_configs": result.red_configs,
                }
            )

            if self.cfg.agent_side == "blue":
                reward = result.blue_reward
            elif self.cfg.agent_side == "red":
                reward = result.red_reward
            else:
                reward = (
                    result.blue_reward
                    if just_acted_side == Side.BLUE
                    else result.red_reward
                )

        return self._obs(), float(reward), terminated, False, info
