"""Gymnasium environment for tournament-style LoL champion drafting.

Internal action space is `Discrete(len(champion_ids))` — i.e. positional
indices into the `champion_ids` tuple. Real champion IDs (which are sparse
in the DB) are only resolved at the predictor boundary.

Transition state is a single `DraftState` (`app.rl.draft`); the env is a thin
gym wrapper that adds the gym spaces, `random_start_steps`, and per-side reward
selection, resolving action indices to real champion ids only at the predictor
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.rl.draft import DRAFT_SEQUENCE, DraftState, DraftStep, Side
from app.rl.reward import (
    Predictor,
    RewardMode,
    RoleBuildOptimizer,
    RoleBuildSampler,
    resolve_rewards,
)


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

        self._state = DraftState.initial(n)

    def current_step(self) -> DraftStep | None:
        return self._state.current_step()

    def get_action_mask(self) -> np.ndarray:
        return self._state.legal_mask()

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
        self._state = DraftState.initial(self.cfg.n_champions)

        # Pre-fill K steps with random legal actions so episodes start
        # from diverse mid-draft states. Uses the env's own np_random.
        for _ in range(self.cfg.random_start_steps):
            legal = np.flatnonzero(self._state.available)
            self._state.apply(int(self.np_random.choice(legal)))
        return self._state.to_obs(), self._info()

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._state.is_terminal():
            raise RuntimeError(
                "Episode already terminated; call reset() before stepping."
            )
        # DraftState.apply validates legality (raises on a masked action).
        just_acted_side = self._state.apply(int(action))
        terminated = self._state.is_terminal()
        reward = 0.0
        info = self._info()

        if terminated:
            blue_team = [int(self._champ_ids[i]) for i in self._state.blue_picks]
            red_team = [int(self._champ_ids[i]) for i in self._state.red_picks]
            if self._optimizer is not None:
                result = self._optimizer(
                    blue_team, red_team, self._predictor, self.cfg.reward_mode
                )
            else:
                assert self._sampler is not None  # checked in __init__
                result = resolve_rewards(
                    blue_team,
                    red_team,
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

        return self._state.to_obs(), float(reward), terminated, False, info
