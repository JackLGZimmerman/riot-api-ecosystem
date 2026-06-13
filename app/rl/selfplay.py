"""Self-play episode generation for the AlphaZero learner.

Each episode plays one full draft using PUCT MCTS at every step. The
sample stored per step is ``(features, search_policy, mask,
acting_side)``. After the terminal step, the per-side reward from the
predictor is broadcast back as the value target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from app.rl.draft import DraftState, Side
from app.rl.mcts import MCTS, MCTSConfig, visit_policy
from app.rl.net import AlphaZeroNet, encode_obs
from app.rl.reward import (
    Predictor,
    RewardMode,
    RoleBuildOptimizer,
    RoleBuildSampler,
)


@dataclass
class EpisodeSamples:
    features: np.ndarray  # [T, obs_dim] float32
    policy_targets: np.ndarray  # [T, n_champions] float32
    masks: np.ndarray  # [T, n_champions] bool
    value_targets: np.ndarray  # [T] float32
    sides: np.ndarray  # [T] int8
    blue_reward: float
    red_reward: float
    info: dict[str, Any]


def play_episode(
    net: AlphaZeroNet,
    predictor: Predictor,
    *,
    n_champions: int,
    champion_ids: tuple[int, ...] | None,
    mcts_cfg: MCTSConfig,
    device: torch.device,
    reward_mode: RewardMode = "expected_value",
    risk_lambda: float = 0.5,
    sampler: RoleBuildSampler | None = None,
    optimizer: RoleBuildOptimizer | None = None,
    rng: np.random.Generator | None = None,
) -> EpisodeSamples:
    rng = rng or np.random.default_rng()
    mcts = MCTS(
        net,
        predictor,
        mcts_cfg,
        device,
        reward_mode=reward_mode,
        risk_lambda=risk_lambda,
        sampler=sampler,
        optimizer=optimizer,
        champion_ids=champion_ids,
        rng=rng,
    )

    state = DraftState.initial(n_champions)
    feats: list[np.ndarray] = []
    pols: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    sides: list[int] = []

    while not state.is_terminal():
        _, visits = mcts.run(state)
        legal = state.legal_mask()
        temperature = (
            mcts_cfg.temperature
            if state.step_idx < mcts_cfg.temperature_drop_step
            else 0.0
        )
        policy_target = visit_policy(visits, legal, temperature)
        # Sample from policy target (handles greedy = one-hot case naturally).
        action = int(rng.choice(n_champions, p=policy_target))
        feats.append(encode_obs(state.to_obs(), n_champions))
        pols.append(policy_target)
        masks.append(legal)
        side = state.current_side()
        sides.append(int(side) if side is not None else -1)
        state.apply(action)

    rewards = mcts.terminal_rewards(state)  # cached if already computed during search
    side_arr = np.asarray(sides, dtype=np.int8)
    value_targets = np.where(
        side_arr == int(Side.BLUE), rewards[Side.BLUE], rewards[Side.RED]
    ).astype(np.float32)
    return EpisodeSamples(
        features=np.stack(feats).astype(np.float32),
        policy_targets=np.stack(pols).astype(np.float32),
        masks=np.stack(masks).astype(bool),
        value_targets=value_targets,
        sides=side_arr,
        blue_reward=float(rewards[Side.BLUE]),
        red_reward=float(rewards[Side.RED]),
        info={"blue_picks": list(state.blue_picks), "red_picks": list(state.red_picks)},
    )
