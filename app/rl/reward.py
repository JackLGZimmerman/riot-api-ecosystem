"""Predictor protocol, role/build sampling, and terminal-state reward resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import numpy as np

from app.ml.config import POSITIONS

RewardMode = Literal["expected_value", "risk_adjusted", "worst_case"]


@dataclass
class RoleBuildConfig:
    """One plausible hidden assignment for a team."""

    roles: dict[int, str]  # champion_id -> role
    builds: dict[int, int]  # champion_id -> build_id
    probability: float = 1.0  # optional prior weight


class Predictor(Protocol):
    """Win-probability model: returns P(blue wins)."""

    def __call__(
        self,
        blue_team: list[int],
        red_team: list[int],
        blue_roles: dict[int, str],
        red_roles: dict[int, str],
        blue_builds: dict[int, int],
        red_builds: dict[int, int],
    ) -> float: ...


RoleBuildSampler = Callable[[list[int], str], list[RoleBuildConfig]]


@dataclass
class OptimizationResult:
    blue_reward: float
    red_reward: float
    p_blue_win_for_blue: float
    p_blue_win_for_red: float
    win_matrix: np.ndarray  # shape (n_blue_cfg, n_red_cfg)
    blue_configs: list[RoleBuildConfig]
    red_configs: list[RoleBuildConfig]


class RoleBuildOptimizer(Protocol):
    """Custom hook that replaces the default sampler + reward-mode aggregation."""

    def __call__(
        self,
        blue_team: list[int],
        red_team: list[int],
        predictor: Predictor,
        reward_mode: RewardMode,
    ) -> OptimizationResult: ...


def default_role_build_sampler(
    team_champions: list[int],
    side: str,
) -> list[RoleBuildConfig]:
    """Single canonical role assignment, build_id=0 for every champion."""
    return [
        RoleBuildConfig(
            roles=dict(zip(team_champions, POSITIONS)),
            builds={c: 0 for c in team_champions},
            probability=1.0,
        )
    ]


def resolve_rewards(
    blue_team: list[int],
    red_team: list[int],
    predictor: Predictor,
    sampler: RoleBuildSampler,
    reward_mode: RewardMode,
    risk_lambda: float = 0.5,
) -> OptimizationResult:
    """Build the full P(blue|cfg_b, cfg_r) matrix and aggregate per mode.

    Each side gets a perspective-adjusted p_blue_win, so the env never
    picks the enemy configuration that would be best for the drafting side.

    - expected_value : symmetric mean over the joint config distribution.
    - risk_adjusted  : mean - lambda * std for blue; mean + lambda * std
                       for red (uncertainty penalised on both sides).
    - worst_case     : min for blue (worst-case blue), max for red
                       (worst-case red); each side assumes the enemy
                       commits to the worst plausible config for it.
    """
    blue_configs = sampler(blue_team, "blue")
    red_configs = sampler(red_team, "red")
    if not blue_configs or not red_configs:
        raise ValueError("Role/build sampler returned no configurations.")

    n_b, n_r = len(blue_configs), len(red_configs)
    win_matrix = np.zeros((n_b, n_r), dtype=np.float64)
    for i, bc in enumerate(blue_configs):
        for j, rc in enumerate(red_configs):
            win_matrix[i, j] = float(
                predictor(
                    blue_team,
                    red_team,
                    bc.roles,
                    rc.roles,
                    bc.builds,
                    rc.builds,
                )
            )

    if reward_mode == "expected_value":
        m = float(win_matrix.mean())
        p_for_blue, p_for_red = m, m
    elif reward_mode == "risk_adjusted":
        m = float(win_matrix.mean())
        s = float(win_matrix.std())
        p_for_blue = m - risk_lambda * s
        p_for_red = m + risk_lambda * s
    elif reward_mode == "worst_case":
        p_for_blue = float(win_matrix.min())
        p_for_red = float(win_matrix.max())
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode!r}")

    return OptimizationResult(
        blue_reward=p_for_blue - 0.5,
        red_reward=(1.0 - p_for_red) - 0.5,
        p_blue_win_for_blue=p_for_blue,
        p_blue_win_for_red=p_for_red,
        win_matrix=win_matrix,
        blue_configs=blue_configs,
        red_configs=red_configs,
    )
