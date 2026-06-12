"""Predictor protocol, pool-based role/build sampling, and terminal reward resolution.

The sampler is the bridge between the env (which only knows champion
indices) and the predictor (which needs full (champion, role, build)
tuples). Roles and builds are unknown until the end of the draft, so
the sampler enumerates plausible joint assignments — never assuming
pick order == role.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Callable, Literal, Protocol

import numpy as np

from app.ml.config import POSITIONS
from app.rl.pool import ChampionPool

RewardMode = Literal["expected_value", "risk_adjusted", "worst_case"]


@dataclass
class RoleBuildConfig:
    """One plausible hidden assignment for a team."""

    roles: dict[int, str]  # champion_id -> role
    builds: dict[int, int]  # champion_id -> build_id
    probability: float = 1.0  # normalised weight within the returned set
    # Share of the team's full enumerated assignment mass the returned set
    # covers (1.0 when nothing was truncated). The same value is stamped on
    # every config from one sampler call.
    retained_mass: float = 1.0


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
    # Truncation coverage of each side's config set (see RoleBuildConfig).
    blue_retained_mass: float = 1.0
    red_retained_mass: float = 1.0


class RoleBuildOptimizer(Protocol):
    """Custom hook that replaces the default sampler + reward-mode aggregation."""

    def __call__(
        self,
        blue_team: list[int],
        red_team: list[int],
        predictor: Predictor,
        reward_mode: RewardMode,
    ) -> OptimizationResult: ...


def _normalized_config_weights(
    configs: list[RoleBuildConfig],
    *,
    side: str,
) -> np.ndarray:
    weights = np.asarray([cfg.probability for cfg in configs], dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError(
            f"{side} role/build probabilities must be finite and non-negative"
        )
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(
            f"{side} role/build probabilities must sum to a positive value"
        )
    return weights / total


def make_pool_sampler(
    pool: ChampionPool,
    top_k_build_configs: int,
) -> RoleBuildSampler:
    """Build a sampler that enumerates valid (role, build) configs from `pool`.

    For each team of 5 champions:
      1. Enumerate every permutation of the 5 champions across the 5 roles
         where every (champion, role) appears in the pool.
      2. For each valid role-assignment, enumerate the Cartesian product
         of per-champion (build, weight) candidates.
      3. Score each (role-assignment, build-assignment) by the product of
         per-champion weights, keep the top K overall, and normalise
         their `probability` to sum to 1.

    If fewer than K configs exist, all of them are returned. If no valid
    role-assignment exists for the team (the pool is missing entries for
    one of the champions in any role), raises ValueError — that signals
    the pool is stale relative to the predictor's champion set.
    """
    if top_k_build_configs <= 0:
        raise ValueError("top_k_build_configs must be >= 1")

    def sampler(team_champions: list[int], side: str) -> list[RoleBuildConfig]:
        del side  # symmetric for now; kept for protocol parity
        if len(team_champions) != len(POSITIONS):
            raise ValueError(
                f"Expected {len(POSITIONS)} champions, got {len(team_champions)}"
            )

        scored: list[tuple[float, dict[int, str], dict[int, int]]] = []
        for role_perm in permutations(POSITIONS):
            assignment = dict(zip(team_champions, role_perm))
            per_champ_builds: list[tuple[tuple[int, float], ...]] = []
            ok = True
            for champ, role in assignment.items():
                builds = pool.builds_for(champ, role)
                if not builds:
                    ok = False
                    break
                per_champ_builds.append(builds)
            if not ok:
                continue
            for combo in product(*per_champ_builds):
                w = 1.0
                build_assignment: dict[int, int] = {}
                for champ, (build_id, weight) in zip(team_champions, combo):
                    w *= weight
                    build_assignment[champ] = build_id
                scored.append((w, assignment, build_assignment))

        if not scored:
            raise ValueError(
                f"No valid role/build assignment for team {team_champions}; "
                f"champion pool is missing entries."
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        full_mass = sum(w for w, _, _ in scored)
        if full_mass <= 0.0:
            raise ValueError(
                f"All role/build assignment weights are zero for team "
                f"{team_champions}; the champion pool carries no usable mass."
            )
        top = scored[:top_k_build_configs]
        total = sum(w for w, _, _ in top)
        retained = total / full_mass
        return [
            RoleBuildConfig(
                roles=roles,
                builds=builds,
                probability=w / total,
                retained_mass=retained,
            )
            for w, roles, builds in top
        ]

    return sampler


def resolve_rewards(
    blue_team: list[int],
    red_team: list[int],
    predictor: Predictor,
    sampler: RoleBuildSampler,
    reward_mode: RewardMode,
    risk_lambda: float = 0.5,
    worst_case_min_probability: float = 0.0,
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
                       `worst_case_min_probability` drops configs below
                       that normalised weight before taking min/max, so a
                       barely plausible assignment cannot dominate the
                       reward; 0.0 keeps every config.
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

    blue_weights = _normalized_config_weights(blue_configs, side="blue")
    red_weights = _normalized_config_weights(red_configs, side="red")
    joint_weights = np.outer(blue_weights, red_weights)
    if reward_mode == "expected_value":
        m = float(np.sum(win_matrix * joint_weights))
        p_for_blue, p_for_red = m, m
    elif reward_mode == "risk_adjusted":
        m = float(np.sum(win_matrix * joint_weights))
        s = float(np.sqrt(np.sum(joint_weights * (win_matrix - m) ** 2)))
        p_for_blue = m - risk_lambda * s
        p_for_red = m + risk_lambda * s
    elif reward_mode == "worst_case":
        blue_mask = blue_weights >= worst_case_min_probability
        red_mask = red_weights >= worst_case_min_probability
        # The threshold is a guard against implausible tails, not a hard
        # filter: if it would empty a side, keep that side's full set.
        if not blue_mask.any():
            blue_mask[:] = True
        if not red_mask.any():
            red_mask[:] = True
        masked = win_matrix[np.ix_(blue_mask, red_mask)]
        p_for_blue = float(masked.min())
        p_for_red = float(masked.max())
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
        blue_retained_mass=min(cfg.retained_mass for cfg in blue_configs),
        red_retained_mass=min(cfg.retained_mass for cfg in red_configs),
    )
