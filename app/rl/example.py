"""Random episode with a dummy predictor + sampler, demonstrating action masking.

Run:
    python -m app.rl.example
"""

from __future__ import annotations

import numpy as np

from app.ml.config import POSITIONS
from app.rl.env import DraftEnv, DraftEnvConfig
from app.rl.reward import RoleBuildConfig


def dummy_predictor(
    blue_team: list[int],
    red_team: list[int],
    blue_roles: dict[int, str],
    red_roles: dict[int, str],
    blue_builds: dict[int, int],
    red_builds: dict[int, int],
) -> float:
    blue_score = sum(c % 7 for c in blue_team) + sum(blue_builds.values())
    red_score = sum(c % 7 for c in red_team) + sum(red_builds.values())
    return float(1.0 / (1.0 + np.exp(-0.1 * (blue_score - red_score))))


def dummy_sampler(team_champions: list[int], side: str) -> list[RoleBuildConfig]:
    canonical = RoleBuildConfig(
        roles=dict(zip(team_champions, POSITIONS)),
        builds={c: 0 for c in team_champions},
        probability=0.7,
    )
    rotated = RoleBuildConfig(
        roles=dict(zip(team_champions, POSITIONS[1:] + POSITIONS[:1])),
        builds={c: 1 for c in team_champions},
        probability=0.3,
    )
    return [canonical, rotated]


def run_random_episode(seed: int = 0) -> None:
    env = DraftEnv(
        predictor=dummy_predictor,
        config=DraftEnvConfig(
            champion_ids=tuple(range(170)),
            agent_side="self_play",
            reward_mode="expected_value",
        ),
        sampler=dummy_sampler,
    )
    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        mask = env.get_action_mask()
        step = info["draft_step"]
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, reward, terminated, truncated, info = env.step(action)
        side_str = "blue" if step.side == 0 else "red"
        kind_str = "ban" if step.action_type == 0 else "pick"
        print(
            f"  {step.label:>4} ({side_str:>4} {kind_str:>4}) "
            f"-> idx {action:>3}  reward={reward:+.3f}"
        )
        done = terminated or truncated

    print()
    print(f"Blue picks (idx): {list(obs['blue_picks'])}")
    print(f"Red picks  (idx): {list(obs['red_picks'])}")
    print(f"Blue bans  (idx): {list(obs['blue_bans'])}")
    print(f"Red bans   (idx): {list(obs['red_bans'])}")
    print()
    print(f"p_blue_win (blue): {info['p_blue_win_for_blue']:.4f}")
    print(f"p_blue_win (red):  {info['p_blue_win_for_red']:.4f}")
    print(f"Blue reward: {info['blue_reward']:+.4f}")
    print(f"Red reward:  {info['red_reward']:+.4f}")


if __name__ == "__main__":
    run_random_episode()
