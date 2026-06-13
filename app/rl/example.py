"""Random episode with a dummy predictor + pool-based sampler.

Run:
    python -m app.rl.example
"""

from __future__ import annotations

import numpy as np

from app.ml.config import POSITIONS
from app.rl.pool import ChampionPool, PoolEntry
from app.rl.reward import make_pool_sampler


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


def dummy_pool(n_champions: int, *, champion_ids: tuple[int, ...] | None = None) -> ChampionPool:
    """Pool where every champion can play every role with two build options.

    Realistic pools (built from priors) restrict each champion to ~1-2 roles.
    `champion_ids` lets callers key the pool by real champion ids (passed to
    the predictor) instead of the env's positional indices.
    """
    ids = champion_ids if champion_ids is not None else tuple(range(n_champions))
    entries: dict[int, tuple[PoolEntry, ...]] = {}
    for cid in ids:
        rows: list[PoolEntry] = []
        for role in POSITIONS:
            rows.append(PoolEntry(role=role, build_id=0, weight=0.7))
            rows.append(PoolEntry(role=role, build_id=1, weight=0.3))
        entries[cid] = tuple(rows)
    return ChampionPool(entries=entries)


# Kept for tests/smoke imports — builds a permissive pool sampler over
# champion ids 100..130 (the range used by alpha_smoke).
dummy_sampler = make_pool_sampler(
    dummy_pool(30, champion_ids=tuple(range(100, 130))),
    top_k_build_configs=4,
)


def run_random_episode(seed: int = 0) -> None:
    from app.rl.env import DraftEnv, DraftEnvConfig  # lazy: only this path needs gym

    n_champions = 170
    pool = dummy_pool(n_champions)
    sampler = make_pool_sampler(pool, top_k_build_configs=4)
    env = DraftEnv(
        predictor=dummy_predictor,
        config=DraftEnvConfig(
            top_k_build_configs=4,
            champion_ids=tuple(range(n_champions)),
            agent_side="self_play",
            reward_mode="expected_value",
        ),
        sampler=sampler,
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
