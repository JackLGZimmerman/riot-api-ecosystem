"""Persistent multiprocessing worker pool for parallel draft rollouts.

Each worker process owns a long-lived predictor + env + policy. The
master sends new policy weights and a batch size; the worker runs that
many self-play episodes and returns flat per-step arrays plus terminal
side rewards. Episodes are independent so this scales near-linearly.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any

import numpy as np
import torch

from app.ml.predictor import load_predictor
from app.rl.env import DraftEnv, DraftEnvConfig
from app.rl.policy import MaskedPolicy, PolicyConfig, encode_obs

# Worker-local globals (process-scoped — picklable boundary is the pool init).
_PREDICTOR = None
_ENV: DraftEnv | None = None
_POLICY: MaskedPolicy | None = None
_N_CHAMPIONS: int = 0


@dataclass
class EpisodeBatch:
    """Flat arrays across all timesteps of a worker's episode batch."""

    features: np.ndarray  # [T, obs_dim] float32
    actions: np.ndarray  # [T] int64
    masks: np.ndarray  # [T, n_champ] bool
    returns: np.ndarray  # [T] float32  (side's terminal reward broadcast)
    blue_rewards: np.ndarray  # [n_episodes] float32
    red_rewards: np.ndarray  # [n_episodes] float32
    p_blue_win: np.ndarray  # [n_episodes] float32  (blue-perspective)


def _worker_init(env_cfg_dict: dict[str, Any], policy_cfg_dict: dict[str, Any]) -> None:
    global _PREDICTOR, _ENV, _POLICY, _N_CHAMPIONS
    torch.set_num_threads(1)  # prevent intra-op thread contention across workers
    _PREDICTOR = load_predictor()
    env_cfg = DraftEnvConfig(**env_cfg_dict)
    _ENV = DraftEnv(_PREDICTOR, env_cfg)
    _POLICY = MaskedPolicy(PolicyConfig(**policy_cfg_dict)).eval()
    _N_CHAMPIONS = env_cfg.n_champions


def _run_episode(
    seed: int,
    *,
    vs_random: bool,
    rng: np.random.Generator,
) -> tuple[
    list[np.ndarray], list[int], list[np.ndarray], list[int], float, float, float
]:
    """Roll one episode.

    vs_random=True : policy plays blue, uniform random plays red. Only blue
        transitions are returned (clean learning signal).
    vs_random=False: self-play — policy plays both sides; transitions from
        both sides are returned, each step receiving its own side's reward.
    """
    assert _ENV is not None and _POLICY is not None
    obs, info = _ENV.reset(seed=seed)
    feats: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    sides: list[int] = []
    done = False
    while not done:
        step = info["draft_step"]
        mask = obs["available_mask"].astype(bool)
        is_blue = int(step.side) == 0
        if vs_random and not is_blue:
            action = int(rng.choice(np.flatnonzero(mask)))
        else:
            feat = encode_obs(obs, _N_CHAMPIONS)
            action = _POLICY.act(feat, mask)
            if not vs_random or is_blue:
                feats.append(feat)
                actions.append(action)
                masks.append(mask)
                sides.append(int(step.side))
        obs, _, terminated, truncated, info = _ENV.step(action)
        done = terminated or truncated
    return (
        feats,
        actions,
        masks,
        sides,
        float(info["blue_reward"]),
        float(info["red_reward"]),
        float(info["p_blue_win_for_blue"]),
    )


def _worker_rollout(args: tuple[bytes, int, int, bool]) -> EpisodeBatch:
    weights_bytes, n_episodes, base_seed, vs_random = args
    assert _POLICY is not None
    state = torch.load(io.BytesIO(weights_bytes), weights_only=True)
    _POLICY.load_state_dict(state)
    _POLICY.eval()
    rng = np.random.default_rng(base_seed)

    all_feats: list[np.ndarray] = []
    all_actions: list[int] = []
    all_masks: list[np.ndarray] = []
    all_returns: list[float] = []
    blue_rewards: list[float] = []
    red_rewards: list[float] = []
    p_blue: list[float] = []

    for i in range(n_episodes):
        feats, actions, masks, sides, br, rr, pbb = _run_episode(
            base_seed + i,
            vs_random=vs_random,
            rng=rng,
        )
        per_side = (br, rr)
        all_feats.extend(feats)
        all_actions.extend(actions)
        all_masks.extend(masks)
        all_returns.extend(per_side[s] for s in sides)
        blue_rewards.append(br)
        red_rewards.append(rr)
        p_blue.append(pbb)

    return EpisodeBatch(
        features=np.stack(all_feats)
        if all_feats
        else np.zeros((0, 0), dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.int64),
        masks=np.stack(all_masks) if all_masks else np.zeros((0, 0), dtype=bool),
        returns=np.asarray(all_returns, dtype=np.float32),
        blue_rewards=np.asarray(blue_rewards, dtype=np.float32),
        red_rewards=np.asarray(red_rewards, dtype=np.float32),
        p_blue_win=np.asarray(p_blue, dtype=np.float32),
    )


def _state_to_bytes(state: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


class RolloutPool:
    """Persistent worker pool — open once, dispatch many batches."""

    def __init__(
        self,
        env_cfg: DraftEnvConfig,
        policy_cfg: PolicyConfig,
        n_workers: int | None = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.policy_cfg = policy_cfg
        self.n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
        ctx = get_context("spawn")
        self.pool = ctx.Pool(
            self.n_workers,
            initializer=_worker_init,
            initargs=(env_cfg.__dict__, policy_cfg.__dict__),
        )

    def rollout(
        self,
        policy: MaskedPolicy,
        episodes_per_worker: int,
        base_seed: int,
        *,
        vs_random: bool = False,
    ) -> list[EpisodeBatch]:
        weights = _state_to_bytes(policy.state_dict())
        seeds = [base_seed + i * episodes_per_worker for i in range(self.n_workers)]
        args = [(weights, episodes_per_worker, s, vs_random) for s in seeds]
        return self.pool.map(_worker_rollout, args)

    def close(self) -> None:
        self.pool.close()
        self.pool.join()

    def __enter__(self) -> RolloutPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
