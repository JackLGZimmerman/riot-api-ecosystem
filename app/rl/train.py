"""REINFORCE training loop for DraftEnv with parallel rollouts.

Run:
    python -m app.rl.train

Live charts:
    tensorboard --logdir app/rl/data/runs
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from app.core.config.settings import PROJECT_ROOT
from app.core.logging.logger import setup_logging_config
from app.ml.predictor import load_predictor
from app.rl.env import DraftEnv, DraftEnvConfig
from app.rl.policy import MaskedPolicy, PolicyConfig, encode_obs
from app.rl.pool import DEFAULT_POOL_PATH, load_pool
from app.rl.reward import make_pool_sampler
from app.rl.rollout import EpisodeBatch, RolloutPool

setup_logging_config()
logger = logging.getLogger(__name__)

RL_DATA_DIR = PROJECT_ROOT / "app" / "rl" / "data"


@dataclass(kw_only=True)
class TrainConfig:
    top_k_build_configs: int
    epochs: int = 200
    episodes_per_worker: int = 8
    n_workers: int | None = None
    lr: float = 3e-4
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    random_start_steps: int = 0
    eval_every: int = 5
    eval_episodes: int = 64
    hidden: int = 256
    train_mode: Literal["vs_random", "self_play"] = "vs_random"
    run_name: str | None = None
    pool_path: Path = DEFAULT_POOL_PATH


def _flatten(batches: list[EpisodeBatch]) -> EpisodeBatch:
    return EpisodeBatch(
        features=np.concatenate([b.features for b in batches], axis=0),
        actions=np.concatenate([b.actions for b in batches], axis=0),
        masks=np.concatenate([b.masks for b in batches], axis=0),
        returns=np.concatenate([b.returns for b in batches], axis=0),
        blue_rewards=np.concatenate([b.blue_rewards for b in batches], axis=0),
        red_rewards=np.concatenate([b.red_rewards for b in batches], axis=0),
        p_blue_win=np.concatenate([b.p_blue_win for b in batches], axis=0),
    )


def _policy_update(
    policy: MaskedPolicy,
    opt: torch.optim.Optimizer,
    batch: EpisodeBatch,
    *,
    entropy_coef: float,
    grad_clip: float,
) -> dict[str, float]:
    feats = torch.from_numpy(batch.features)
    actions = torch.from_numpy(batch.actions)
    masks = torch.from_numpy(batch.masks)
    returns = torch.from_numpy(batch.returns)
    # Per-side baseline subtraction via batch mean.
    advantage = returns - returns.mean()
    advantage = advantage / (advantage.std() + 1e-6)

    logits = policy.logits(feats, masks)
    log_probs_all = torch.log_softmax(logits, dim=-1)
    log_pa = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs_all).sum(dim=-1).mean()

    policy_loss = -(advantage * log_pa).mean()
    loss = policy_loss - entropy_coef * entropy

    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    opt.step()

    return {
        "policy_loss": float(policy_loss.item()),
        "entropy": float(entropy.item()),
        "grad_norm": float(grad_norm.item()),
    }


def _evaluate_vs_random(
    policy: MaskedPolicy,
    env: DraftEnv,
    n_episodes: int,
    *,
    n_champions: int,
    seed: int,
) -> dict[str, float]:
    """Policy plays blue, uniform-random plays red. Returns mean blue-side stats."""
    rng = np.random.default_rng(seed)
    blue_rewards = []
    p_blue = []
    illegal_count = 0
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        done = False
        while not done:
            step = info["draft_step"]
            mask = obs["available_mask"].astype(bool)
            if int(step.side) == 0:
                feat = encode_obs(obs, n_champions)
                action = policy.act(feat, mask, greedy=False)
            else:
                action = int(rng.choice(np.flatnonzero(mask)))
            obs, _, terminated, truncated, info = env.step(action)
            if "illegal_action" in info:
                illegal_count += 1
            done = terminated or truncated
        blue_rewards.append(float(info["blue_reward"]))
        p_blue.append(float(info["p_blue_win_for_blue"]))
    return {
        "eval_blue_reward_mean": float(np.mean(blue_rewards)),
        "eval_p_blue_win_mean": float(np.mean(p_blue)),
        "eval_win_rate": float(np.mean([r > 0.5 for r in p_blue])),
        "eval_illegal": int(illegal_count),
    }


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def train(cfg: TrainConfig) -> Path:
    predictor = load_predictor()
    n_champions = len(predictor.champion_ids)
    policy_cfg = PolicyConfig(n_champions=n_champions, hidden=cfg.hidden)
    env_cfg = DraftEnvConfig(
        top_k_build_configs=cfg.top_k_build_configs,
        champion_ids=predictor.champion_ids,
        agent_side="self_play" if cfg.train_mode == "self_play" else "blue",
        random_start_steps=cfg.random_start_steps,
    )
    vs_random = cfg.train_mode == "vs_random"

    policy = MaskedPolicy(policy_cfg)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    run_name = cfg.run_name or time.strftime("draft_%Y%m%d_%H%M%S")
    run_dir = RL_DATA_DIR / "runs" / run_name
    jsonl_path = RL_DATA_DIR / "logs" / f"{run_name}.jsonl"
    writer = SummaryWriter(log_dir=str(run_dir))
    logger.info(
        "Run: %s  workers=%s  episodes/worker=%d",
        run_name,
        cfg.n_workers,
        cfg.episodes_per_worker,
    )

    champion_pool = load_pool(cfg.pool_path)
    eval_sampler = make_pool_sampler(champion_pool, cfg.top_k_build_configs)
    eval_env = DraftEnv(
        predictor,
        DraftEnvConfig(
            top_k_build_configs=cfg.top_k_build_configs,
            champion_ids=predictor.champion_ids,
            agent_side="blue",
            random_start_steps=0,
        ),
        sampler=eval_sampler,
    )

    global_step = 0
    base_seed = 0
    with RolloutPool(
        env_cfg,
        policy_cfg,
        n_workers=cfg.n_workers,
        pool_path=cfg.pool_path,
    ) as pool:
        for epoch in range(cfg.epochs):
            t0 = time.time()
            batches = pool.rollout(
                policy,
                cfg.episodes_per_worker,
                base_seed,
                vs_random=vs_random,
            )
            batch = _flatten(batches)
            t_rollout = time.time() - t0

            t1 = time.time()
            update_stats = _policy_update(
                policy,
                opt,
                batch,
                entropy_coef=cfg.entropy_coef,
                grad_clip=cfg.grad_clip,
            )
            t_update = time.time() - t1

            n_episodes = batch.blue_rewards.size
            ep_per_sec = n_episodes / max(t_rollout, 1e-6)
            metrics = {
                "epoch": epoch,
                "n_episodes": int(n_episodes),
                "ep_per_sec": float(ep_per_sec),
                "rollout_sec": float(t_rollout),
                "update_sec": float(t_update),
                "blue_reward_mean": float(batch.blue_rewards.mean()),
                "red_reward_mean": float(batch.red_rewards.mean()),
                "p_blue_win_mean": float(batch.p_blue_win.mean()),
                "p_blue_win_std": float(batch.p_blue_win.std()),
                **update_stats,
            }

            if epoch % cfg.eval_every == 0:
                eval_stats = _evaluate_vs_random(
                    policy,
                    eval_env,
                    cfg.eval_episodes,
                    n_champions=n_champions,
                    seed=10_000 + epoch,
                )
                metrics.update(eval_stats)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(k, v, global_step)
            _write_jsonl(jsonl_path, metrics)
            global_step += 1
            base_seed += (
                cfg.n_workers * cfg.episodes_per_worker if cfg.n_workers else 1024
            )

            logger.info(
                "epoch=%d eps/s=%.1f p_blue=%.3f±%.3f loss=%.4f ent=%.3f%s",
                epoch,
                ep_per_sec,
                metrics["p_blue_win_mean"],
                metrics["p_blue_win_std"],
                metrics["policy_loss"],
                metrics["entropy"],
                f" eval_wr={metrics['eval_win_rate']:.3f}"
                if "eval_win_rate" in metrics
                else "",
            )

    writer.close()
    ckpt = RL_DATA_DIR / "policies" / f"{run_name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "policy_cfg": asdict(policy_cfg),
            "train_cfg": asdict(cfg),
        },
        ckpt,
    )
    logger.info("Saved policy: %s", ckpt)
    logger.info("JSONL log:    %s", jsonl_path)
    logger.info("TensorBoard:  tensorboard --logdir %s", run_dir.parent)
    return ckpt


if __name__ == "__main__":
    train(TrainConfig(top_k_build_configs=8))
