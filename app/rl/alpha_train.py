"""AlphaZero-style training loop for the draft env.

Run:
    python -m app.rl.alpha_train

Each epoch:
  1. Generate ``episodes_per_epoch`` self-play episodes (MCTS-improved).
  2. Concatenate per-step samples into a replay buffer.
  3. Run ``epochs_per_iter`` mini-batch SGD passes (policy CE + value MSE).
  4. Log scalars and persist a checkpoint.

Workers run rollouts in parallel via ``multiprocessing`` when
``n_workers > 1``. Each worker owns its own predictor + net copy on the
chosen device.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from app.core.config.settings import PROJECT_ROOT
from app.core.logging.logger import setup_logging_config
from app.ml.predictor import load_predictor
from app.rl.draft import Side
from app.rl.league import League, elo_update, sprt
from app.rl.net import AlphaNetConfig, AlphaZeroNet, auto_device
from app.rl.mcts import MCTSConfig
from app.rl.pool import DEFAULT_POOL_PATH, load_pool
from app.rl.reward import RewardMode, make_pool_sampler
from app.rl.selfplay import EpisodeSamples, play_episode
from app.rl.worker import bytes_to_state, make_spawn_pool, state_to_bytes

setup_logging_config()
logger = logging.getLogger(__name__)

RL_DATA_DIR = PROJECT_ROOT / "app" / "rl" / "data"


@dataclass(kw_only=True)
class AlphaTrainConfig:
    top_k_build_configs: int
    iterations: int = 50
    episodes_per_iter: int = 16
    n_workers: int = 1
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    reward_mode: RewardMode = "expected_value"
    risk_lambda: float = 0.5
    pool_path: Path = DEFAULT_POOL_PATH
    # MCTS
    simulations: int = 64
    c_puct: float = 1.5
    beam_width: int = 32
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature: float = 1.0
    temperature_drop_step: int = 10
    # Net + optimisation
    hidden: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 256
    epochs_per_iter: int = 2
    value_loss_coef: float = 1.0
    # Adversarial league (off by default; forces inline generation)
    league: bool = False
    league_dir: Path | None = None
    self_play_frac: float = 0.5
    eval_games: int = 64
    promote_every: int = 5
    elo0: float = 0.0
    elo1: float = 15.0
    sprt_alpha: float = 0.05
    sprt_beta: float = 0.05
    # Persistence
    run_name: str | None = None
    save_every: int = 5


def _mcts_cfg(cfg: AlphaTrainConfig) -> MCTSConfig:
    return MCTSConfig(
        simulations=cfg.simulations,
        c_puct=cfg.c_puct,
        beam_width=cfg.beam_width,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_eps=cfg.dirichlet_eps,
        temperature=cfg.temperature,
        temperature_drop_step=cfg.temperature_drop_step,
    )


def _stack(samples: list[EpisodeSamples]) -> EpisodeSamples:
    return EpisodeSamples(
        features=np.concatenate([s.features for s in samples], axis=0),
        policy_targets=np.concatenate([s.policy_targets for s in samples], axis=0),
        masks=np.concatenate([s.masks for s in samples], axis=0),
        value_targets=np.concatenate([s.value_targets for s in samples], axis=0),
        sides=np.concatenate([s.sides for s in samples], axis=0),
        blue_reward=float(np.mean([s.blue_reward for s in samples])),
        red_reward=float(np.mean([s.red_reward for s in samples])),
        info={"n_episodes": len(samples)},
    )


# ---- worker plumbing -------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(net_cfg_dict: dict, train_cfg_dict: dict, device_str: str) -> None:
    torch.set_num_threads(1)
    predictor = load_predictor()
    net_cfg = AlphaNetConfig(**net_cfg_dict)
    device = torch.device(device_str)
    net = AlphaZeroNet(net_cfg).to(device).eval()
    train_cfg = AlphaTrainConfig(**train_cfg_dict)
    sampler = make_pool_sampler(
        load_pool(train_cfg.pool_path), train_cfg.top_k_build_configs
    )
    _WORKER.update(
        {
            "predictor": predictor,
            "net": net,
            "device": device,
            "train_cfg": train_cfg,
            "champion_ids": predictor.champion_ids,
            "n_champions": len(predictor.champion_ids),
            "sampler": sampler,
        }
    )


def _worker_play(args: tuple[bytes, int]) -> EpisodeSamples:
    weights_bytes, seed = args
    _WORKER["net"].load_state_dict(
        bytes_to_state(weights_bytes, map_location=_WORKER["device"])
    )
    cfg: AlphaTrainConfig = _WORKER["train_cfg"]
    return play_episode(
        _WORKER["net"],
        _WORKER["predictor"],
        n_champions=_WORKER["n_champions"],
        champion_ids=_WORKER["champion_ids"],
        mcts_cfg=_mcts_cfg(cfg),
        device=_WORKER["device"],
        reward_mode=cfg.reward_mode,
        risk_lambda=cfg.risk_lambda,
        sampler=_WORKER["sampler"],
        rng=np.random.default_rng(seed),
    )


def _generate_inline(
    net: AlphaZeroNet,
    predictor,
    n_champions,
    champion_ids,
    cfg: AlphaTrainConfig,
    device,
    base_seed: int,
    sampler,
) -> list[EpisodeSamples]:
    out: list[EpisodeSamples] = []
    net.eval()
    for i in range(cfg.episodes_per_iter):
        out.append(
            play_episode(
                net,
                predictor,
                n_champions=n_champions,
                champion_ids=champion_ids,
                mcts_cfg=_mcts_cfg(cfg),
                device=device,
                reward_mode=cfg.reward_mode,
                risk_lambda=cfg.risk_lambda,
                sampler=sampler,
                rng=np.random.default_rng(base_seed + i),
            )
        )
    return out


# ---- adversarial league ----------------------------------------------


def _load_opponent(
    path: str, net_cfg: AlphaNetConfig, device, cache: dict[str, AlphaZeroNet]
) -> AlphaZeroNet:
    """Instantiate + load a frozen league checkpoint, cached by path."""
    net = cache.get(path)
    if net is None:
        net = AlphaZeroNet(net_cfg).to(device).eval()
        net.load_state_dict(bytes_to_state(Path(path).read_bytes(), map_location=device))
        cache[path] = net
    return net


def _league_episode(
    net, opp, predictor, n_champions, champion_ids, cfg, device, sampler, learner_side, rng
) -> tuple[EpisodeSamples, float]:
    """One asymmetric episode (learner=MCTS, opponent=frozen greedy). Returns
    the learner-side samples and the learner's terminal reward."""
    ep = play_episode(
        net, predictor,
        n_champions=n_champions, champion_ids=champion_ids,
        mcts_cfg=_mcts_cfg(cfg), device=device,
        reward_mode=cfg.reward_mode, risk_lambda=cfg.risk_lambda, sampler=sampler,
        opponent_net=opp, learner_side=learner_side, rng=rng,
    )
    reward = ep.blue_reward if learner_side == Side.BLUE else ep.red_reward
    return ep, reward


def _generate_league(
    net, predictor, n_champions, champion_ids, cfg, device, base_seed,
    sampler, league: League, net_cfg, opp_cache,
) -> list[EpisodeSamples]:
    """Inline generation mixing self-play with PFSP asymmetric episodes."""
    out: list[EpisodeSamples] = []
    net.eval()
    for i in range(cfg.episodes_per_iter):
        rng = np.random.default_rng(base_seed + i)
        if not league.entries or rng.random() < cfg.self_play_frac:
            out.append(
                play_episode(
                    net, predictor,
                    n_champions=n_champions, champion_ids=champion_ids,
                    mcts_cfg=_mcts_cfg(cfg), device=device,
                    reward_mode=cfg.reward_mode, risk_lambda=cfg.risk_lambda,
                    sampler=sampler, rng=rng,
                )
            )
            continue
        idx, entry = league.sample_opponent(rng)
        opp = _load_opponent(entry.path, net_cfg, device, opp_cache)
        learner_side = Side.BLUE if rng.random() < 0.5 else Side.RED
        ep, reward = _league_episode(
            net, opp, predictor, n_champions, champion_ids, cfg, device,
            sampler, learner_side, rng,
        )
        league.record(idx, reward > 0)
        out.append(ep)
    return out


def _eval_vs_champion(
    net, champion_net, predictor, n_champions, champion_ids, cfg, device, sampler, base_seed
) -> tuple[int, int, int]:
    """Side-balanced learner-vs-champion match; returns (wins, losses, draws)."""
    wins = losses = draws = 0
    for i in range(cfg.eval_games):
        learner_side = Side.BLUE if i % 2 == 0 else Side.RED
        _, reward = _league_episode(
            net, champion_net, predictor, n_champions, champion_ids, cfg, device,
            sampler, learner_side, np.random.default_rng(base_seed + i),
        )
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


# ---- training loop ---------------------------------------------------


def _update(
    net: AlphaZeroNet,
    opt: torch.optim.Optimizer,
    batch: EpisodeSamples,
    *,
    cfg: AlphaTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    feats = torch.from_numpy(batch.features).to(device)
    pols = torch.from_numpy(batch.policy_targets).to(device)
    masks = torch.from_numpy(batch.masks).to(device)
    values_t = torch.from_numpy(batch.value_targets).to(device)

    n = feats.shape[0]
    perm = torch.randperm(n, device=device)
    pol_loss_sum = 0.0
    val_loss_sum = 0.0
    grad_norm_sum = 0.0
    n_steps = 0
    for epoch in range(cfg.epochs_per_iter):
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            bf, bp, bm, bv = feats[idx], pols[idx], masks[idx], values_t[idx]
            logits, value_pred = net(bf)
            logits = logits.masked_fill(~bm, -1e9)
            log_probs = torch.log_softmax(logits, dim=-1)
            policy_loss = -(bp * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(value_pred, bv)
            loss = policy_loss + cfg.value_loss_coef * value_loss
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            pol_loss_sum += float(policy_loss.item())
            val_loss_sum += float(value_loss.item())
            grad_norm_sum += float(gn.item())
            n_steps += 1
        perm = torch.randperm(n, device=device)
    return {
        "policy_loss": pol_loss_sum / max(n_steps, 1),
        "value_loss": val_loss_sum / max(n_steps, 1),
        "grad_norm": grad_norm_sum / max(n_steps, 1),
        "train_steps": float(n_steps),
    }


def train(cfg: AlphaTrainConfig) -> Path:
    device = auto_device(cfg.device)
    logger.info("AlphaZero device: %s", device)

    predictor = load_predictor()
    n_champions = len(predictor.champion_ids)
    champion_ids = predictor.champion_ids
    net_cfg = AlphaNetConfig(n_champions=n_champions, hidden=cfg.hidden)
    net = AlphaZeroNet(net_cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    inline_sampler = make_pool_sampler(load_pool(cfg.pool_path), cfg.top_k_build_configs)

    run_name = cfg.run_name or time.strftime("alpha_%Y%m%d_%H%M%S")
    ckpt_dir = RL_DATA_DIR / "policies"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = RL_DATA_DIR / "logs" / f"{run_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # League setup: load (or bootstrap) the opponent pool and the live Elo.
    league_dir = cfg.league_dir or (ckpt_dir / "league")
    learner_rating = cfg.elo0
    opp_cache: dict[str, AlphaZeroNet] = {}
    league: League | None = None
    if cfg.league:
        league = League.load(league_dir)
        if not league.entries:
            league.admit(
                state_to_bytes({k: v.cpu() for k, v in net.state_dict().items()}),
                learner_rating, league_dir,
            )
            league.save(league_dir)
        learner_rating = league.champion.rating

    pool = None
    if cfg.n_workers > 1 and not cfg.league:
        pool = make_spawn_pool(
            cfg.n_workers,
            _worker_init,
            (asdict(net_cfg), asdict(cfg), str(device)),
        )

    try:
        for it in range(cfg.iterations):
            t0 = time.time()
            base_seed = 10_000 * (it + 1)
            if league is not None:
                samples = _generate_league(
                    net, predictor, n_champions, champion_ids, cfg, device,
                    base_seed, inline_sampler, league, net_cfg, opp_cache,
                )
            elif pool is not None:
                weights = state_to_bytes(
                    {k: v.cpu() for k, v in net.state_dict().items()}
                )
                args = [(weights, base_seed + i) for i in range(cfg.episodes_per_iter)]
                samples = pool.map(_worker_play, args)
            else:
                samples = _generate_inline(
                    net,
                    predictor,
                    n_champions,
                    champion_ids,
                    cfg,
                    device,
                    base_seed,
                    inline_sampler,
                )
            t_play = time.time() - t0

            batch = _stack(samples)
            t1 = time.time()
            stats = _update(net, opt, batch, cfg=cfg, device=device)
            t_update = time.time() - t1

            metrics = {
                "iter": it,
                "episodes": cfg.episodes_per_iter,
                "steps": int(batch.features.shape[0]),
                "blue_reward_mean": float(batch.blue_reward),
                "red_reward_mean": float(batch.red_reward),
                "value_target_mean": float(batch.value_targets.mean()),
                "play_sec": t_play,
                "update_sec": t_update,
                **stats,
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(metrics) + "\n")
            logger.info(
                "iter=%d play=%.1fs upd=%.2fs blue=%.3f red=%.3f pol=%.4f val=%.4f",
                it,
                t_play,
                t_update,
                metrics["blue_reward_mean"],
                metrics["red_reward_mean"],
                metrics["policy_loss"],
                metrics["value_loss"],
            )

            # SPRT-gated promotion: eval vs champion, update Elo, admit on accept.
            if league is not None and (it + 1) % cfg.promote_every == 0:
                champ_net = _load_opponent(
                    league.champion.path, net_cfg, device, opp_cache
                )
                w, l, d = _eval_vs_champion(
                    net, champ_net, predictor, n_champions, champion_ids, cfg,
                    device, inline_sampler, 900_000 * (it + 1),
                )
                score = (w + 0.5 * d) / max(w + l + d, 1)
                learner_rating, _ = elo_update(
                    learner_rating, league.champion.rating, score
                )
                verdict = sprt(
                    w, l, d, elo0=cfg.elo0, elo1=cfg.elo1,
                    alpha=cfg.sprt_alpha, beta=cfg.sprt_beta,
                )
                logger.info(
                    "league eval w=%d l=%d d=%d score=%.3f elo=%.1f verdict=%s",
                    w, l, d, score, learner_rating, verdict,
                )
                if verdict == "accept":
                    league.admit(
                        state_to_bytes(
                            {k: v.cpu() for k, v in net.state_dict().items()}
                        ),
                        learner_rating, league_dir,
                    )
                    league.save(league_dir)
                    logger.info(
                        "PROMOTED -> champion (entries=%d, elo=%.1f)",
                        len(league.entries), learner_rating,
                    )

            if (it + 1) % cfg.save_every == 0 or (it + 1) == cfg.iterations:
                ckpt = ckpt_dir / f"{run_name}.pt"
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "net_cfg": asdict(net_cfg),
                        "train_cfg": asdict(cfg),
                        "iter": it,
                    },
                    ckpt,
                )
                logger.info("Saved checkpoint: %s", ckpt)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    final = ckpt_dir / f"{run_name}.pt"
    return final


if __name__ == "__main__":
    train(AlphaTrainConfig(top_k_build_configs=8))
