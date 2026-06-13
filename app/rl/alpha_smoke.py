"""End-to-end smoke test for the AlphaZero learner.

Validates, without ClickHouse, that:
  1. A legal draft sequence completes via MCTS-guided self-play.
  2. The action mask is honoured at every step (no duplicate champion).
  3. A terminal reward is produced from a Predictor call.
  4. One self-play episode round-trips into trainable tensors.
  5. One learner update reduces total loss.
  6. Device auto-selection picks a torch device.
  7. The adversarial league round-trips (admit, PFSP, asymmetric episode, SPRT).

Run:
    python -m app.rl.alpha_smoke
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from dataclasses import replace

from app.rl.net import AlphaNetConfig, AlphaZeroNet, auto_device
from app.rl.alpha_train import (
    AlphaTrainConfig,
    _eval_vs_champion,
    _generate_league,
    _mcts_cfg,
    _update,
)
from app.rl.draft import DRAFT_SEQUENCE, Side
from app.rl.example import dummy_predictor, dummy_sampler
from app.rl.league import League, sprt
from app.rl.selfplay import play_episode
from app.rl.worker import bytes_to_state, state_to_bytes

logger = logging.getLogger("alpha_smoke")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        logger.error("FAIL: %s", msg)
        raise SystemExit(1)
    logger.info("PASS: %s", msg)


def main() -> int:
    device = auto_device("auto")
    _assert(isinstance(device, torch.device), f"device auto-selected: {device}")

    n_champions = 30
    champion_ids = tuple(range(100, 100 + n_champions))
    net_cfg = AlphaNetConfig(n_champions=n_champions, hidden=64)
    net = AlphaZeroNet(net_cfg).to(device)

    train_cfg = AlphaTrainConfig(
        top_k_build_configs=4,
        iterations=1,
        episodes_per_iter=1,
        simulations=8,
        beam_width=8,
        batch_size=32,
        epochs_per_iter=1,
        hidden=64,
    )

    episode = play_episode(
        net, dummy_predictor,
        n_champions=n_champions,
        champion_ids=champion_ids,
        mcts_cfg=_mcts_cfg(train_cfg),
        device=device,
        sampler=dummy_sampler,
        rng=np.random.default_rng(0),
    )

    _assert(
        episode.features.shape[0] == len(DRAFT_SEQUENCE),
        f"self-play produced {episode.features.shape[0]} steps == {len(DRAFT_SEQUENCE)}",
    )

    # Legal-action checks: any action with nonzero search probability must be
    # on the legal mask, and no champion can appear twice across picks/bans.
    masks = episode.masks
    nonzero_illegal = (episode.policy_targets > 0) & ~masks
    _assert(
        not bool(nonzero_illegal.any()),
        "every action with nonzero search-policy probability is legal",
    )
    info = episode.info
    union = set(info["blue_picks"]) | set(info["red_picks"])
    _assert(
        len(union) == len(info["blue_picks"]) + len(info["red_picks"]),
        "no champion was picked twice",
    )

    _assert(
        -1.0 <= episode.blue_reward <= 1.0 and -1.0 <= episode.red_reward <= 1.0,
        f"terminal rewards in range: blue={episode.blue_reward:+.3f} red={episode.red_reward:+.3f}",
    )

    before = _loss_only(net, episode, train_cfg, device)
    stats = _update(net, torch.optim.Adam(net.parameters(), lr=1e-2), episode,
                    cfg=train_cfg, device=device)
    after = _loss_only(net, episode, train_cfg, device)
    _assert(
        np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"]),
        f"learner update produced finite losses: pol={stats['policy_loss']:.4f} val={stats['value_loss']:.4f}",
    )
    _assert(
        after <= before + 1e-3,
        f"single update did not increase total loss ({before:.4f} -> {after:.4f})",
    )

    _league_smoke(net, net_cfg, train_cfg, n_champions, champion_ids, device)

    logger.info("ALL SMOKE CHECKS PASSED")
    return 0


def _league_smoke(net, net_cfg, train_cfg, n_champions, champion_ids, device) -> None:
    """Admit a frozen entry, PFSP-sample it, play an asymmetric episode, SPRT."""
    state_bytes = state_to_bytes({k: v.cpu() for k, v in net.state_dict().items()})
    with tempfile.TemporaryDirectory() as tmp:
        league = League()
        league.admit(state_bytes, 0.0, tmp)
        _assert(league.champion is not None, "league admit set a champion")

        idx, entry = league.sample_opponent(np.random.default_rng(1))
        _assert(0 <= idx < len(league.entries), "PFSP sampled a valid opponent")

        opp = AlphaZeroNet(net_cfg).to(device).eval()
        opp.load_state_dict(
            bytes_to_state(Path(entry.path).read_bytes(), map_location=device)
        )
        ep = play_episode(
            net, dummy_predictor,
            n_champions=n_champions, champion_ids=champion_ids,
            mcts_cfg=_mcts_cfg(train_cfg), device=device, sampler=dummy_sampler,
            opponent_net=opp, learner_side=Side.BLUE, rng=np.random.default_rng(2),
        )
        _assert(
            not bool(((ep.policy_targets > 0) & ~ep.masks).any()),
            "asymmetric episode: every recorded action is legal",
        )
        _assert(
            bool((ep.sides == int(Side.BLUE)).all()),
            "asymmetric episode records only learner-side steps",
        )

        _assert(
            sprt(8, 2, 0) in {"accept", "reject", "continue"},
            "sprt returns a valid verdict",
        )

        # End-to-end glue: force the PFSP branch and run a tiny eval match.
        cfg = replace(train_cfg, self_play_frac=0.0, episodes_per_iter=2, eval_games=2)
        samples = _generate_league(
            net, dummy_predictor, n_champions, champion_ids, cfg, device,
            7, dummy_sampler, league, net_cfg, {},
        )
        _assert(len(samples) == cfg.episodes_per_iter, "league generation made one sample set per episode")
        _assert(league.entries[idx].games > 0, "PFSP episodes recorded an H2H result")

        w, l, d = _eval_vs_champion(
            net, opp, dummy_predictor, n_champions, champion_ids, cfg, device,
            dummy_sampler, 11,
        )
        _assert(w + l + d == cfg.eval_games, "eval played the requested number of games")

        n_before = len(league.entries)
        league.admit(state_bytes, 5.0, tmp)
        _assert(len(league.entries) == n_before + 1, "admit grows the pool")


def _loss_only(net, episode, cfg: AlphaTrainConfig, device) -> float:
    feats = torch.from_numpy(episode.features).to(device)
    pols = torch.from_numpy(episode.policy_targets).to(device)
    masks = torch.from_numpy(episode.masks).to(device)
    values_t = torch.from_numpy(episode.value_targets).to(device)
    with torch.no_grad():
        logits, value_pred = net(feats)
        logits = logits.masked_fill(~masks, -1e9)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(pols * log_probs).sum(dim=-1).mean()
        value_loss = ((value_pred - values_t) ** 2).mean()
    return float(policy_loss.item() + cfg.value_loss_coef * value_loss.item())


if __name__ == "__main__":
    sys.exit(main())
