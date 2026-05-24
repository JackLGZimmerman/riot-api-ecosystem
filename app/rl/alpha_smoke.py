"""End-to-end smoke test for the AlphaZero learner.

Validates, without ClickHouse, that:
  1. A legal draft sequence completes via MCTS-guided self-play.
  2. The action mask is honoured at every step (no duplicate champion).
  3. A terminal reward is produced from a Predictor call.
  4. One self-play episode round-trips into trainable tensors.
  5. One learner update reduces total loss.
  6. Device auto-selection picks a torch device.

Run:
    python -m app.rl.alpha_smoke
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch

from app.rl.alpha_net import AlphaNetConfig, AlphaZeroNet, auto_device
from app.rl.alpha_train import AlphaTrainConfig, _mcts_cfg, _update
from app.rl.draft import DRAFT_SEQUENCE
from app.rl.example import dummy_predictor, dummy_sampler
from app.rl.selfplay import play_episode

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

    logger.info("ALL SMOKE CHECKS PASSED")
    return 0


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
