"""Masked-categorical torch policy for the draft env.

Observation encoder produces a flat float vector. Policy forward returns
logits over indices; sampling and log-prob always apply the action mask
so illegal actions have zero probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from app.rl.draft import DRAFT_SEQUENCE

# 4 multi-hot vectors (blue/red picks, blue/red bans) + 3 scalars.
CTX_DIM = 3


def obs_dim(n_champions: int) -> int:
    return 4 * n_champions + CTX_DIM


def encode_obs(obs: dict[str, Any], n_champions: int) -> np.ndarray:
    """Flatten a DraftEnv observation into a float32 feature vector."""
    feat = np.zeros(obs_dim(n_champions), dtype=np.float32)
    for offset, key in enumerate(("blue_picks", "red_picks", "blue_bans", "red_bans")):
        slots = obs[key]
        valid = slots[slots >= 0]
        if valid.size:
            feat[offset * n_champions + valid] = 1.0
    feat[-3] = float(obs["acting_side"])
    feat[-2] = float(obs["action_type"])
    feat[-1] = float(obs["step"]) / float(len(DRAFT_SEQUENCE))
    return feat


@dataclass(frozen=True)
class PolicyConfig:
    n_champions: int
    hidden: int = 256


class MaskedPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(obs_dim(cfg.n_champions), cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.n_champions),
        )

    def logits(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(features).masked_fill(~mask, -1e9)

    @torch.no_grad()
    def act(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        *,
        greedy: bool = False,
    ) -> int:
        logits = self.logits(
            torch.from_numpy(features)[None],
            torch.from_numpy(mask)[None],
        )[0]
        if greedy:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
