"""Shared policy-value network for AlphaZero-style draft search.

Uses the same `encode_obs` features as `MaskedPolicy` so the trunk sees
the public draft state plus a few scalar context features. The policy
head outputs logits over champion indices; the value head outputs a
scalar in [-1, 1] from the perspective of the side currently acting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from app.rl.policy import obs_dim


@dataclass(frozen=True)
class AlphaNetConfig:
    n_champions: int
    hidden: int = 256


class AlphaZeroNet(nn.Module):
    def __init__(self, cfg: AlphaNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = obs_dim(cfg.n_champions)
        self.trunk = nn.Sequential(
            nn.Linear(d, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(cfg.hidden, cfg.n_champions)
        self.value_head = nn.Linear(cfg.hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return logits, value


def auto_device(prefer: str = "auto") -> torch.device:
    """Pick CUDA > MPS > CPU. Honour an explicit choice when given."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
