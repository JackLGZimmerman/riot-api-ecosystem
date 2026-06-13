"""Networks for the draft env: masked policy (REINFORCE) + policy-value net (AlphaZero).

Both share `encode_obs` features (public draft state + scalar context) and the
same two-layer trunk. `MaskedPolicy` adds a single logits head; `AlphaZeroNet`
adds separate policy + value heads.
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


def _trunk(d: int, hidden: int) -> list[nn.Module]:
    """Two-layer ReLU body shared by both nets. Returned as a list so callers
    splat it (`*_trunk(...)`) into their own nn.Sequential — this keeps each
    net's existing state_dict keys (net.0/net.2/net.4, trunk.0/trunk.2)."""
    return [nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()]


@dataclass(frozen=True)
class PolicyConfig:
    n_champions: int
    hidden: int = 256


class MaskedPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            *_trunk(obs_dim(cfg.n_champions), cfg.hidden),
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


@dataclass(frozen=True)
class AlphaNetConfig:
    n_champions: int
    hidden: int = 256


class AlphaZeroNet(nn.Module):
    def __init__(self, cfg: AlphaNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = nn.Sequential(*_trunk(obs_dim(cfg.n_champions), cfg.hidden))
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
