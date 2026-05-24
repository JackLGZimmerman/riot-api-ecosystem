"""PUCT MCTS over draft continuations with action masking.

Tree search operates on a lightweight pure-data ``DraftState`` that
mirrors :class:`app.rl.env.DraftEnv` internals — cloning is just a few
small array/list copies. Leaf evaluation uses the policy-value net for
non-terminal states and the supplied ``Predictor`` for terminal states
via :func:`app.rl.reward.resolve_rewards`.

Per-node memory is kept small by retaining priors and visit stats only
for the top-``beam_width`` legal actions (a beam/MCTS hybrid). The full
legal mask still constrains exploration; the beam only constrains which
actions are *expanded*. With ~950 actions per node this prevents the
tree blowing up in memory while keeping the AlphaZero algorithm intact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from app.rl.alpha_net import AlphaZeroNet
from app.rl.draft import DRAFT_SEQUENCE, ActionType, DraftStep, Side
from app.rl.policy import encode_obs
from app.rl.reward import (
    Predictor,
    RewardMode,
    RoleBuildOptimizer,
    RoleBuildSampler,
    default_role_build_sampler,
    resolve_rewards,
)


@dataclass
class DraftState:
    """Pure-data mirror of :class:`DraftEnv` internals; cheap to clone."""

    n_champions: int
    blue_picks: list[int] = field(default_factory=list)
    red_picks: list[int] = field(default_factory=list)
    blue_bans: list[int] = field(default_factory=list)
    red_bans: list[int] = field(default_factory=list)
    available: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    step_idx: int = 0

    @classmethod
    def initial(cls, n_champions: int) -> "DraftState":
        return cls(
            n_champions=n_champions,
            available=np.ones(n_champions, dtype=np.int8),
        )

    def clone(self) -> "DraftState":
        return DraftState(
            n_champions=self.n_champions,
            blue_picks=list(self.blue_picks),
            red_picks=list(self.red_picks),
            blue_bans=list(self.blue_bans),
            red_bans=list(self.red_bans),
            available=self.available.copy(),
            step_idx=self.step_idx,
        )

    def current_step(self) -> DraftStep | None:
        if self.step_idx >= len(DRAFT_SEQUENCE):
            return None
        return DRAFT_SEQUENCE[self.step_idx]

    def is_terminal(self) -> bool:
        return self.step_idx >= len(DRAFT_SEQUENCE)

    def legal_mask(self) -> np.ndarray:
        return self.available.astype(bool)

    def apply(self, action: int) -> Side:
        step = DRAFT_SEQUENCE[self.step_idx]
        if not (0 <= action < self.n_champions) or not self.available[action]:
            raise ValueError(f"Illegal action {action} at step {self.step_idx}.")
        if step.action_type == ActionType.BAN:
            (self.blue_bans if step.side == Side.BLUE else self.red_bans).append(action)
        else:
            (self.blue_picks if step.side == Side.BLUE else self.red_picks).append(
                action
            )
        self.available[action] = 0
        self.step_idx += 1
        return step.side

    def to_obs(self) -> dict[str, Any]:
        def _pad(xs: list[int]) -> np.ndarray:
            out = np.full(5, -1, dtype=np.int32)
            for i, v in enumerate(xs):
                out[i] = v
            return out

        step = self.current_step()
        return {
            "blue_picks": _pad(self.blue_picks),
            "red_picks": _pad(self.red_picks),
            "blue_bans": _pad(self.blue_bans),
            "red_bans": _pad(self.red_bans),
            "available_mask": self.available.copy(),
            "step": self.step_idx,
            "acting_side": 0 if step is None else int(step.side),
            "action_type": 0 if step is None else int(step.action_type),
        }

    def current_side(self) -> Side | None:
        step = self.current_step()
        return None if step is None else step.side


@dataclass
class MCTSConfig:
    simulations: int = 64
    c_puct: float = 1.5
    beam_width: int = 32  # keep priors/stats only for top-K legal actions per node
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25  # noise mixed into the root prior; 0 disables
    temperature: float = 1.0  # visit-count temperature for action sampling
    temperature_drop_step: int = 10  # after this draft step, go greedy (temperature=0)


class _Node:
    __slots__ = (
        "side",
        "is_terminal",
        "actions",
        "priors",
        "N",
        "W",
        "children",
        "N_sum",
        "terminal_rewards",
    )

    def __init__(self, side: Side | None, is_terminal: bool) -> None:
        self.side: Side | None = side
        self.is_terminal: bool = is_terminal
        # Top-K beam over legal actions, set at expansion time.
        self.actions: np.ndarray | None = None  # int64 [K]
        self.priors: np.ndarray | None = None  # float32 [K], normalised within beam
        self.N: np.ndarray | None = None  # int32 [K]
        self.W: np.ndarray | None = None  # float32 [K]
        self.children: dict[int, _Node] = {}
        self.N_sum: int = 0
        # For terminal nodes: per-side reward (from each side's own perspective).
        self.terminal_rewards: dict[Side, float] | None = None


class MCTS:
    """One MCTS instance per self-play episode; net+predictor evaluations are cached."""

    def __init__(
        self,
        net: AlphaZeroNet,
        predictor: Predictor,
        cfg: MCTSConfig,
        device: torch.device,
        *,
        reward_mode: RewardMode = "expected_value",
        risk_lambda: float = 0.5,
        sampler: RoleBuildSampler | None = None,
        optimizer: RoleBuildOptimizer | None = None,
        champion_ids: tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.net = net
        self.predictor = predictor
        self.cfg = cfg
        self.device = device
        self.reward_mode = reward_mode
        self.risk_lambda = risk_lambda
        self.sampler = sampler or default_role_build_sampler
        self.optimizer = optimizer
        self.champion_ids = champion_ids
        self.rng = rng or np.random.default_rng()
        self._net_cache: dict[bytes, tuple[np.ndarray, float]] = {}
        self._term_cache: dict[bytes, dict[Side, float]] = {}

    # ---- public API ---------------------------------------------------

    def run(self, root_state: DraftState) -> tuple[_Node, np.ndarray]:
        """Run ``simulations`` rollouts from ``root_state``.

        Returns the root node and a full-width visit distribution of
        length ``n_champions`` (zeros for actions outside the beam).
        """
        root = _Node(
            side=root_state.current_side(),
            is_terminal=root_state.is_terminal(),
        )
        if not root.is_terminal:
            self._expand(root, root_state, add_noise=True)
        for _ in range(self.cfg.simulations):
            self._simulate(root, root_state.clone())
        visits = np.zeros(root_state.n_champions, dtype=np.float64)
        if root.actions is not None and root.N is not None:
            visits[root.actions] = root.N.astype(np.float64)
        return root, visits

    # ---- core search --------------------------------------------------

    def _simulate(self, root: _Node, state: DraftState) -> None:
        path: list[tuple[_Node, int]] = []  # (node, beam_index)
        node = root
        while True:
            if node.is_terminal:
                rewards = self.terminal_rewards(state)
                self._backup(path, rewards_terminal=rewards)
                return
            beam_idx = self._select(node)
            action = int(node.actions[beam_idx])  # type: ignore[index]
            path.append((node, beam_idx))
            child = node.children.get(action)
            if child is None:
                state.apply(action)
                child = _Node(
                    side=state.current_side(),
                    is_terminal=state.is_terminal(),
                )
                node.children[action] = child
                if child.is_terminal:
                    rewards = self.terminal_rewards(state)
                    self._backup(path, rewards_terminal=rewards)
                else:
                    value_for_child_side = self._expand(child, state, add_noise=False)
                    self._backup(
                        path, leaf_side=child.side, leaf_value=value_for_child_side
                    )
                return
            state.apply(action)
            node = child

    def _select(self, node: _Node) -> int:
        assert node.actions is not None and node.priors is not None
        assert node.N is not None and node.W is not None
        sqrt_total = math.sqrt(max(1, node.N_sum))
        q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
        u = self.cfg.c_puct * node.priors * sqrt_total / (1.0 + node.N)
        return int(np.argmax(q + u))

    def _backup(
        self,
        path: list[tuple[_Node, int]],
        *,
        rewards_terminal: dict[Side, float] | None = None,
        leaf_side: Side | None = None,
        leaf_value: float | None = None,
    ) -> None:
        for parent, beam_idx in path:
            if rewards_terminal is not None:
                v = rewards_terminal[
                    parent.side
                ]  # parent.side is the actor at this edge
            else:
                assert leaf_side is not None and leaf_value is not None
                v = leaf_value if parent.side == leaf_side else -leaf_value
            assert parent.N is not None and parent.W is not None
            parent.N[beam_idx] += 1
            parent.W[beam_idx] += v
            parent.N_sum += 1

    # ---- expansion / leaf evaluation ---------------------------------

    def _expand(self, node: _Node, state: DraftState, *, add_noise: bool) -> float:
        """Fill node's beam and return the net's value from acting side's perspective."""
        priors_full, value = self._net_eval(state)
        legal_actions = np.flatnonzero(state.available)
        if legal_actions.size == 0:
            raise RuntimeError("No legal actions during MCTS expansion.")
        k = min(self.cfg.beam_width, legal_actions.size)
        legal_priors = priors_full[legal_actions]
        top_idx = np.argpartition(-legal_priors, k - 1)[:k]
        actions = legal_actions[top_idx]
        beam_priors = legal_priors[top_idx].astype(np.float32)
        s = beam_priors.sum()
        beam_priors = beam_priors / s if s > 0 else np.full_like(beam_priors, 1.0 / k)
        if add_noise and self.cfg.dirichlet_eps > 0 and k > 1:
            noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * k).astype(
                np.float32
            )
            beam_priors = (
                1 - self.cfg.dirichlet_eps
            ) * beam_priors + self.cfg.dirichlet_eps * noise
        node.actions = actions.astype(np.int64)
        node.priors = beam_priors
        node.N = np.zeros(k, dtype=np.int32)
        node.W = np.zeros(k, dtype=np.float32)
        return value

    # ---- caches -------------------------------------------------------

    def _state_key(self, state: DraftState) -> bytes:
        # Canonical: ban multisets are order-invariant; pick order also doesn't
        # affect the observation encoding (multi-hot). Step idx fully
        # determines which side moves next.
        return state.available.tobytes() + state.step_idx.to_bytes(2, "little")

    def _net_eval(self, state: DraftState) -> tuple[np.ndarray, float]:
        key = self._state_key(state)
        cached = self._net_cache.get(key)
        if cached is not None:
            return cached
        feat = encode_obs(state.to_obs(), state.n_champions)
        with torch.no_grad():
            x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
            logits, value = self.net(x)
            mask = torch.from_numpy(state.legal_mask()).unsqueeze(0).to(self.device)
            masked_logits = logits.masked_fill(~mask, -1e9)
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0).cpu().numpy()
            v = float(value.squeeze(0).cpu().item())
        out = (probs, v)
        self._net_cache[key] = out
        return out

    def terminal_rewards(self, state: DraftState) -> dict[Side, float]:
        # Cache per terminal team composition (order-invariant).
        bp = tuple(sorted(state.blue_picks))
        rp = tuple(sorted(state.red_picks))
        key = repr((bp, rp)).encode()
        cached = self._term_cache.get(key)
        if cached is not None:
            return cached
        ids = self.champion_ids
        blue_team = (
            [int(ids[i]) for i in state.blue_picks] if ids else list(state.blue_picks)
        )
        red_team = (
            [int(ids[i]) for i in state.red_picks] if ids else list(state.red_picks)
        )
        if self.optimizer is not None:
            result = self.optimizer(
                blue_team, red_team, self.predictor, self.reward_mode
            )
        else:
            result = resolve_rewards(
                blue_team,
                red_team,
                self.predictor,
                self.sampler,
                self.reward_mode,
                self.risk_lambda,
            )
        rewards = {
            Side.BLUE: float(result.blue_reward),
            Side.RED: float(result.red_reward),
        }
        self._term_cache[key] = rewards
        return rewards


def visit_policy(
    visits: np.ndarray, legal: np.ndarray, temperature: float
) -> np.ndarray:
    """Convert MCTS visit counts to a normalised policy target (legal-only)."""
    out = np.zeros_like(visits, dtype=np.float32)
    legal_idx = np.flatnonzero(legal)
    v = visits[legal_idx]
    if v.sum() <= 0:
        out[legal_idx] = 1.0 / len(legal_idx)
        return out
    if temperature <= 1e-6:
        best = legal_idx[int(np.argmax(v))]
        out[best] = 1.0
        return out
    scaled = np.power(v, 1.0 / temperature)
    out[legal_idx] = scaled / scaled.sum()
    return out
