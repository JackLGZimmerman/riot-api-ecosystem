# pyright: reportPrivateImportUsage=false

"""Match-Outcome Hypergraph Neural Network (HGNN) win-rate model.

Predicts ``P(blue wins)`` from the cached interaction priors. The keystone is a
single message function (`HypergraphConvLayer`) that, for any hyperedge, splits
its members into *allies* and *enemies relative to the receiving node* via
`TEAM_OF`. 2vX and 1v1 reuse the same mechanism; adding 3vX/2v1 later is a single
registry entry in `EDGE_TYPES`.

The Bayesian feature bridge (`posterior_mean_var`): the cache stores the smoothed
posterior mean ``mu`` and a support count; the Beta-Binomial posterior variance
``mu(1-mu)/(count+strength+1)`` gives the per-edge ``sigma^2`` that the confidence
gate (`PhiEncoder`, design §4) uses to throttle sparse edges. This is the same
hierarchical shrinkage as `app/core/utils/smoothing.py`.

Identity embeddings initialise each player node, while explicit relationship
residual channels give the edge encoders the same joint-minus-generic-1vX signal
that the structured model consumed directly. 3vX/2v1 edges remain a cache/schema
extension rather than a message-passing change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

N_PLAYERS = 10
N_RELATIONSHIP_EDGES = 45
TEAM_OF: tuple[int, ...] = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
TEAM_PAIRS: tuple[tuple[int, int], ...] = tuple(combinations(range(5), 2))  # C(5,2)=10
LOGIT_EPS = 1e-6


@dataclass(frozen=True)
class EdgeType:
    """One hyperedge type = canonical member-index list + per-slot stat orientation.

    `member_idx` lists, per edge, the node indices of its ordered members. `orient`
    is +1 if the stored stat is already from that member's perspective, -1 if it
    must be flipped (in logit space a flip is a sign change, since
    ``logit(1-p) = -logit(p)``). The stat is stored from team A's (blue) side, so a
    cross-team member on team B reads the flipped value.
    """

    name: str
    feat_key: str  # mu/var key prefix in the input dict
    member_idx: tuple[tuple[int, ...], ...]
    orient: tuple[tuple[float, ...], ...]


# 2vX: same-team pairs, blue C(5,2) then red C(5,2); stat is each pair's own team
# win rate, so no orientation flip. Order matches synergy_2vx (build_dataset._TEAM_PAIRS).
_TWOVX_MEMBERS = tuple((a, b) for a, b in TEAM_PAIRS) + tuple((a + 5, b + 5) for a, b in TEAM_PAIRS)
# 1v1: product(blue, red), slot = blue*5+red; stat is blue-perspective, so the red
# member (slot 1) reads the flipped value.
_ONEV1_MEMBERS = tuple((b, r + 5) for b in range(5) for r in range(5))

EDGE_TYPES: dict[str, EdgeType] = {
    "twovx": EdgeType(
        name="twovx",
        feat_key="2vx",
        member_idx=_TWOVX_MEMBERS,
        orient=tuple((1.0, 1.0) for _ in _TWOVX_MEMBERS),
    ),
    "onev1": EdgeType(
        name="onev1",
        feat_key="1v1",
        member_idx=_ONEV1_MEMBERS,
        orient=tuple((1.0, -1.0) for _ in _ONEV1_MEMBERS),
    ),
}
INTRA_TYPES: tuple[str, ...] = ("twovx",)  # same-team (synergy)
CROSS_TYPES: tuple[str, ...] = ("onev1",)  # cross-team (matchup)


@dataclass(frozen=True)
class HGNNConfig:
    n_champions: int = 951  # embedding rows for champion ids (raw id; size from cache meta)
    n_builds: int = 11  # embedding rows for build ids
    build_vocab: tuple[str, ...] = ()  # build label -> embedding index (for the predictor)
    node_dim: int = 96
    msg_dim: int = 96
    edge_hidden: int = 64  # phi output width
    value_hidden: tuple[int, ...] = (64,)
    gate_hidden: tuple[int, ...] = (32,)
    message_hidden: tuple[int, ...] = (128,)
    update_hidden: tuple[int, ...] = (96,)
    node_init_hidden: tuple[int, ...] = (96,)
    readout_hidden: tuple[int, ...] = (256,)
    residual_head_hidden: tuple[int, ...] = (128,)
    prior_shortcut_hidden: tuple[int, ...] = ()
    n_intra: int = 2
    n_cross: int = 2
    dropout: float = 0.1
    logit_clip: float | None = 5.0
    use_count_features: bool = True
    use_edge_residual_features: bool = True
    use_residual_head: bool = True
    use_prior_shortcut: bool = True


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _logit(prob: torch.Tensor, clip: float | None = None) -> torch.Tensor:
    p = prob.clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)
    out = torch.log(p / (1.0 - p))
    if clip is not None and clip > 0.0:
        return out.clamp(-float(clip), float(clip))
    return out


def _clip_logit(value: torch.Tensor, clip: float | None = None) -> torch.Tensor:
    if clip is not None and clip > 0.0:
        return value.clamp(-float(clip), float(clip))
    return value


def relationship_logit_features(
    *,
    mu_1vx: torch.Tensor,
    mu_2vx: torch.Tensor,
    mu_1v1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Explicit relationship residuals over the generic 1vX expectation.

    The structured model previously handed the network the important residual:
    joint relationship logit minus the logit of the same generic 1vX baseline
    used by cache smoothing (`build_dataset._composite_interaction_priors`).
    Keep the same idea in the HGNN tensor contract.
    """
    blue = mu_1vx[:, :5]
    red = mu_1vx[:, 5:]

    joint_1v1 = _logit(mu_1v1)
    expected_prob_1v1 = (0.5 + (blue[:, :, None] - red[:, None, :]) / 2.0).reshape(-1, 25)
    expected_1v1 = _logit(expected_prob_1v1)

    expected_2vx_parts: list[torch.Tensor] = []
    for offset in (0, 5):
        side = mu_1vx[:, offset : offset + 5]
        for a_idx, b_idx in TEAM_PAIRS:
            expected_2vx_parts.append(_logit(0.5 * (side[:, a_idx] + side[:, b_idx])))
    expected_2vx = torch.stack(expected_2vx_parts, dim=1)
    joint_2vx = _logit(mu_2vx)

    return {
        "joint_logit_1v1": joint_1v1,
        "expected_logit_1v1": expected_1v1,
        "delta_logit_1v1": joint_1v1 - expected_1v1,
        "joint_logit_2vx": joint_2vx,
        "expected_logit_2vx": expected_2vx,
        "delta_logit_2vx": joint_2vx - expected_2vx,
    }


def posterior_mean_var(
    rate: torch.Tensor,
    count: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Beta-Binomial posterior mean and variance for a cached smoothed rate.

    `rate` is the shrunk posterior mean (already smoothed upstream). Treating
    ``count + strength`` as the effective sample size, the posterior variance is
    ``mu(1-mu)/(count+strength+1)`` — small for well-observed edges (gate opens),
    large for barely-observed ones (gate suppresses).
    """
    mu = rate.clamp(0.0, 1.0)
    n_eff = count.clamp_min(0.0) + float(strength)
    var = mu * (1.0 - mu) / (n_eff + 1.0)
    return mu, var


def support_features(
    count: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit support features for HGNN edge encoders.

    Posterior variance has a tiny numeric range for this dataset, so the model also
    sees a direct confidence score, log support, and a missing flag.
    """
    count_f = count.clamp_min(0.0)
    denom = count_f + float(strength)
    confidence = torch.where(denom > 0.0, count_f / denom.clamp_min(1.0e-12), torch.zeros_like(count_f))
    log_count = torch.log1p(count_f)
    missing = (count_f <= 0.0).to(count.dtype)
    return confidence, log_count, missing


def build_hgnn_inputs(
    *,
    champion_id: Any,
    build_id: Any,
    win_rate: Any,
    matchup_1v1: Any,
    synergy_2vx: Any,
    p1_cnt: Any,
    m1v1_cnt: Any,
    s2vx_cnt: Any,
    m1v1_eff_n: Any = None,
    s2vx_eff_n: Any = None,
    strength: float,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Turn raw cache/prior arrays into the model's node/edge tensors.

    Single source of truth shared by training and the runtime predictor. Accepts
    numpy arrays or tensors. mu values stay in probability space (the model maps
    them to logits internally); champion/build ids become long embedding indices.

    The interaction posterior variance uses the *effective* sample size from
    nested pooling (`m1v1_eff_n` / `s2vx_eff_n`) when provided, so an edge that is
    sparse at the build level but well-supported by a denser parent opens the
    φ-gate instead of being suppressed. The explicit support features
    (confidence / log_count / missing) keep using the raw build-level counts.
    """

    def to_tensor(arr: Any) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def to_long(arr: Any) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.long, device=device)

    count_1vx = to_tensor(p1_cnt)
    count_2vx = to_tensor(s2vx_cnt)
    count_1v1 = to_tensor(m1v1_cnt)
    eff_n_2vx = count_2vx if s2vx_eff_n is None else to_tensor(s2vx_eff_n)
    eff_n_1v1 = count_1v1 if m1v1_eff_n is None else to_tensor(m1v1_eff_n)
    mu_1vx, var_1vx = posterior_mean_var(to_tensor(win_rate), count_1vx, strength)
    mu_2vx, var_2vx = posterior_mean_var(to_tensor(synergy_2vx), eff_n_2vx, strength)
    mu_1v1, var_1v1 = posterior_mean_var(to_tensor(matchup_1v1), eff_n_1v1, strength)
    conf_1vx, log_count_1vx, missing_1vx = support_features(count_1vx, strength)
    conf_2vx, log_count_2vx, missing_2vx = support_features(count_2vx, strength)
    conf_1v1, log_count_1v1, missing_1v1 = support_features(count_1v1, strength)
    inputs = {
        "champion_id": to_long(champion_id),
        "build_id": to_long(build_id),
        "mu_1vx": mu_1vx,
        "var_1vx": var_1vx,
        "conf_1vx": conf_1vx,
        "log_count_1vx": log_count_1vx,
        "missing_1vx": missing_1vx,
        "mu_2vx": mu_2vx,
        "var_2vx": var_2vx,
        "conf_2vx": conf_2vx,
        "log_count_2vx": log_count_2vx,
        "missing_2vx": missing_2vx,
        "mu_1v1": mu_1v1,
        "var_1v1": var_1v1,
        "conf_1v1": conf_1v1,
        "log_count_1v1": log_count_1v1,
        "missing_1v1": missing_1v1,
    }
    inputs.update(relationship_logit_features(mu_1vx=mu_1vx, mu_2vx=mu_2vx, mu_1v1=mu_1v1))
    return inputs


def swap_hgnn_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Team-swap augmentation (design §8/§9): swap blue<->red, return mirrored inputs.

    Used with a flipped label so the learned function is approximately antisymmetric
    (``P(A beats B) = 1 - P(B beats A)``) and the training data is effectively doubled.
    """

    def swap_halves(x: torch.Tensor, half: int) -> torch.Tensor:
        return torch.cat([x[:, half:], x[:, :half]], dim=1)

    def swap_1v1_mu(x: torch.Tensor) -> torch.Tensor:
        # new[newblue=j, newred=i] = P(oldred_j beats oldblue_i) = 1 - old[i,j]
        return (1.0 - x.reshape(-1, 5, 5)).transpose(1, 2).reshape(-1, 25)

    def swap_1v1_var(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, 5, 5).transpose(1, 2).reshape(-1, 25)

    def swap_1v1_signed(x: torch.Tensor) -> torch.Tensor:
        return -x.reshape(-1, 5, 5).transpose(1, 2).reshape(-1, 25)

    swapped = {
        "champion_id": swap_halves(inputs["champion_id"], 5),
        "build_id": swap_halves(inputs["build_id"], 5),
        "mu_1vx": swap_halves(inputs["mu_1vx"], 5),
        "var_1vx": swap_halves(inputs["var_1vx"], 5),
        "mu_2vx": swap_halves(inputs["mu_2vx"], 10),
        "var_2vx": swap_halves(inputs["var_2vx"], 10),
        "mu_1v1": swap_1v1_mu(inputs["mu_1v1"]),
        "var_1v1": swap_1v1_var(inputs["var_1v1"]),
    }
    for suffix, half in (("1vx", 5), ("2vx", 10)):
        for prefix in ("conf", "log_count", "missing"):
            key = f"{prefix}_{suffix}"
            if key in inputs:
                swapped[key] = swap_halves(inputs[key], half)
    for prefix in ("conf", "log_count", "missing"):
        key = f"{prefix}_1v1"
        if key in inputs:
            swapped[key] = swap_1v1_var(inputs[key])
    for prefix in ("joint_logit", "expected_logit", "delta_logit"):
        key_2vx = f"{prefix}_2vx"
        if key_2vx in inputs:
            swapped[key_2vx] = swap_halves(inputs[key_2vx], 10)
        key_1v1 = f"{prefix}_1v1"
        if key_1v1 in inputs:
            swapped[key_1v1] = swap_1v1_signed(inputs[key_1v1])
    return swapped


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int, *, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for hidden_dim in hidden:
        layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


class PhiEncoder(nn.Module):
    """Uncertainty-gated statistic encoder (design §4).

    Node priors use ``[logit(mu), sigma^2]`` plus optional support features.
    Relationship edges can additionally use ``[joint, expected, joint-expected]``
    so the model sees residual interaction signal directly.
    """

    def __init__(self, config: HGNNConfig, *, use_residual_features: bool = False) -> None:
        super().__init__()
        self.use_count_features = config.use_count_features
        self.use_residual_features = use_residual_features
        if self.use_residual_features:
            value_dim = 7 if self.use_count_features else 4
        else:
            value_dim = 5 if self.use_count_features else 2
        gate_dim = 4 if self.use_count_features else 1
        self.value = _mlp(value_dim, config.value_hidden, config.edge_hidden, dropout=config.dropout)
        self.gate = _mlp(gate_dim, config.gate_hidden, config.edge_hidden, dropout=config.dropout)

    def forward(
        self,
        mu_logit: torch.Tensor,
        var: torch.Tensor,
        confidence: torch.Tensor | None = None,
        log_count: torch.Tensor | None = None,
        missing: torch.Tensor | None = None,
        expected_logit: torch.Tensor | None = None,
        delta_logit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        precision = 1.0 / (1.0 + var)
        if self.use_residual_features:
            if expected_logit is None or delta_logit is None:
                raise ValueError("PhiEncoder residual features require expected_logit/delta_logit tensors")
            value_parts = [mu_logit, expected_logit, delta_logit, var]
        else:
            value_parts = [mu_logit, var]
        if self.use_count_features:
            if confidence is None or log_count is None or missing is None:
                raise ValueError("PhiEncoder count features require confidence/log_count/missing tensors")
            value_input = torch.stack([*value_parts, confidence, log_count, missing], dim=-1)
            gate_input = torch.stack([precision, confidence, log_count, missing], dim=-1)
        else:
            value_input = torch.stack(value_parts, dim=-1)
            gate_input = precision.unsqueeze(-1)
        value = self.value(value_input)
        gate = torch.sigmoid(self.gate(gate_input))
        return gate * value


class IdentityEncoder(nn.Module):
    """Multiplicative (champion, role, build) node initialisation (design §3).

    role and build modulate the champion via ``(1 + W·e)``, so an untrained/rare
    modifier defaults to a near-identity transform of the champion embedding — the
    architectural mirror of the Bayesian shrinkage ``θ(c,r,b) → θ(c,r) → θ(c)``.
    The last embedding row of champion/build is reserved for unknown ids.
    """

    def __init__(self, n_champions: int, n_builds: int, dim: int) -> None:
        super().__init__()
        self.champion = nn.Embedding(n_champions + 1, dim)
        self.role = nn.Embedding(5, dim)
        self.build = nn.Embedding(n_builds + 1, dim)
        self.w_role = nn.Linear(dim, dim, bias=False)
        self.w_build = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer(
            "role_idx",
            torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long),
            persistent=False,
        )

    def forward(self, champion_id: torch.Tensor, build_id: torch.Tensor) -> torch.Tensor:
        e_c = self.champion(champion_id)  # [B, 10, dim]
        e_r = self.role(cast(torch.Tensor, self.role_idx)).unsqueeze(0)  # [1, 10, dim]
        e_b = self.build(build_id)  # [B, 10, dim]
        return self.norm(e_c * (1.0 + self.w_role(e_r)) * (1.0 + self.w_build(e_b)))


class AttnPool(nn.Module):
    """Learned-query attention pool over a team's 5 nodes (design §8)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.scale = dim**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, 5, d]
        scores = (x @ self.query) / self.scale
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class HypergraphConvLayer(nn.Module):
    """One round of the unified hyperedge message + residual node update (§5, §6).

    The masks/indices are passed in (registered once on the model) so layers stay
    pure weight-holders and a new edge type needs no layer change.
    """

    def __init__(self, config: HGNNConfig) -> None:
        super().__init__()
        d = config.node_dim
        msg_in = 3 * d + 2 + config.edge_hidden  # h_t, ally pool, enemy pool, 2 flags, phi
        self.message = _mlp(msg_in, config.message_hidden, config.msg_dim, dropout=config.dropout)
        self.update = _mlp(d + config.msg_dim, config.update_hidden, d, dropout=config.dropout)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        h: torch.Tensor,  # [B, 10, d]
        phi: torch.Tensor,  # [B, E, M, edge_hidden] (oriented per member)
        member_idx: torch.Tensor,  # [E, M] long
        ally_mask: torch.Tensor,  # [E, M, M] float
        enemy_mask: torch.Tensor,  # [E, M, M] float
        deg: torch.Tensor,  # [10] float
    ) -> torch.Tensor:
        b, _, d = h.shape
        h_mem = h[:, member_idx]  # [B, E, M, d]
        ally_count = ally_mask.sum(-1).clamp_min(1.0)  # [E, M]
        enemy_count = enemy_mask.sum(-1).clamp_min(1.0)
        ally_pool = torch.einsum("ets,besd->betd", ally_mask, h_mem) / ally_count[None, :, :, None]
        enemy_pool = torch.einsum("ets,besd->betd", enemy_mask, h_mem) / enemy_count[None, :, :, None]
        ally_flag = (ally_mask.sum(-1) > 0).to(h.dtype)[None, :, :, None].expand(b, -1, -1, 1)
        enemy_flag = (enemy_mask.sum(-1) > 0).to(h.dtype)[None, :, :, None].expand(b, -1, -1, 1)
        msg = self.message(
            torch.cat([h_mem, ally_pool, enemy_pool, ally_flag, enemy_flag, phi], dim=-1)
        )  # [B, E, M, msg_dim]

        node_acc = h.new_zeros(b, N_PLAYERS, msg.shape[-1])
        for slot in range(member_idx.shape[1]):
            node_acc.index_add_(1, member_idx[:, slot], msg[:, :, slot, :])
        node_msg = node_acc / deg.clamp_min(1.0).view(1, N_PLAYERS, 1)
        return self.norm(h + self.update(torch.cat([h, node_msg], dim=-1)))


class HGNNWinModel(nn.Module):
    def __init__(self, config: HGNNConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        if config is not None and overrides:
            raise ValueError("Pass either config or keyword overrides, not both")
        self.config = config or HGNNConfig(**overrides)
        c = self.config

        self.phi = nn.ModuleDict(
            {
                "1vx": PhiEncoder(c, use_residual_features=False),
                "twovx": PhiEncoder(c, use_residual_features=c.use_edge_residual_features),
                "onev1": PhiEncoder(c, use_residual_features=c.use_edge_residual_features),
            }
        )
        self.identity = IdentityEncoder(c.n_champions, c.n_builds, c.node_dim)
        # Node init fuses the multiplicative identity with the 1vX posterior (§3).
        self.node_init = _mlp(
            c.node_dim + c.edge_hidden, c.node_init_hidden, c.node_dim, dropout=c.dropout
        )
        self.node_norm = nn.LayerNorm(c.node_dim)
        self.intra_layers = nn.ModuleList(HypergraphConvLayer(c) for _ in range(c.n_intra))
        self.cross_layers = nn.ModuleList(HypergraphConvLayer(c) for _ in range(c.n_cross))

        self.attn_pool = AttnPool(c.node_dim)
        self.team_proj = nn.Linear(c.node_dim * 3, c.node_dim)
        self.residual_head = (
            _mlp(
                N_RELATIONSHIP_EDGES * 4,
                c.residual_head_hidden,
                c.node_dim,
                dropout=c.dropout,
            )
            if c.use_residual_head
            else None
        )
        self.prior_shortcut = (
            _mlp(
                15 + N_RELATIONSHIP_EDGES * 2,
                c.prior_shortcut_hidden,
                1,
                dropout=0.0,
            )
            if c.use_prior_shortcut
            else None
        )
        head_in = c.node_dim * (5 if c.use_residual_head else 4)
        self.head = _mlp(head_in, c.readout_hidden, 1, dropout=c.dropout)

        self._register_edge_buffers()

    def _register_edge_buffers(self) -> None:
        team_of = torch.as_tensor(TEAM_OF, dtype=torch.long)
        for key, edge in EDGE_TYPES.items():
            member_idx = torch.as_tensor(edge.member_idx, dtype=torch.long)  # [E, M]
            orient = torch.as_tensor(edge.orient, dtype=torch.float32)  # [E, M]
            teams = team_of[member_idx]  # [E, M]
            same = teams[:, :, None] == teams[:, None, :]  # [E, M, M]
            not_self = ~torch.eye(member_idx.shape[1], dtype=torch.bool).unsqueeze(0)
            ally_mask = (same & not_self).to(torch.float32)
            enemy_mask = ((~same) & not_self).to(torch.float32)
            deg = torch.zeros(N_PLAYERS)
            deg.index_add_(0, member_idx.reshape(-1), torch.ones(member_idx.numel()))
            self.register_buffer(f"{key}_member_idx", member_idx, persistent=False)
            self.register_buffer(f"{key}_orient", orient, persistent=False)
            self.register_buffer(f"{key}_ally_mask", ally_mask, persistent=False)
            self.register_buffer(f"{key}_enemy_mask", enemy_mask, persistent=False)
            self.register_buffer(f"{key}_deg", deg, persistent=False)

    def _edge_phi(
        self,
        key: str,
        mu: torch.Tensor,
        var: torch.Tensor,
        joint_logit: torch.Tensor | None = None,
        expected_logit: torch.Tensor | None = None,
        delta_logit: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        log_count: torch.Tensor | None = None,
        missing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Oriented per-member phi for an edge type: [B, E, M, edge_hidden]."""
        orient = cast(torch.Tensor, getattr(self, f"{key}_orient"))  # [E, M]
        base_logit = _logit(mu, self.config.logit_clip) if joint_logit is None else _clip_logit(joint_logit, self.config.logit_clip)
        mu_logit = base_logit[:, :, None] * orient[None]  # [B, E, M]
        var_m = var[:, :, None].expand(-1, -1, orient.shape[1])  # [B, E, M]
        expected_m = (
            _clip_logit(expected_logit, self.config.logit_clip)[:, :, None] * orient[None]
            if expected_logit is not None
            else None
        )
        delta_m = (
            _clip_logit(delta_logit, self.config.logit_clip)[:, :, None] * orient[None]
            if delta_logit is not None
            else None
        )
        confidence_m = (
            confidence[:, :, None].expand(-1, -1, orient.shape[1])
            if confidence is not None
            else None
        )
        log_count_m = (
            log_count[:, :, None].expand(-1, -1, orient.shape[1])
            if log_count is not None
            else None
        )
        missing_m = (
            missing[:, :, None].expand(-1, -1, orient.shape[1])
            if missing is not None
            else None
        )
        return self.phi[key](
            mu_logit,
            var_m,
            confidence_m,
            log_count_m,
            missing_m,
            expected_logit=expected_m,
            delta_logit=delta_m,
        )

    def _run_block(
        self, h: torch.Tensor, layers: nn.ModuleList, key: str, phi: torch.Tensor
    ) -> torch.Tensor:
        member_idx = cast(torch.Tensor, getattr(self, f"{key}_member_idx"))
        ally_mask = cast(torch.Tensor, getattr(self, f"{key}_ally_mask"))
        enemy_mask = cast(torch.Tensor, getattr(self, f"{key}_enemy_mask"))
        deg = cast(torch.Tensor, getattr(self, f"{key}_deg"))
        for layer in layers:
            h = layer(h, phi, member_idx, ally_mask, enemy_mask, deg)
        return h

    def _readout(self, team: torch.Tensor) -> torch.Tensor:  # team: [B, 5, d]
        pooled = torch.cat([team.mean(dim=1), team.max(dim=1).values, self.attn_pool(team)], dim=-1)
        return self.team_proj(pooled)

    def _residual_readout(
        self,
        *,
        delta_logit_2vx: torch.Tensor,
        delta_logit_1v1: torch.Tensor,
        conf_2vx: torch.Tensor | None,
        conf_1v1: torch.Tensor | None,
        missing_2vx: torch.Tensor | None,
        missing_1v1: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.residual_head is None:
            raise RuntimeError("residual_head is disabled")
        signed_2vx = torch.cat([delta_logit_2vx[:, :10], -delta_logit_2vx[:, 10:]], dim=1)
        delta = _clip_logit(torch.cat([delta_logit_1v1, signed_2vx], dim=1), self.config.logit_clip)
        if conf_1v1 is None or conf_2vx is None:
            confidence = torch.ones_like(delta)
        else:
            confidence = torch.cat([conf_1v1, conf_2vx], dim=1)
        if missing_1v1 is None or missing_2vx is None:
            missing = torch.zeros_like(delta)
        else:
            missing = torch.cat([missing_1v1, missing_2vx], dim=1)
        residual_input = torch.cat([delta, confidence, delta * confidence, missing], dim=1)
        return self.residual_head(residual_input)

    def _prior_shortcut_logit(
        self,
        *,
        mu_1vx: torch.Tensor,
        delta_logit_2vx: torch.Tensor,
        delta_logit_1v1: torch.Tensor,
        conf_2vx: torch.Tensor | None,
        conf_1v1: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.prior_shortcut is None:
            raise RuntimeError("prior_shortcut is disabled")
        identity = _logit(mu_1vx, self.config.logit_clip)
        blue = identity[:, :5]
        red = identity[:, 5:]
        base = torch.cat([blue, red, blue - red], dim=1)
        signed_2vx = torch.cat([delta_logit_2vx[:, :10], -delta_logit_2vx[:, 10:]], dim=1)
        delta = _clip_logit(torch.cat([delta_logit_1v1, signed_2vx], dim=1), self.config.logit_clip)
        if conf_1v1 is None or conf_2vx is None:
            confidence = torch.ones_like(delta)
        else:
            confidence = torch.cat([conf_1v1, conf_2vx], dim=1)
        shortcut_input = torch.cat([base, delta, delta * confidence], dim=1)
        return self.prior_shortcut(shortcut_input).squeeze(-1)

    def _forward_impl(
        self,
        *,
        champion_id: torch.Tensor,
        build_id: torch.Tensor,
        mu_1vx: torch.Tensor,
        var_1vx: torch.Tensor,
        mu_2vx: torch.Tensor,
        var_2vx: torch.Tensor,
        mu_1v1: torch.Tensor,
        var_1v1: torch.Tensor,
        joint_logit_2vx: torch.Tensor | None = None,
        expected_logit_2vx: torch.Tensor | None = None,
        delta_logit_2vx: torch.Tensor | None = None,
        joint_logit_1v1: torch.Tensor | None = None,
        expected_logit_1v1: torch.Tensor | None = None,
        delta_logit_1v1: torch.Tensor | None = None,
        conf_1vx: torch.Tensor | None = None,
        log_count_1vx: torch.Tensor | None = None,
        missing_1vx: torch.Tensor | None = None,
        conf_2vx: torch.Tensor | None = None,
        log_count_2vx: torch.Tensor | None = None,
        missing_2vx: torch.Tensor | None = None,
        conf_1v1: torch.Tensor | None = None,
        log_count_1v1: torch.Tensor | None = None,
        missing_1v1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.config.use_edge_residual_features or self.config.use_residual_head:
            if (
                joint_logit_2vx is None
                or expected_logit_2vx is None
                or delta_logit_2vx is None
                or joint_logit_1v1 is None
                or expected_logit_1v1 is None
                or delta_logit_1v1 is None
            ):
                features = relationship_logit_features(mu_1vx=mu_1vx, mu_2vx=mu_2vx, mu_1v1=mu_1v1)
                joint_logit_2vx = features["joint_logit_2vx"]
                expected_logit_2vx = features["expected_logit_2vx"]
                delta_logit_2vx = features["delta_logit_2vx"]
                joint_logit_1v1 = features["joint_logit_1v1"]
                expected_logit_1v1 = features["expected_logit_1v1"]
                delta_logit_1v1 = features["delta_logit_1v1"]

        # Node init: multiplicative identity (§3) fused with the 1vX posterior.
        h0 = self.identity(champion_id, build_id)
        phi_node = self.phi["1vx"](
            _logit(mu_1vx, self.config.logit_clip),
            var_1vx,
            conf_1vx,
            log_count_1vx,
            missing_1vx,
        )
        h = self.node_norm(self.node_init(torch.cat([h0, phi_node], dim=-1)))

        phi_2vx = self._edge_phi(
            "twovx",
            mu_2vx,
            var_2vx,
            joint_logit_2vx,
            expected_logit_2vx,
            delta_logit_2vx,
            conf_2vx,
            log_count_2vx,
            missing_2vx,
        )
        for key in INTRA_TYPES:
            h = self._run_block(h, self.intra_layers, key, phi_2vx)

        phi_1v1 = self._edge_phi(
            "onev1",
            mu_1v1,
            var_1v1,
            joint_logit_1v1,
            expected_logit_1v1,
            delta_logit_1v1,
            conf_1v1,
            log_count_1v1,
            missing_1v1,
        )
        for key in CROSS_TYPES:
            h = self._run_block(h, self.cross_layers, key, phi_1v1)

        a = self._readout(h[:, :5])
        b = self._readout(h[:, 5:])
        head_parts = [a, b, a - b, a * b]
        if self.residual_head is not None:
            if delta_logit_2vx is None or delta_logit_1v1 is None:
                raise ValueError("Residual head requires relationship delta logits")
            head_parts.append(
                self._residual_readout(
                    delta_logit_2vx=delta_logit_2vx,
                    delta_logit_1v1=delta_logit_1v1,
                    conf_2vx=conf_2vx,
                    conf_1v1=conf_1v1,
                    missing_2vx=missing_2vx,
                    missing_1v1=missing_1v1,
                )
            )
        logit = self.head(torch.cat(head_parts, dim=-1)).squeeze(-1)
        if self.prior_shortcut is not None:
            if delta_logit_2vx is None or delta_logit_1v1 is None:
                raise ValueError("Prior shortcut requires relationship delta logits")
            logit = logit + self._prior_shortcut_logit(
                mu_1vx=mu_1vx,
                delta_logit_2vx=delta_logit_2vx,
                delta_logit_1v1=delta_logit_1v1,
                conf_2vx=conf_2vx,
                conf_1v1=conf_1v1,
            )
        return {"final_logit": logit}

    def forward(
        self,
        *,
        champion_id: torch.Tensor,
        build_id: torch.Tensor,
        mu_1vx: torch.Tensor,
        var_1vx: torch.Tensor,
        mu_2vx: torch.Tensor,
        var_2vx: torch.Tensor,
        mu_1v1: torch.Tensor,
        var_1v1: torch.Tensor,
        joint_logit_2vx: torch.Tensor | None = None,
        expected_logit_2vx: torch.Tensor | None = None,
        delta_logit_2vx: torch.Tensor | None = None,
        joint_logit_1v1: torch.Tensor | None = None,
        expected_logit_1v1: torch.Tensor | None = None,
        delta_logit_1v1: torch.Tensor | None = None,
        conf_1vx: torch.Tensor | None = None,
        log_count_1vx: torch.Tensor | None = None,
        missing_1vx: torch.Tensor | None = None,
        conf_2vx: torch.Tensor | None = None,
        log_count_2vx: torch.Tensor | None = None,
        missing_2vx: torch.Tensor | None = None,
        conf_1v1: torch.Tensor | None = None,
        log_count_1v1: torch.Tensor | None = None,
        missing_1v1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "champion_id": champion_id,
            "build_id": build_id,
            "mu_1vx": mu_1vx,
            "var_1vx": var_1vx,
            "mu_2vx": mu_2vx,
            "var_2vx": var_2vx,
            "mu_1v1": mu_1v1,
            "var_1v1": var_1v1,
        }
        optional = {
            "joint_logit_2vx": joint_logit_2vx,
            "expected_logit_2vx": expected_logit_2vx,
            "delta_logit_2vx": delta_logit_2vx,
            "joint_logit_1v1": joint_logit_1v1,
            "expected_logit_1v1": expected_logit_1v1,
            "delta_logit_1v1": delta_logit_1v1,
            "conf_1vx": conf_1vx,
            "log_count_1vx": log_count_1vx,
            "missing_1vx": missing_1vx,
            "conf_2vx": conf_2vx,
            "log_count_2vx": log_count_2vx,
            "missing_2vx": missing_2vx,
            "conf_1v1": conf_1v1,
            "log_count_1v1": log_count_1v1,
            "missing_1v1": missing_1v1,
        }
        inputs.update({key: value for key, value in optional.items() if value is not None})
        return self._forward_impl(**inputs)


def _config_from_payload(payload: dict[str, Any]) -> HGNNConfig:
    config_dict = dict(payload.get("model_config", {}))
    if "use_count_features" not in config_dict:
        # Legacy artifacts were trained with PhiEncoder([logit(mu), sigma^2]).
        config_dict["use_count_features"] = False
    if "use_edge_residual_features" not in config_dict:
        config_dict["use_edge_residual_features"] = False
    if "use_residual_head" not in config_dict:
        config_dict["use_residual_head"] = False
    if "use_prior_shortcut" not in config_dict:
        config_dict["use_prior_shortcut"] = False
    config_dict.pop("use_typed_residual_summary", None)
    config_dict.pop("enforce_antisymmetry", None)
    for key in (
        "build_vocab",
        "value_hidden",
        "gate_hidden",
        "message_hidden",
        "update_hidden",
        "node_init_hidden",
        "readout_hidden",
        "residual_head_hidden",
        "prior_shortcut_hidden",
    ):
        if key in config_dict:
            config_dict[key] = tuple(config_dict[key])
    return HGNNConfig(**config_dict)


def save_hgnn_model(path: Path, model: HGNNWinModel, *, confidence_strength: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "hgnn",
            "model_config": asdict(model.config),
            "confidence_strength": float(confidence_strength),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_hgnn_model(path: Path, *, device: str = "cpu") -> tuple[HGNNWinModel, HGNNConfig, float]:
    payload = cast(dict[str, Any], torch.load(path, map_location=device, weights_only=True))
    config = _config_from_payload(payload)
    model = HGNNWinModel(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    strength = float(payload.get("confidence_strength", 30.0))
    return model, config, strength


__all__ = [
    "EDGE_TYPES",
    "HGNNConfig",
    "HGNNWinModel",
    "IdentityEncoder",
    "TEAM_OF",
    "TEAM_PAIRS",
    "build_hgnn_inputs",
    "load_hgnn_model",
    "posterior_mean_var",
    "relationship_logit_features",
    "resolve_device",
    "save_hgnn_model",
    "support_features",
    "swap_hgnn_inputs",
]
