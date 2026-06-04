# pyright: reportPrivateImportUsage=false

"""Match-Outcome HGNN win-rate model.

Production currently uses identity embeddings on top of the 1vX player prior.
Direct 1v1/2vX integrations remain as explicit research capacity but are
disabled by default, and training/inference share one model shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from app.core.utils.common import TEAM_PAIRS as TEAM_PAIRS
from app.ml.semantic_group_features import SEMANTIC_GROUP_FEATURE_DIM

N_PLAYERS = 10
N_MATCHUPS_1V1 = 25
N_SYNERGIES_2VX = 20
N_RELATIONSHIP_EDGES = 45
LOGIT_EPS = 1e-6


@dataclass(frozen=True)
class HGNNConfig:
    n_champions: int = 951  # embedding rows for champion ids (raw id; size from cache meta)
    n_builds: int = 11  # embedding rows for build ids
    build_vocab: tuple[str, ...] = ()  # build label -> embedding index (for the predictor)
    node_dim: int = 96
    edge_hidden: int = 64  # phi output width
    value_hidden: tuple[int, ...] = (64,)
    gate_hidden: tuple[int, ...] = (32,)
    node_init_hidden: tuple[int, ...] = (96,)
    readout_hidden: tuple[int, ...] = (256,)
    team_slot_readout_hidden: tuple[int, ...] = ()
    residual_head_hidden: tuple[int, ...] = (128,)
    use_relationship_integrations: bool = False
    use_1vx_posterior_variance: bool = True
    dropout: float = 0.1
    logit_clip: float | None = 5.0
    structural_antisymmetry: bool = False
    structural_antisymmetry_scale: float = 0.5
    identity_static_sidecar_dim: int = 0
    identity_full_game_sidecar_dim: int = 0
    identity_temporal_sidecar_dim: int = 0
    use_identity_static_sidecar: bool = False
    use_identity_full_game_sidecar: bool = False
    use_identity_temporal_sidecar: bool = False
    identity_encoder_sidecar_hidden: tuple[int, ...] = (64,)
    identity_encoder_sidecar_dropout: float = 0.0
    identity_encoder_sidecar_support_strength: float = 30.0
    use_identity_semantic_context_head: bool = False
    semantic_context_dim: int = 96
    semantic_context_hidden: tuple[int, ...] = (128,)
    semantic_context_dropout: float = 0.0
    semantic_context_support_strength: float = 30.0
    use_learned_semantic_moe: bool = False
    semantic_moe_num_experts: int = 8
    semantic_moe_top_k: int = 2
    semantic_moe_factor_dim: int = 64
    semantic_moe_factor_hidden: tuple[int, ...] = (128,)
    semantic_moe_router_hidden: tuple[int, ...] = (128,)
    semantic_moe_expert_hidden: tuple[int, ...] = (64,)
    semantic_moe_dropout: float = 0.0
    semantic_moe_context_token_dropout: float = 0.05
    semantic_moe_temperature: float = 1.0
    semantic_moe_support_strength: float = 30.0
    semantic_moe_balance_weight: float = 1.0e-2
    semantic_moe_entropy_weight: float = 1.0e-3
    semantic_moe_factor_orthogonality_weight: float = 1.0e-3
    semantic_moe_factor_variance_weight: float = 1.0e-3
    semantic_moe_factor_std_floor: float = 0.05
    semantic_moe_delta_l2_weight: float = 0.0
    use_semantic_group_features: bool = False
    semantic_group_feature_dim: int = SEMANTIC_GROUP_FEATURE_DIM
    semantic_group_relationship_hidden: tuple[int, ...] = (128,)
    semantic_group_relationship_dropout: float = 0.0
    semantic_group_relationship_l2_weight: float = 0.0


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
    used by cache smoothing (`smoothing.composite_interaction_priors`).
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
        "delta_logit_1v1": joint_1v1 - expected_1v1,
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
    """Explicit support features derived from support counts.

    Posterior variance has a tiny numeric range for this dataset, so the model also
    sees direct confidence and log support. The missing flag is retained only for
    opt-in relationship heads where absent relationship tables are distinguishable
    from neutral internal placeholders.
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
    p1_cnt: Any,
    strength: float,
    matchup_1v1: Any | None = None,
    synergy_2vx: Any | None = None,
    m1v1_cnt: Any | None = None,
    s2vx_cnt: Any | None = None,
    identity_static_sidecar: Any | None = None,
    identity_full_game_sidecar: Any | None = None,
    identity_temporal_sidecar: Any | None = None,
    identity_encoder_support: Any | None = None,
    semantic_group_features: Any | None = None,
    include_relationship_features: bool = False,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Turn raw cache/prior arrays into the model's node/edge tensors.

    Single source of truth shared by training and the runtime predictor. Accepts
    numpy arrays or tensors. mu values stay in probability space (the model maps
    them to logits internally); champion/build ids become long embedding indices.

    The production path no longer consumes direct 1v1/2vX relationship arrays.
    When they are absent, neutral internal placeholders preserve the model
    tensor contract for swapping. Explicit research callers can still pass
    relationship arrays and set include_relationship_features=True to expose
    their deltas/support tensors.
    """

    def to_tensor(arr: Any) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def to_long(arr: Any) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.long, device=device)

    count_1vx = to_tensor(p1_cnt)
    mu_1vx, var_1vx = posterior_mean_var(to_tensor(win_rate), count_1vx, strength)

    def neutral_relationship(width: int, value: float) -> torch.Tensor:
        return count_1vx.new_full((count_1vx.shape[0], width), value)

    count_2vx = (
        neutral_relationship(N_SYNERGIES_2VX, 0.0)
        if s2vx_cnt is None
        else to_tensor(s2vx_cnt)
    )
    count_1v1 = (
        neutral_relationship(N_MATCHUPS_1V1, 0.0)
        if m1v1_cnt is None
        else to_tensor(m1v1_cnt)
    )
    mu_2vx = (
        neutral_relationship(N_SYNERGIES_2VX, 0.5)
        if synergy_2vx is None
        else to_tensor(synergy_2vx).clamp(0.0, 1.0)
    )
    mu_1v1 = (
        neutral_relationship(N_MATCHUPS_1V1, 0.5)
        if matchup_1v1 is None
        else to_tensor(matchup_1v1).clamp(0.0, 1.0)
    )
    conf_1vx, log_count_1vx, _ = support_features(count_1vx, strength)
    inputs = {
        "champion_id": to_long(champion_id),
        "build_id": to_long(build_id),
        "mu_1vx": mu_1vx,
        "var_1vx": var_1vx,
        "conf_1vx": conf_1vx,
        "log_count_1vx": log_count_1vx,
        "mu_2vx": mu_2vx,
        "mu_1v1": mu_1v1,
    }
    if include_relationship_features:
        conf_2vx, log_count_2vx, missing_2vx = support_features(count_2vx, strength)
        conf_1v1, log_count_1v1, missing_1v1 = support_features(count_1v1, strength)
        inputs.update(
            {
                "conf_2vx": conf_2vx,
                "log_count_2vx": log_count_2vx,
                "missing_2vx": missing_2vx,
                "conf_1v1": conf_1v1,
                "log_count_1v1": log_count_1v1,
                "missing_1v1": missing_1v1,
            }
        )
    if include_relationship_features:
        inputs.update(relationship_logit_features(mu_1vx=mu_1vx, mu_2vx=mu_2vx, mu_1v1=mu_1v1))
    for name, value in (
        ("identity_static_sidecar", identity_static_sidecar),
        ("identity_full_game_sidecar", identity_full_game_sidecar),
        ("identity_temporal_sidecar", identity_temporal_sidecar),
        ("identity_encoder_support", identity_encoder_support),
        ("semantic_group_features", semantic_group_features),
    ):
        if value is not None:
            inputs[name] = to_tensor(value)
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

    def swap_1v1_matrix(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, 5, 5).transpose(1, 2).reshape(-1, 25)

    def swap_1v1_signed(x: torch.Tensor) -> torch.Tensor:
        return -x.reshape(-1, 5, 5).transpose(1, 2).reshape(-1, 25)

    def swap_1v1_detail(x: torch.Tensor) -> torch.Tensor:
        return -x.reshape(x.shape[0], 5, 5, x.shape[-1]).transpose(1, 2).reshape(x.shape[0], 25, x.shape[-1])

    swapped = {
        "champion_id": swap_halves(inputs["champion_id"], 5),
        "build_id": swap_halves(inputs["build_id"], 5),
        "mu_1vx": swap_halves(inputs["mu_1vx"], 5),
        "var_1vx": swap_halves(inputs["var_1vx"], 5),
        "mu_2vx": swap_halves(inputs["mu_2vx"], 10),
        "mu_1v1": swap_1v1_mu(inputs["mu_1v1"]),
    }
    for key in (
        "identity_static_sidecar",
        "identity_full_game_sidecar",
        "identity_temporal_sidecar",
        "identity_encoder_support",
        "semantic_group_features",
    ):
        if key in inputs:
            swapped[key] = swap_halves(inputs[key], 5)
    if "var_2vx" in inputs:
        swapped["var_2vx"] = swap_halves(inputs["var_2vx"], 10)
    if "var_1v1" in inputs:
        swapped["var_1v1"] = swap_1v1_matrix(inputs["var_1v1"])
    for suffix, half, prefixes in (
        ("1vx", 5, ("conf", "log_count")),
        ("2vx", 10, ("conf", "log_count", "missing")),
    ):
        for prefix in prefixes:
            key = f"{prefix}_{suffix}"
            if key in inputs:
                swapped[key] = swap_halves(inputs[key], half)
    for prefix in ("conf", "log_count", "missing"):
        key = f"{prefix}_1v1"
        if key in inputs:
            swapped[key] = swap_1v1_matrix(inputs[key])
    if "delta_logit_2vx" in inputs:
        swapped["delta_logit_2vx"] = swap_halves(inputs["delta_logit_2vx"], 10)
    if "delta_logit_1v1" in inputs:
        swapped["delta_logit_1v1"] = swap_1v1_signed(inputs["delta_logit_1v1"])
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


def _zero_last_linear(module: nn.Module) -> None:
    last: nn.Module | None = None
    if isinstance(module, nn.Sequential) and len(module) > 0:
        last = module[-1]
    elif isinstance(module, nn.Linear):
        last = module
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


class PhiEncoder(nn.Module):
    """Uncertainty-gated encoder for the 1vX node posterior."""

    def __init__(self, config: HGNNConfig) -> None:
        super().__init__()
        self.use_variance = bool(config.use_1vx_posterior_variance)
        value_dim = 4 if self.use_variance else 3
        gate_dim = 3 if self.use_variance else 2
        self.value = _mlp(value_dim, config.value_hidden, config.edge_hidden, dropout=config.dropout)
        self.gate = _mlp(gate_dim, config.gate_hidden, config.edge_hidden, dropout=config.dropout)

    def forward(
        self,
        mu_logit: torch.Tensor,
        var: torch.Tensor,
        confidence: torch.Tensor | None = None,
        log_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if confidence is None or log_count is None:
            raise ValueError("PhiEncoder requires confidence/log_count tensors")
        if self.use_variance:
            precision = 1.0 / (1.0 + var)
            value_input = torch.stack([mu_logit, var, confidence, log_count], dim=-1)
            gate_input = torch.stack([precision, confidence, log_count], dim=-1)
        else:
            value_input = torch.stack([mu_logit, confidence, log_count], dim=-1)
            gate_input = torch.stack([confidence, log_count], dim=-1)
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

    def components(
        self,
        champion_id: torch.Tensor,
        build_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        e_c = self.champion(champion_id)  # [B, 10, dim]
        e_r = (
            self.role(cast(torch.Tensor, self.role_idx))
            .unsqueeze(0)
            .expand(champion_id.shape[0], -1, -1)
        )  # [B, 10, dim]
        e_b = self.build(build_id)  # [B, 10, dim]
        fused = self.norm(e_c * (1.0 + self.w_role(e_r)) * (1.0 + self.w_build(e_b)))
        return {
            "champion": e_c,
            "role": e_r,
            "build": e_b,
            "fused": fused,
        }

    def forward(self, champion_id: torch.Tensor, build_id: torch.Tensor) -> torch.Tensor:
        return self.components(champion_id, build_id)["fused"]


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


# own / ally / enemy semantic feature blocks scored per slot. The first seven
# are means/interactions; the last three add order-statistics (max) so convex
# composition signal ("3 burst threats") is not averaged away by mean pooling.
N_SEMANTIC_FEATURE_BLOCKS = 10

# Explicit semantic-group relationship blocks. These stay at the raw feature
# grain so a focus identity can learn separate responses to every promoted
# grouping across own, allied, enemy, count, tail, and interaction contexts.
N_SEMANTIC_GROUP_RELATION_BLOCKS = 11


class IdentitySemanticContextHead(nn.Module):
    """Slot-level own/ally/enemy latent interaction over the three encoder blocks."""

    def __init__(self, config: HGNNConfig) -> None:
        super().__init__()
        input_dim = (
            int(config.identity_static_sidecar_dim)
            + int(config.identity_full_game_sidecar_dim)
            + int(config.identity_temporal_sidecar_dim)
        )
        if input_dim <= 0:
            raise ValueError("semantic context head requires non-empty identity sidecar dims")
        if (
            int(config.identity_static_sidecar_dim) <= 0
            or int(config.identity_full_game_sidecar_dim) <= 0
            or int(config.identity_temporal_sidecar_dim) <= 0
        ):
            raise ValueError(
                "semantic context head requires static, full-game, and temporal sidecar dims"
            )
        context_dim = int(config.semantic_context_dim)
        if context_dim <= 0:
            raise ValueError("semantic_context_dim must be positive")
        self.static_dim = int(config.identity_static_sidecar_dim)
        self.full_game_dim = int(config.identity_full_game_sidecar_dim)
        self.temporal_dim = int(config.identity_temporal_sidecar_dim)
        self.support_strength = float(config.semantic_context_support_strength)
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, context_dim),
            nn.ReLU(),
            nn.LayerNorm(context_dim),
        )
        self.score = _mlp(
            context_dim * N_SEMANTIC_FEATURE_BLOCKS,
            tuple(int(dim) for dim in config.semantic_context_hidden),
            1,
            dropout=float(config.semantic_context_dropout),
        )
        last = self.score[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        # Learned amplitude on the (zero-initialised) context residual. Starts at
        # 1.0 and stays a no-op at init because the score head is zero-init; it
        # lets the optimiser grow the context correction without fighting the
        # score head's weight decay, countering the systematic effect-shrinkage.
        self.context_scale = nn.Parameter(torch.ones(()))

    def _confidence(self, support: torch.Tensor) -> torch.Tensor:
        strength = max(self.support_strength, 0.0)
        support = support.to(dtype=torch.float32).clamp_min(0.0)
        return support / (support + strength + 1.0e-12)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted = values * weights.unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        return weighted.sum(dim=1) / denom

    @staticmethod
    def _leave_one_out_max(own: torch.Tensor) -> torch.Tensor:
        """Per-slot max over the other four own-team slots (extremity, LOO).

        Uses top-2 over the team so the focus slot is excluded without a Python
        loop: if a slot is its own argmax, it falls back to the runner-up.
        """
        top2 = own.topk(2, dim=1)
        max1 = top2.values[:, 0]
        max2 = top2.values[:, 1]
        argmax0 = top2.indices[:, 0]
        slot_ids = torch.arange(own.shape[1], device=own.device).view(1, -1, 1)
        focus_is_argmax = argmax0.unsqueeze(1) == slot_ids
        return torch.where(focus_is_argmax, max2.unsqueeze(1), max1.unsqueeze(1))

    def _side_contexts(
        self,
        own: torch.Tensor,
        enemy: torch.Tensor,
        own_conf: torch.Tensor,
        enemy_conf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        own_sum = (own * own_conf.unsqueeze(-1)).sum(dim=1, keepdim=True)
        own_den = own_conf.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        ally_num = own_sum - own * own_conf.unsqueeze(-1)
        ally_den = (own_den - own_conf).clamp_min(1.0e-12).unsqueeze(-1)
        ally_context = ally_num / ally_den
        enemy_context = self._weighted_mean(enemy, enemy_conf).unsqueeze(1).expand_as(own)
        # Extremity (max) summaries preserve the convex composition signal that
        # the support-weighted mean averages away (e.g. "3 burst threats").
        ally_max = self._leave_one_out_max(own)
        enemy_max = enemy.max(dim=1).values.unsqueeze(1).expand_as(own)
        return ally_context, enemy_context, ally_max, enemy_max

    def forward(
        self,
        *,
        static: torch.Tensor,
        full_game: torch.Tensor,
        temporal: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        if support.ndim != 2 or support.shape[1] != N_PLAYERS:
            raise ValueError("identity_encoder_support must have shape [batch, 10]")
        for name, value, expected_dim in (
            ("identity_static_sidecar", static, self.static_dim),
            ("identity_full_game_sidecar", full_game, self.full_game_dim),
            ("identity_temporal_sidecar", temporal, self.temporal_dim),
        ):
            if value.ndim != 3 or value.shape[1] != N_PLAYERS:
                raise ValueError(f"{name} must have shape [batch, 10, {expected_dim}]")
            if value.shape[0] != support.shape[0] or value.shape[2] != expected_dim:
                raise ValueError(f"{name} must have shape [batch, 10, {expected_dim}]")
        latent = torch.cat([static, full_game, temporal], dim=-1)
        semantic = self.project(latent)
        confidence = self._confidence(support).to(dtype=semantic.dtype)

        blue, red = semantic[:, :5], semantic[:, 5:]
        blue_conf, red_conf = confidence[:, :5], confidence[:, 5:]
        blue_ally, blue_enemy, blue_ally_max, blue_enemy_max = self._side_contexts(
            blue, red, blue_conf, red_conf
        )
        red_ally, red_enemy, red_ally_max, red_enemy_max = self._side_contexts(
            red, blue, red_conf, blue_conf
        )
        own = torch.cat([blue, red], dim=1)
        ally = torch.cat([blue_ally, red_ally], dim=1)
        enemy = torch.cat([blue_enemy, red_enemy], dim=1)
        ally_max = torch.cat([blue_ally_max, red_ally_max], dim=1)
        enemy_max = torch.cat([blue_enemy_max, red_enemy_max], dim=1)
        features = torch.cat(
            [
                own,
                ally,
                enemy,
                own * ally,
                own * enemy,
                ally * enemy,
                enemy - ally,
                ally_max,
                enemy_max,
                own * enemy_max,
            ],
            dim=-1,
        )
        slot_scores = self.score(features).squeeze(-1) * confidence
        context = slot_scores[:, :5].mean(dim=1) - slot_scores[:, 5:].mean(dim=1)
        return self.context_scale * context


class LearnedSemanticMoEHead(nn.Module):
    """Learned slot-factor MoE over identity and frozen encoder sidecar latents."""

    def __init__(self, config: HGNNConfig) -> None:
        super().__init__()
        self.static_dim = int(config.identity_static_sidecar_dim)
        self.full_game_dim = int(config.identity_full_game_sidecar_dim)
        self.temporal_dim = int(config.identity_temporal_sidecar_dim)
        sidecar_dim = self.static_dim + self.full_game_dim + self.temporal_dim
        if sidecar_dim <= 0:
            raise ValueError("learned semantic MoE requires non-empty identity sidecar dims")
        if self.static_dim <= 0 or self.full_game_dim <= 0 or self.temporal_dim <= 0:
            raise ValueError(
                "learned semantic MoE requires static, full-game, and temporal sidecar dims"
            )

        self.num_experts = int(config.semantic_moe_num_experts)
        self.top_k = int(config.semantic_moe_top_k)
        self.factor_dim = int(config.semantic_moe_factor_dim)
        if self.num_experts <= 0:
            raise ValueError("semantic_moe_num_experts must be positive")
        if self.top_k <= 0:
            raise ValueError("semantic_moe_top_k must be positive")
        if self.factor_dim <= 0:
            raise ValueError("semantic_moe_factor_dim must be positive")
        self.top_k = min(self.top_k, self.num_experts)
        self.temperature = max(float(config.semantic_moe_temperature), 1.0e-6)
        self.support_strength = float(config.semantic_moe_support_strength)
        self.context_token_dropout = max(float(config.semantic_moe_context_token_dropout), 0.0)
        self.balance_weight = float(config.semantic_moe_balance_weight)
        self.entropy_weight = float(config.semantic_moe_entropy_weight)
        self.factor_orthogonality_weight = float(
            config.semantic_moe_factor_orthogonality_weight
        )
        self.factor_variance_weight = float(config.semantic_moe_factor_variance_weight)
        self.factor_std_floor = float(config.semantic_moe_factor_std_floor)
        self.delta_l2_weight = float(config.semantic_moe_delta_l2_weight)
        self.use_semantic_group_features = bool(config.use_semantic_group_features)
        self.group_feature_dim = int(config.semantic_group_feature_dim)
        if self.use_semantic_group_features and self.group_feature_dim <= 0:
            raise ValueError("semantic_group_feature_dim must be positive when enabled")

        sidecar_token_dim = sidecar_dim + 2
        identity_token_dim = int(config.node_dim) * 4
        group_context_dim = self.group_feature_dim * 7 if self.use_semantic_group_features else 0
        group_token_dim = self.factor_dim if self.use_semantic_group_features else 0
        group_relationship_dim = (
            self.group_feature_dim * N_SEMANTIC_GROUP_RELATION_BLOCKS
            if self.use_semantic_group_features
            else 0
        )
        token_dim = identity_token_dim + self.factor_dim * 5 + group_token_dim + 2
        relationship_token_dim = identity_token_dim + self.factor_dim
        dropout = float(config.semantic_moe_dropout)
        self.sidecar_factor = nn.Sequential(
            nn.LayerNorm(sidecar_token_dim),
            _mlp(
                sidecar_token_dim,
                tuple(int(dim) for dim in config.semantic_moe_factor_hidden),
                self.factor_dim,
                dropout=dropout,
            ),
            nn.LayerNorm(self.factor_dim),
        )
        self.factor = nn.Sequential(
            nn.LayerNorm(token_dim),
            _mlp(
                token_dim,
                tuple(int(dim) for dim in config.semantic_moe_factor_hidden),
                self.factor_dim,
                dropout=dropout,
            ),
            nn.LayerNorm(self.factor_dim),
        )
        self.group_context = (
            nn.Sequential(
                nn.LayerNorm(group_context_dim),
                _mlp(
                    group_context_dim,
                    tuple(int(dim) for dim in config.semantic_moe_factor_hidden),
                    self.factor_dim,
                    dropout=dropout,
                ),
                nn.LayerNorm(self.factor_dim),
            )
            if self.use_semantic_group_features
            else None
        )
        if self.group_context is not None:
            inner = self.group_context[1]
            _zero_last_linear(inner)
        self.group_relationship = (
            nn.Sequential(
                nn.LayerNorm(relationship_token_dim),
                _mlp(
                    relationship_token_dim,
                    tuple(int(dim) for dim in config.semantic_group_relationship_hidden),
                    group_relationship_dim + 1,
                    dropout=float(config.semantic_group_relationship_dropout),
                ),
            )
            if self.use_semantic_group_features
            else None
        )
        if self.group_relationship is not None:
            inner = self.group_relationship[1]
            _zero_last_linear(inner)
        self.group_relationship_context_dim = group_relationship_dim
        self.group_relationship_l2_weight = float(
            config.semantic_group_relationship_l2_weight
        )
        self.router = _mlp(
            self.factor_dim + int(config.node_dim),
            tuple(int(dim) for dim in config.semantic_moe_router_hidden),
            self.num_experts,
            dropout=dropout,
        )
        self.experts = nn.ModuleList(
            [
                _mlp(
                    self.factor_dim,
                    tuple(int(dim) for dim in config.semantic_moe_expert_hidden),
                    1,
                    dropout=dropout,
                )
                for _ in range(self.num_experts)
            ]
        )
        for expert in self.experts:
            _zero_last_linear(expert)
        self.response_scale = nn.Parameter(torch.ones(()))

    def _confidence(self, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        strength = max(self.support_strength, 0.0)
        support = support.to(dtype=torch.float32).clamp_min(0.0)
        confidence = support / (support + strength + 1.0e-12)
        return confidence, torch.log1p(support)

    def _mask_context_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.context_token_dropout <= 0.0:
            return tokens, tokens.new_ones(())
        keep_prob = max(1.0 - self.context_token_dropout, 1.0e-6)
        mask = (torch.rand(tokens.shape[:2] + (1,), device=tokens.device) < keep_prob).to(
            dtype=tokens.dtype
        )
        return tokens * mask / keep_prob, mask.mean()

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        return (values * weights.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _leave_one_out_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        total = (values * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        denom = weights.sum(dim=1, keepdim=True).unsqueeze(-1)
        loo_total = total - values * weights.unsqueeze(-1)
        loo_denom = (denom - weights.unsqueeze(-1)).clamp_min(1.0e-12)
        return loo_total / loo_denom

    @staticmethod
    def _leave_one_out_sum(values: torch.Tensor) -> torch.Tensor:
        return values.sum(dim=1, keepdim=True) - values

    @staticmethod
    def _leave_one_out_max(values: torch.Tensor) -> torch.Tensor:
        top2 = values.topk(2, dim=1)
        max1 = top2.values[:, 0]
        max2 = top2.values[:, 1]
        argmax0 = top2.indices[:, 0]
        slot_ids = torch.arange(values.shape[1], device=values.device).view(1, -1, 1)
        focus_is_argmax = argmax0.unsqueeze(1) == slot_ids
        return torch.where(focus_is_argmax, max2.unsqueeze(1), max1.unsqueeze(1))

    def _side_contexts(
        self,
        own: torch.Tensor,
        enemy: torch.Tensor,
        own_conf: torch.Tensor,
        enemy_conf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ally_mean = self._leave_one_out_mean(own, own_conf)
        enemy_mean = self._weighted_mean(enemy, enemy_conf).unsqueeze(1).expand_as(own)
        ally_max = self._leave_one_out_max(own)
        enemy_max = enemy.max(dim=1).values.unsqueeze(1).expand_as(own)
        return ally_mean, enemy_mean, ally_max, enemy_max

    def _context_features(
        self,
        sidecar_factor: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        blue, red = sidecar_factor[:, :5], sidecar_factor[:, 5:]
        blue_conf, red_conf = confidence[:, :5], confidence[:, 5:]
        blue_ally, blue_enemy, blue_ally_max, blue_enemy_max = self._side_contexts(
            blue, red, blue_conf, red_conf
        )
        red_ally, red_enemy, red_ally_max, red_enemy_max = self._side_contexts(
            red, blue, red_conf, blue_conf
        )
        ally = torch.cat([blue_ally, red_ally], dim=1)
        enemy = torch.cat([blue_enemy, red_enemy], dim=1)
        ally_max = torch.cat([blue_ally_max, red_ally_max], dim=1)
        enemy_max = torch.cat([blue_enemy_max, red_enemy_max], dim=1)
        return torch.cat([sidecar_factor, ally, enemy, ally_max, enemy_max], dim=-1)

    def _validate_group_features(
        self,
        semantic_group_features: torch.Tensor | None,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.use_semantic_group_features:
            return None
        if semantic_group_features is None:
            raise ValueError("learned semantic MoE requires semantic_group_features")
        if (
            semantic_group_features.ndim != 3
            or semantic_group_features.shape[1] != N_PLAYERS
            or semantic_group_features.shape[2] != self.group_feature_dim
        ):
            raise ValueError(
                "semantic_group_features must have shape "
                f"[batch, 10, {self.group_feature_dim}]"
            )
        if semantic_group_features.shape[0] != reference.shape[0]:
            raise ValueError(
                "semantic_group_features batch dimension must match identity inputs"
            )
        return semantic_group_features.to(dtype=reference.dtype)

    def _group_context_features(
        self,
        semantic_group_features: torch.Tensor,
    ) -> torch.Tensor:
        (
            own,
            ally_mean,
            enemy_mean,
            ally_sum,
            enemy_sum,
            ally_max,
            enemy_max,
        ) = self._group_summary_features(semantic_group_features)
        return torch.cat(
            [own, ally_mean, enemy_mean, ally_sum, enemy_sum, ally_max, enemy_max],
            dim=-1,
        )

    def _group_summary_features(
        self,
        semantic_group_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        blue, red = semantic_group_features[:, :5], semantic_group_features[:, 5:]
        blue_ally_mean = self._leave_one_out_mean(
            blue,
            torch.ones_like(blue[:, :, 0]),
        )
        red_ally_mean = self._leave_one_out_mean(
            red,
            torch.ones_like(red[:, :, 0]),
        )
        blue_enemy_mean = red.mean(dim=1, keepdim=True).expand_as(blue)
        red_enemy_mean = blue.mean(dim=1, keepdim=True).expand_as(red)
        blue_ally_sum = self._leave_one_out_sum(blue)
        red_ally_sum = self._leave_one_out_sum(red)
        blue_enemy_sum = red.sum(dim=1, keepdim=True).expand_as(blue)
        red_enemy_sum = blue.sum(dim=1, keepdim=True).expand_as(red)
        blue_ally_max = self._leave_one_out_max(blue)
        red_ally_max = self._leave_one_out_max(red)
        blue_enemy_max = red.max(dim=1).values.unsqueeze(1).expand_as(blue)
        red_enemy_max = blue.max(dim=1).values.unsqueeze(1).expand_as(red)

        own = torch.cat([blue, red], dim=1)
        ally_mean = torch.cat([blue_ally_mean, red_ally_mean], dim=1)
        enemy_mean = torch.cat([blue_enemy_mean, red_enemy_mean], dim=1)
        ally_sum = torch.cat([blue_ally_sum, red_ally_sum], dim=1)
        enemy_sum = torch.cat([blue_enemy_sum, red_enemy_sum], dim=1)
        ally_max = torch.cat([blue_ally_max, red_ally_max], dim=1)
        enemy_max = torch.cat([blue_enemy_max, red_enemy_max], dim=1)
        return own, ally_mean, enemy_mean, ally_sum, enemy_sum, ally_max, enemy_max

    def _group_relationship_features(
        self,
        semantic_group_features: torch.Tensor,
    ) -> torch.Tensor:
        (
            own,
            ally_mean,
            enemy_mean,
            ally_sum,
            enemy_sum,
            ally_max,
            enemy_max,
        ) = self._group_summary_features(semantic_group_features)
        return torch.cat(
            [
                own,
                ally_mean,
                enemy_mean,
                ally_sum,
                enemy_sum,
                ally_max,
                enemy_max,
                enemy_mean - ally_mean,
                enemy_sum - ally_sum,
                own * enemy_sum,
                own * ally_sum,
            ],
            dim=-1,
        )

    def _group_relationship_outputs(
        self,
        *,
        identity_token: torch.Tensor,
        group_features: torch.Tensor | None,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.group_relationship is None:
            zero = identity_token.new_zeros(identity_token.shape[:2])
            return {
                "slot_delta": zero,
                "logit": zero[:, :5].mean(dim=1) - zero[:, 5:].mean(dim=1),
                "coeff_l2_loss": identity_token.new_zeros(()),
                "coeff_norm": identity_token.new_zeros(()),
                "context_norm": identity_token.new_zeros(()),
            }
        if group_features is None:
            raise ValueError("semantic group relationship head requires group features")
        relation_features = self._group_relationship_features(group_features)
        coeff_and_bias = self.group_relationship(identity_token)
        coeff = coeff_and_bias[..., : self.group_relationship_context_dim]
        bias = coeff_and_bias[..., self.group_relationship_context_dim]
        scale = float(max(self.group_relationship_context_dim, 1)) ** -0.5
        slot_delta_raw = (coeff * relation_features).sum(dim=-1) * scale + bias
        slot_delta = slot_delta_raw * confidence
        logit = slot_delta[:, :5].mean(dim=1) - slot_delta[:, 5:].mean(dim=1)
        return {
            "slot_delta": slot_delta,
            "logit": logit,
            "coeff_l2_loss": coeff.pow(2).mean(),
            "coeff_norm": coeff.norm(dim=-1).mean(),
            "context_norm": relation_features.norm(dim=-1).mean(),
        }

    def _factor_regularization(
        self,
        semantic_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = semantic_factor.reshape(-1, semantic_factor.shape[-1])
        centered = flat - flat.mean(dim=0, keepdim=True)
        std = centered.std(dim=0, unbiased=False)
        variance_loss = (self.factor_std_floor - std).clamp_min(0.0).pow(2).mean()
        normalized = centered / std.clamp_min(1.0e-6)
        denom = max(int(flat.shape[0] - 1), 1)
        corr = normalized.transpose(0, 1) @ normalized / float(denom)
        off_diag = corr - torch.diag(torch.diagonal(corr))
        orthogonality_loss = off_diag.pow(2).mean()
        return orthogonality_loss, variance_loss, std.mean(), std.min()

    def _validate_sidecar(
        self,
        *,
        static: torch.Tensor,
        full_game: torch.Tensor,
        temporal: torch.Tensor,
        support: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if support.ndim != 2 or support.shape[1] != N_PLAYERS:
            raise ValueError("identity_encoder_support must have shape [batch, 10]")
        for name, value, expected_dim in (
            ("identity_static_sidecar", static, self.static_dim),
            ("identity_full_game_sidecar", full_game, self.full_game_dim),
            ("identity_temporal_sidecar", temporal, self.temporal_dim),
        ):
            if value.ndim != 3 or value.shape[1] != N_PLAYERS:
                raise ValueError(f"{name} must have shape [batch, 10, {expected_dim}]")
            if value.shape[0] != support.shape[0] or value.shape[2] != expected_dim:
                raise ValueError(f"{name} must have shape [batch, 10, {expected_dim}]")
        dtype = reference.dtype
        return (
            static.to(dtype=dtype),
            full_game.to(dtype=dtype),
            temporal.to(dtype=dtype),
            support.to(dtype=torch.float32),
        )

    @staticmethod
    def regularization_from_outputs(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = outputs.get("semantic_moe_regularization_loss")
        if loss is not None:
            return loss
        final_logit = outputs.get("final_logit")
        if final_logit is None:
            raise ValueError("outputs must include final_logit or semantic_moe_regularization_loss")
        return final_logit.new_zeros(())

    @staticmethod
    def stats_from_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in outputs.items()
            if key.startswith("semantic_moe_") and key != "semantic_moe_regularization_loss"
        }

    def forward(
        self,
        *,
        champion_embedding: torch.Tensor,
        role_embedding: torch.Tensor,
        build_embedding: torch.Tensor,
        fused_identity: torch.Tensor,
        static: torch.Tensor,
        full_game: torch.Tensor,
        temporal: torch.Tensor,
        support: torch.Tensor,
        semantic_group_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        static, full_game, temporal, support = self._validate_sidecar(
            static=static,
            full_game=full_game,
            temporal=temporal,
            support=support,
            reference=fused_identity,
        )
        group_features = self._validate_group_features(
            semantic_group_features,
            reference=fused_identity,
        )
        confidence, log_support = self._confidence(support)
        confidence = confidence.to(dtype=fused_identity.dtype)
        log_support = log_support.to(dtype=fused_identity.dtype)

        sidecar_token = torch.cat(
            [static, full_game, temporal, confidence.unsqueeze(-1), log_support.unsqueeze(-1)],
            dim=-1,
        )
        sidecar_factor = self.sidecar_factor(sidecar_token)
        sidecar_factor, token_keep_fraction = self._mask_context_tokens(sidecar_factor)
        context_features = self._context_features(sidecar_factor, confidence)
        token_parts = [
            champion_embedding,
            role_embedding,
            build_embedding,
            fused_identity,
            context_features,
        ]
        if self.group_context is not None:
            assert group_features is not None
            token_parts.append(self.group_context(self._group_context_features(group_features)))
        token_parts.extend([confidence.unsqueeze(-1), log_support.unsqueeze(-1)])
        token = torch.cat(token_parts, dim=-1)
        semantic_factor = self.factor(token)
        relationship_token = torch.cat(
            [
                champion_embedding,
                role_embedding,
                build_embedding,
                fused_identity,
                semantic_factor,
            ],
            dim=-1,
        )
        group_relationship = self._group_relationship_outputs(
            identity_token=relationship_token,
            group_features=group_features,
            confidence=confidence,
        )

        router_input = torch.cat([semantic_factor, fused_identity], dim=-1)
        router_logits = self.router(router_input) / self.temperature
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1)
        route_probs = router_logits.new_zeros(router_logits.shape)
        route_probs.scatter_(-1, top_indices, top_weights)

        expert_outputs = torch.cat(
            [expert(semantic_factor) for expert in self.experts],
            dim=-1,
        )
        slot_delta_raw = (expert_outputs * route_probs).sum(dim=-1)
        expert_slot_delta = self.response_scale * slot_delta_raw * confidence
        slot_delta = expert_slot_delta + group_relationship["slot_delta"]
        logit = slot_delta[:, :5].mean(dim=1) - slot_delta[:, 5:].mean(dim=1)

        expert_usage = route_probs.mean(dim=(0, 1))
        selected_fraction = (route_probs > 0.0).to(dtype=route_probs.dtype).mean(dim=(0, 1))
        target_usage = route_probs.new_full((self.num_experts,), 1.0 / float(self.num_experts))
        balance_loss = (expert_usage - target_usage).pow(2).mean()
        entropy = -(route_probs.clamp_min(1.0e-12) * route_probs.clamp_min(1.0e-12).log()).sum(dim=-1)
        max_entropy = router_logits.new_tensor(float(self.top_k)).log()
        entropy_loss = (max_entropy - entropy).clamp_min(0.0).mean()
        factor_orthogonality_loss, factor_variance_loss, factor_std_mean, factor_std_min = (
            self._factor_regularization(semantic_factor)
        )
        delta_l2_loss = slot_delta_raw.pow(2).mean()
        regularization_loss = (
            self.balance_weight * balance_loss
            + self.entropy_weight * entropy_loss
            + self.factor_orthogonality_weight * factor_orthogonality_loss
            + self.factor_variance_weight * factor_variance_loss
            + self.delta_l2_weight * delta_l2_loss
            + self.group_relationship_l2_weight * group_relationship["coeff_l2_loss"]
        )
        return {
            "semantic_moe_logit": logit,
            "semantic_moe_slot_delta": slot_delta,
            "semantic_moe_expert_logit": (
                expert_slot_delta[:, :5].mean(dim=1)
                - expert_slot_delta[:, 5:].mean(dim=1)
            ),
            "semantic_moe_group_relationship_logit": group_relationship["logit"],
            "semantic_moe_group_relationship_slot_delta": group_relationship["slot_delta"],
            "semantic_moe_router_probs": route_probs,
            "semantic_moe_topk_indices": top_indices,
            "semantic_moe_topk_weights": top_weights,
            "semantic_moe_expert_usage": expert_usage,
            "semantic_moe_expert_selected_fraction": selected_fraction,
            "semantic_moe_router_entropy": entropy.mean(),
            "semantic_moe_factor_norm": semantic_factor.norm(dim=-1).mean(),
            "semantic_moe_balance_loss": balance_loss,
            "semantic_moe_entropy_loss": entropy_loss,
            "semantic_moe_factor_orthogonality_loss": factor_orthogonality_loss,
            "semantic_moe_factor_variance_loss": factor_variance_loss,
            "semantic_moe_factor_std_mean": factor_std_mean,
            "semantic_moe_factor_std_min": factor_std_min,
            "semantic_moe_context_token_keep_fraction": token_keep_fraction,
            "semantic_moe_delta_l2_loss": delta_l2_loss,
            "semantic_moe_group_relationship_l2_loss": group_relationship["coeff_l2_loss"],
            "semantic_moe_group_relationship_coeff_norm": group_relationship["coeff_norm"],
            "semantic_moe_group_relationship_context_norm": group_relationship["context_norm"],
            "semantic_moe_regularization_loss": regularization_loss,
            "semantic_moe_group_features_enabled": logit.new_tensor(
                1.0 if self.use_semantic_group_features else 0.0
            ),
            "semantic_moe_group_feature_dim": logit.new_tensor(float(self.group_feature_dim)),
            "semantic_moe_group_relationship_enabled": logit.new_tensor(
                1.0 if self.group_relationship is not None else 0.0
            ),
        }


class HGNNWinModel(nn.Module):
    def __init__(self, config: HGNNConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        if config is not None and overrides:
            raise ValueError("Pass either config or keyword overrides, not both")
        self.config = config or HGNNConfig(**overrides)
        c = self.config

        self.phi = nn.ModuleDict({"1vx": PhiEncoder(c)})
        self.identity = IdentityEncoder(c.n_champions, c.n_builds, c.node_dim)
        sidecar_input_dim = self._identity_sidecar_input_dim(c)
        self.identity_sidecar = (
            _mlp(
                sidecar_input_dim + 2,
                c.identity_encoder_sidecar_hidden,
                c.node_dim,
                dropout=c.identity_encoder_sidecar_dropout,
            )
            if sidecar_input_dim > 0
            else None
        )
        if self.identity_sidecar is not None:
            last = self.identity_sidecar[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        self.identity_semantic_context = (
            IdentitySemanticContextHead(c)
            if c.use_identity_semantic_context_head
            else None
        )
        self.learned_semantic_moe = (
            LearnedSemanticMoEHead(c)
            if c.use_learned_semantic_moe
            else None
        )
        # Node init fuses the multiplicative identity with the 1vX posterior (§3).
        self.node_init = _mlp(
            c.node_dim + c.edge_hidden, c.node_init_hidden, c.node_dim, dropout=c.dropout
        )
        self.node_norm = nn.LayerNorm(c.node_dim)

        self.attn_pool = AttnPool(c.node_dim)
        self.team_proj = nn.Linear(c.node_dim * 2, c.node_dim)
        self.team_slot_readout = (
            _mlp(c.node_dim * 5, c.team_slot_readout_hidden, c.node_dim, dropout=c.dropout)
            if c.team_slot_readout_hidden
            else None
        )
        if self.team_slot_readout is not None:
            nn.init.zeros_(self.team_slot_readout[-1].weight)
            nn.init.zeros_(self.team_slot_readout[-1].bias)
        self.residual_head = _mlp(
            N_RELATIONSHIP_EDGES * 4,
            c.residual_head_hidden,
            c.node_dim,
            dropout=c.dropout,
        )
        self.head = _mlp(c.node_dim * 5, c.readout_hidden, 1, dropout=c.dropout)

    @staticmethod
    def _identity_sidecar_input_dim(config: HGNNConfig) -> int:
        dim = 0
        if config.use_identity_static_sidecar:
            dim += int(config.identity_static_sidecar_dim)
        if config.use_identity_full_game_sidecar:
            dim += int(config.identity_full_game_sidecar_dim)
        if config.use_identity_temporal_sidecar:
            dim += int(config.identity_temporal_sidecar_dim)
        return dim

    def _readout(self, team: torch.Tensor) -> torch.Tensor:  # team: [B, 5, d]
        pooled = torch.cat([team.mean(dim=1), self.attn_pool(team)], dim=-1)
        out = self.team_proj(pooled)
        if self.team_slot_readout is not None:
            out = out + self.team_slot_readout(team.flatten(start_dim=1))
        return out

    def _signed_relationship_tensor(
        self,
        tensor_2vx: torch.Tensor,
        tensor_1v1: torch.Tensor,
    ) -> torch.Tensor:
        signed_2vx = torch.cat([tensor_2vx[:, :10], -tensor_2vx[:, 10:]], dim=1)
        return _clip_logit(torch.cat([tensor_1v1, signed_2vx], dim=1), self.config.logit_clip)

    def _relationship_support_tensor(
        self,
        tensor_2vx: torch.Tensor | None,
        tensor_1v1: torch.Tensor | None,
        *,
        default_like: torch.Tensor,
        default_value: float,
    ) -> torch.Tensor:
        if tensor_1v1 is None or tensor_2vx is None:
            return torch.full_like(default_like, default_value)
        return torch.cat([tensor_1v1, tensor_2vx], dim=1)

    def _relationship_residual_features(
        self,
        *,
        delta_logit_2vx: torch.Tensor,
        delta_logit_1v1: torch.Tensor,
        conf_2vx: torch.Tensor | None,
        conf_1v1: torch.Tensor | None,
        missing_2vx: torch.Tensor | None = None,
        missing_1v1: torch.Tensor | None = None,
        include_support: bool,
    ) -> torch.Tensor:
        delta = self._signed_relationship_tensor(delta_logit_2vx, delta_logit_1v1)
        confidence = self._relationship_support_tensor(
            conf_2vx,
            conf_1v1,
            default_like=delta,
            default_value=1.0,
        )
        parts = [delta]
        if include_support:
            parts.append(confidence)
        parts.append(delta * confidence)
        if include_support:
            missing = self._relationship_support_tensor(
                missing_2vx,
                missing_1v1,
                default_like=delta,
                default_value=0.0,
            )
            parts.append(missing)
        return torch.cat(parts, dim=1)

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
        residual_input = self._relationship_residual_features(
            delta_logit_2vx=delta_logit_2vx,
            delta_logit_1v1=delta_logit_1v1,
            conf_2vx=conf_2vx,
            conf_1v1=conf_1v1,
            missing_2vx=missing_2vx,
            missing_1v1=missing_1v1,
            include_support=True,
        )
        return self.residual_head(residual_input)

    def _sidecar_block(
        self,
        value: torch.Tensor | None,
        *,
        batch_size: int,
        dim: int,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if dim <= 0:
            return reference.new_zeros((batch_size, 10, 0))
        if value is None:
            return reference.new_zeros((batch_size, 10, dim))
        if value.ndim != 3 or value.shape[1] != 10 or value.shape[2] != dim:
            raise ValueError(f"{name} must have shape [batch, 10, {dim}]")
        return value.to(dtype=reference.dtype)

    def _sidecar_node_features(
        self,
        *,
        champion_id: torch.Tensor,
        identity_static_sidecar: torch.Tensor | None,
        identity_full_game_sidecar: torch.Tensor | None,
        identity_temporal_sidecar: torch.Tensor | None,
        identity_encoder_support: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.identity_sidecar is None:
            return None
        c = self.config
        batch_size = int(champion_id.shape[0])
        parts: list[torch.Tensor] = []
        if c.use_identity_static_sidecar:
            parts.append(
                self._sidecar_block(
                    identity_static_sidecar,
                    batch_size=batch_size,
                    dim=int(c.identity_static_sidecar_dim),
                    reference=self.identity.champion.weight,
                    name="identity_static_sidecar",
                )
            )
        if c.use_identity_full_game_sidecar:
            parts.append(
                self._sidecar_block(
                    identity_full_game_sidecar,
                    batch_size=batch_size,
                    dim=int(c.identity_full_game_sidecar_dim),
                    reference=self.identity.champion.weight,
                    name="identity_full_game_sidecar",
                )
            )
        if c.use_identity_temporal_sidecar:
            parts.append(
                self._sidecar_block(
                    identity_temporal_sidecar,
                    batch_size=batch_size,
                    dim=int(c.identity_temporal_sidecar_dim),
                    reference=self.identity.champion.weight,
                    name="identity_temporal_sidecar",
                )
            )
        if not parts:
            return None
        if identity_encoder_support is None:
            support = champion_id.new_zeros((batch_size, 10), dtype=torch.float32)
        else:
            if identity_encoder_support.ndim != 2 or identity_encoder_support.shape[1] != 10:
                raise ValueError("identity_encoder_support must have shape [batch, 10]")
            support = identity_encoder_support.to(dtype=torch.float32)
        strength = max(float(c.identity_encoder_sidecar_support_strength), 0.0)
        confidence = support / (support + strength + 1.0e-12)
        log_support = torch.log1p(support.clamp_min(0.0))
        sidecar_input = torch.cat(
            [*parts, confidence.unsqueeze(-1), log_support.unsqueeze(-1)],
            dim=-1,
        )
        return self.identity_sidecar(sidecar_input)

    def _semantic_context_logit(
        self,
        *,
        identity_static_sidecar: torch.Tensor | None,
        identity_full_game_sidecar: torch.Tensor | None,
        identity_temporal_sidecar: torch.Tensor | None,
        identity_encoder_support: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.identity_semantic_context is None:
            return None
        missing = [
            name
            for name, value in (
                ("identity_static_sidecar", identity_static_sidecar),
                ("identity_full_game_sidecar", identity_full_game_sidecar),
                ("identity_temporal_sidecar", identity_temporal_sidecar),
                ("identity_encoder_support", identity_encoder_support),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "semantic context head requires all identity sidecar inputs: "
                + ", ".join(missing)
            )
        return self.identity_semantic_context(
            static=cast(torch.Tensor, identity_static_sidecar),
            full_game=cast(torch.Tensor, identity_full_game_sidecar),
            temporal=cast(torch.Tensor, identity_temporal_sidecar),
            support=cast(torch.Tensor, identity_encoder_support),
        )

    def _learned_semantic_moe_outputs(
        self,
        *,
        identity_components: dict[str, torch.Tensor],
        identity_static_sidecar: torch.Tensor | None,
        identity_full_game_sidecar: torch.Tensor | None,
        identity_temporal_sidecar: torch.Tensor | None,
        identity_encoder_support: torch.Tensor | None,
        semantic_group_features: torch.Tensor | None,
    ) -> dict[str, torch.Tensor] | None:
        if self.learned_semantic_moe is None:
            return None
        required_inputs: list[tuple[str, torch.Tensor | None]] = [
            ("identity_static_sidecar", identity_static_sidecar),
            ("identity_full_game_sidecar", identity_full_game_sidecar),
            ("identity_temporal_sidecar", identity_temporal_sidecar),
            ("identity_encoder_support", identity_encoder_support),
        ]
        if self.config.use_semantic_group_features:
            required_inputs.append(("semantic_group_features", semantic_group_features))
        missing = [name for name, value in required_inputs if value is None]
        if missing:
            raise ValueError(
                "learned semantic MoE requires all identity sidecar inputs: "
                + ", ".join(missing)
            )
        return self.learned_semantic_moe(
            champion_embedding=identity_components["champion"],
            role_embedding=identity_components["role"],
            build_embedding=identity_components["build"],
            fused_identity=identity_components["fused"],
            static=cast(torch.Tensor, identity_static_sidecar),
            full_game=cast(torch.Tensor, identity_full_game_sidecar),
            temporal=cast(torch.Tensor, identity_temporal_sidecar),
            support=cast(torch.Tensor, identity_encoder_support),
            semantic_group_features=semantic_group_features,
        )

    def semantic_moe_regularization_loss(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.learned_semantic_moe is not None:
            return self.learned_semantic_moe.regularization_from_outputs(outputs)
        final_logit = outputs.get("final_logit")
        if final_logit is None:
            raise ValueError("outputs must include final_logit")
        return final_logit.new_zeros(())

    def semantic_moe_stats(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.learned_semantic_moe is None:
            return {}
        return self.learned_semantic_moe.stats_from_outputs(outputs)

    def _forward_impl(
        self,
        *,
        champion_id: torch.Tensor,
        build_id: torch.Tensor,
        mu_1vx: torch.Tensor,
        var_1vx: torch.Tensor,
        mu_2vx: torch.Tensor,
        mu_1v1: torch.Tensor,
        delta_logit_2vx: torch.Tensor | None = None,
        delta_logit_1v1: torch.Tensor | None = None,
        conf_1vx: torch.Tensor | None = None,
        log_count_1vx: torch.Tensor | None = None,
        conf_2vx: torch.Tensor | None = None,
        log_count_2vx: torch.Tensor | None = None,
        missing_2vx: torch.Tensor | None = None,
        conf_1v1: torch.Tensor | None = None,
        log_count_1v1: torch.Tensor | None = None,
        missing_1v1: torch.Tensor | None = None,
        identity_static_sidecar: torch.Tensor | None = None,
        identity_full_game_sidecar: torch.Tensor | None = None,
        identity_temporal_sidecar: torch.Tensor | None = None,
        identity_encoder_support: torch.Tensor | None = None,
        semantic_group_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        relationships_enabled = bool(self.config.use_relationship_integrations)
        if relationships_enabled and (delta_logit_2vx is None or delta_logit_1v1 is None):
            features = relationship_logit_features(mu_1vx=mu_1vx, mu_2vx=mu_2vx, mu_1v1=mu_1v1)
            delta_logit_2vx = features["delta_logit_2vx"]
            delta_logit_1v1 = features["delta_logit_1v1"]

        # Node init: multiplicative identity (§3) fused with the 1vX posterior.
        identity_components = self.identity.components(champion_id, build_id)
        h0 = identity_components["fused"]
        sidecar = self._sidecar_node_features(
            champion_id=champion_id,
            identity_static_sidecar=identity_static_sidecar,
            identity_full_game_sidecar=identity_full_game_sidecar,
            identity_temporal_sidecar=identity_temporal_sidecar,
            identity_encoder_support=identity_encoder_support,
        )
        if sidecar is not None:
            h0 = h0 + sidecar
        phi_node = self.phi["1vx"](
            _logit(mu_1vx, self.config.logit_clip),
            var_1vx,
            conf_1vx,
            log_count_1vx,
        )
        h = self.node_norm(self.node_init(torch.cat([h0, phi_node], dim=-1)))

        a = self._readout(h[:, :5])
        b = self._readout(h[:, 5:])
        head_parts = [a, b, a - b, a * b]
        if relationships_enabled:
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
        else:
            head_parts.append(a.new_zeros(a.shape))
        base_logit = self.head(torch.cat(head_parts, dim=-1)).squeeze(-1)
        context_logit = self._semantic_context_logit(
            identity_static_sidecar=identity_static_sidecar,
            identity_full_game_sidecar=identity_full_game_sidecar,
            identity_temporal_sidecar=identity_temporal_sidecar,
            identity_encoder_support=identity_encoder_support,
        )
        if context_logit is None:
            context_logit = base_logit.new_zeros(base_logit.shape)
        moe_outputs = self._learned_semantic_moe_outputs(
            identity_components=identity_components,
            identity_static_sidecar=identity_static_sidecar,
            identity_full_game_sidecar=identity_full_game_sidecar,
            identity_temporal_sidecar=identity_temporal_sidecar,
            identity_encoder_support=identity_encoder_support,
            semantic_group_features=semantic_group_features,
        )
        if moe_outputs is not None:
            context_logit = context_logit + moe_outputs["semantic_moe_logit"]
            return {
                "base_logit": base_logit,
                "context_logit": context_logit,
                "final_logit": base_logit + context_logit,
                **moe_outputs,
            }
        return {
            "base_logit": base_logit,
            "context_logit": context_logit,
            "final_logit": base_logit + context_logit,
        }

    def forward(
        self,
        *,
        champion_id: torch.Tensor,
        build_id: torch.Tensor,
        mu_1vx: torch.Tensor,
        var_1vx: torch.Tensor,
        mu_2vx: torch.Tensor,
        mu_1v1: torch.Tensor,
        delta_logit_2vx: torch.Tensor | None = None,
        delta_logit_1v1: torch.Tensor | None = None,
        conf_1vx: torch.Tensor | None = None,
        log_count_1vx: torch.Tensor | None = None,
        conf_2vx: torch.Tensor | None = None,
        log_count_2vx: torch.Tensor | None = None,
        missing_2vx: torch.Tensor | None = None,
        conf_1v1: torch.Tensor | None = None,
        log_count_1v1: torch.Tensor | None = None,
        missing_1v1: torch.Tensor | None = None,
        identity_static_sidecar: torch.Tensor | None = None,
        identity_full_game_sidecar: torch.Tensor | None = None,
        identity_temporal_sidecar: torch.Tensor | None = None,
        identity_encoder_support: torch.Tensor | None = None,
        semantic_group_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "champion_id": champion_id,
            "build_id": build_id,
            "mu_1vx": mu_1vx,
            "var_1vx": var_1vx,
            "mu_2vx": mu_2vx,
            "mu_1v1": mu_1v1,
        }
        optional = {
            "delta_logit_2vx": delta_logit_2vx,
            "delta_logit_1v1": delta_logit_1v1,
            "conf_1vx": conf_1vx,
            "log_count_1vx": log_count_1vx,
            "conf_2vx": conf_2vx,
            "log_count_2vx": log_count_2vx,
            "missing_2vx": missing_2vx,
            "conf_1v1": conf_1v1,
            "log_count_1v1": log_count_1v1,
            "missing_1v1": missing_1v1,
            "identity_static_sidecar": identity_static_sidecar,
            "identity_full_game_sidecar": identity_full_game_sidecar,
            "identity_temporal_sidecar": identity_temporal_sidecar,
            "identity_encoder_support": identity_encoder_support,
            "semantic_group_features": semantic_group_features,
        }
        inputs.update(optional)
        filtered = {key: value for key, value in inputs.items() if value is not None}
        if self.config.structural_antisymmetry:
            direct = self._forward_impl(**filtered)
            swapped = self._forward_impl(**swap_hgnn_inputs(filtered))
            scale = float(self.config.structural_antisymmetry_scale)
            base_logit = scale * (direct["base_logit"] - swapped["base_logit"])
            context_logit = scale * (direct["context_logit"] - swapped["context_logit"])
            outputs = {
                "base_logit": base_logit,
                "context_logit": context_logit,
                "final_logit": base_logit + context_logit,
            }
            if "semantic_moe_logit" in direct and "semantic_moe_logit" in swapped:
                for key in (
                    "semantic_moe_balance_loss",
                    "semantic_moe_entropy_loss",
                    "semantic_moe_delta_l2_loss",
                    "semantic_moe_factor_orthogonality_loss",
                    "semantic_moe_factor_variance_loss",
                    "semantic_moe_regularization_loss",
                    "semantic_moe_router_entropy",
                    "semantic_moe_factor_norm",
                    "semantic_moe_factor_std_mean",
                    "semantic_moe_factor_std_min",
                    "semantic_moe_context_token_keep_fraction",
                    "semantic_moe_expert_usage",
                    "semantic_moe_expert_selected_fraction",
                    "semantic_moe_group_features_enabled",
                    "semantic_moe_group_feature_dim",
                    "semantic_moe_group_relationship_l2_loss",
                    "semantic_moe_group_relationship_coeff_norm",
                    "semantic_moe_group_relationship_context_norm",
                    "semantic_moe_group_relationship_enabled",
                ):
                    if key in direct and key in swapped:
                        outputs[key] = 0.5 * (direct[key] + swapped[key])
                outputs.update(
                    {
                        "semantic_moe_logit": scale
                        * (direct["semantic_moe_logit"] - swapped["semantic_moe_logit"]),
                        "semantic_moe_expert_logit": scale
                        * (
                            direct["semantic_moe_expert_logit"]
                            - swapped["semantic_moe_expert_logit"]
                        ),
                        "semantic_moe_group_relationship_logit": scale
                        * (
                            direct["semantic_moe_group_relationship_logit"]
                            - swapped["semantic_moe_group_relationship_logit"]
                        ),
                        "semantic_moe_slot_delta": direct["semantic_moe_slot_delta"],
                        "semantic_moe_group_relationship_slot_delta": direct[
                            "semantic_moe_group_relationship_slot_delta"
                        ],
                        "semantic_moe_router_probs": direct["semantic_moe_router_probs"],
                        "semantic_moe_topk_indices": direct["semantic_moe_topk_indices"],
                        "semantic_moe_topk_weights": direct["semantic_moe_topk_weights"],
                    }
                )
            return outputs
        return self._forward_impl(**filtered)


def _config_from_payload(payload: dict[str, Any]) -> HGNNConfig:
    raw_config = dict(payload.get("model_config", {}))
    if "use_relationship_integrations" not in raw_config:
        # Older model artifacts were trained with direct 1v1/2vX integration enabled
        # before the flag existed. Preserve their saved semantics on load; newly
        # trained models record the now-disabled default explicitly.
        raw_config["use_relationship_integrations"] = True
    allowed = {field.name for field in fields(HGNNConfig)}
    config_dict = {
        key: value
        for key, value in raw_config.items()
        if key in allowed
    }
    for key in (
        "build_vocab",
        "value_hidden",
        "gate_hidden",
        "node_init_hidden",
        "readout_hidden",
        "residual_head_hidden",
        "team_slot_readout_hidden",
        "identity_encoder_sidecar_hidden",
        "semantic_context_hidden",
        "semantic_moe_factor_hidden",
        "semantic_moe_router_hidden",
        "semantic_moe_expert_hidden",
        "semantic_group_relationship_hidden",
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
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    strength = float(payload.get("confidence_strength", 30.0))
    return model, config, strength


__all__ = [
    "HGNNConfig",
    "HGNNWinModel",
    "IdentityEncoder",
    "IdentitySemanticContextHead",
    "LearnedSemanticMoEHead",
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
