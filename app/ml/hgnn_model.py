# pyright: reportPrivateImportUsage=false

"""Match-Outcome HGNN win-rate model.

Production uses identity embeddings on top of the 1vX player prior. Training
and inference share one model shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from app.core.utils.common import TEAM_PAIRS as TEAM_PAIRS
from app.ml.loadout_patch_features import (
    LOADOUT_SIGNED_FEATURE_INDICES,
    PATCH_SIGNED_FEATURE_INDICES,
)
from app.ml.semantic_group_features import SEMANTIC_GROUP_FEATURE_DIM

N_PLAYERS = 10
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
    semantic_moe_num_experts: int = 128
    semantic_moe_top_k: int = 32
    semantic_moe_factor_dim: int = 64
    semantic_moe_factor_hidden: tuple[int, ...] = (128,)
    semantic_moe_router_hidden: tuple[int, ...] = (128,)
    semantic_moe_expert_hidden: tuple[int, ...] = (64,)
    semantic_moe_dropout: float = 0.0
    semantic_moe_context_token_dropout: float = 0.05
    semantic_moe_architecture: str = "convex_encoder_mix"
    semantic_moe_view_gate_hidden: tuple[int, ...] = (64,)
    semantic_moe_view_top_k: int = 2
    semantic_moe_view_router_noise: float = 0.01
    semantic_moe_view_balance_weight: float = 1.0e-2
    semantic_moe_view_entropy_weight: float = 1.0e-3
    semantic_moe_temperature: float = 1.0
    semantic_moe_support_strength: float = 30.0
    semantic_moe_balance_weight: float = 1.0e-2
    semantic_moe_entropy_weight: float = 1.0e-3
    semantic_moe_factor_orthogonality_weight: float = 1.0e-3
    semantic_moe_factor_variance_weight: float = 1.0e-3
    semantic_moe_factor_std_floor: float = 0.05
    semantic_moe_delta_l2_weight: float = 0.0
    semantic_moe_max_abs_slot_delta: float = 0.0
    use_semantic_group_features: bool = False
    semantic_group_feature_dim: int = SEMANTIC_GROUP_FEATURE_DIM
    semantic_group_relationship_hidden: tuple[int, ...] = (128,)
    semantic_group_relationship_dropout: float = 0.0
    semantic_group_relationship_l2_weight: float = 0.0
    loadout_feature_dim: int = 0
    patch_feature_dim: int = 0
    loadout_residual_hidden: tuple[int, ...] = (32,)
    loadout_residual_dropout: float = 0.0
    patch_residual_hidden: tuple[int, ...] = ()
    patch_residual_dropout: float = 0.0
    patch_residual_max_abs_logit: float = 0.15


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
    sees direct confidence and log support, plus a missing flag for absent support.
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
    identity_static_sidecar: Any | None = None,
    identity_full_game_sidecar: Any | None = None,
    identity_temporal_sidecar: Any | None = None,
    identity_encoder_support: Any | None = None,
    semantic_group_features: Any | None = None,
    loadout_features: Any | None = None,
    patch_features: Any | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Turn raw cache/prior arrays into the model's node/edge tensors.

    Single source of truth shared by training and the runtime predictor. Accepts
    numpy arrays or tensors. mu values stay in probability space (the model maps
    them to logits internally); champion/build ids become long embedding indices.
    """

    def to_tensor(arr: Any) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def to_long(arr: Any) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.long, device=device)

    count_1vx = to_tensor(p1_cnt)
    mu_1vx, var_1vx = posterior_mean_var(to_tensor(win_rate), count_1vx, strength)
    conf_1vx, log_count_1vx, _ = support_features(count_1vx, strength)
    inputs = {
        "champion_id": to_long(champion_id),
        "build_id": to_long(build_id),
        "mu_1vx": mu_1vx,
        "var_1vx": var_1vx,
        "conf_1vx": conf_1vx,
        "log_count_1vx": log_count_1vx,
    }
    for name, value in (
        ("identity_static_sidecar", identity_static_sidecar),
        ("identity_full_game_sidecar", identity_full_game_sidecar),
        ("identity_temporal_sidecar", identity_temporal_sidecar),
        ("identity_encoder_support", identity_encoder_support),
        ("semantic_group_features", semantic_group_features),
        ("loadout_features", loadout_features),
        ("patch_features", patch_features),
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

    def flip_signed_columns(
        x: torch.Tensor,
        signed_indices: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, features]")
        out = x.clone()
        if signed_indices:
            idx = torch.as_tensor(signed_indices, dtype=torch.long, device=x.device)
            out.index_copy_(1, idx, -x.index_select(1, idx))
        return out

    swapped = {
        "champion_id": swap_halves(inputs["champion_id"], 5),
        "build_id": swap_halves(inputs["build_id"], 5),
        "mu_1vx": swap_halves(inputs["mu_1vx"], 5),
        "var_1vx": swap_halves(inputs["var_1vx"], 5),
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
    if "loadout_features" in inputs:
        swapped["loadout_features"] = flip_signed_columns(
            inputs["loadout_features"],
            LOADOUT_SIGNED_FEATURE_INDICES,
            "loadout_features",
        )
    if "patch_features" in inputs:
        swapped["patch_features"] = flip_signed_columns(
            inputs["patch_features"],
            PATCH_SIGNED_FEATURE_INDICES,
            "patch_features",
        )
    for prefix in ("conf", "log_count"):
        key = f"{prefix}_1vx"
        if key in inputs:
            swapped[key] = swap_halves(inputs[key], 5)
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
        architecture = str(config.semantic_moe_architecture).strip().lower().replace("-", "_")
        aliases = {
            "convex_view_mix": "convex_encoder_mix",
            "view_mix": "convex_encoder_mix",
        }
        architecture = aliases.get(architecture, architecture)
        allowed_architectures = {"convex_encoder_mix"}
        if architecture not in allowed_architectures:
            known = ", ".join(sorted(allowed_architectures | set(aliases)))
            raise ValueError(f"unknown semantic_moe_architecture {architecture!r}; expected one of: {known}")
        self.architecture = architecture
        self.view_names = ("static", "full_game", "temporal")
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
        self.view_top_k = min(
            max(int(config.semantic_moe_view_top_k), 1),
            len(self.view_names),
        )
        self.view_router_noise = max(float(config.semantic_moe_view_router_noise), 0.0)
        self.view_balance_weight = float(config.semantic_moe_view_balance_weight)
        self.view_entropy_weight = float(config.semantic_moe_view_entropy_weight)
        self.balance_weight = float(config.semantic_moe_balance_weight)
        self.entropy_weight = float(config.semantic_moe_entropy_weight)
        self.factor_orthogonality_weight = float(
            config.semantic_moe_factor_orthogonality_weight
        )
        self.factor_variance_weight = float(config.semantic_moe_factor_variance_weight)
        self.factor_std_floor = float(config.semantic_moe_factor_std_floor)
        self.delta_l2_weight = float(config.semantic_moe_delta_l2_weight)
        self.max_abs_slot_delta = max(float(config.semantic_moe_max_abs_slot_delta), 0.0)
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

        def make_sidecar_factor(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                _mlp(
                    input_dim,
                    tuple(int(dim) for dim in config.semantic_moe_factor_hidden),
                    self.factor_dim,
                    dropout=dropout,
                ),
                nn.LayerNorm(self.factor_dim),
            )

        def make_factor(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                _mlp(
                    input_dim,
                    tuple(int(dim) for dim in config.semantic_moe_factor_hidden),
                    self.factor_dim,
                    dropout=dropout,
                ),
                nn.LayerNorm(self.factor_dim),
            )

        def make_experts() -> nn.ModuleList:
            experts = nn.ModuleList(
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
            for expert in experts:
                _zero_last_linear(expert)
            return experts

        self.sidecar_factor = make_sidecar_factor(sidecar_token_dim)
        self.factor = make_factor(token_dim)
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
        self.experts = make_experts()

        self.view_dims = {
            "static": self.static_dim,
            "full_game": self.full_game_dim,
            "temporal": self.temporal_dim,
        }
        uses_view_sidecar_factors = True
        uses_view_gate = True
        self.view_sidecar_factors = (
            nn.ModuleDict(
                {
                    name: make_sidecar_factor(int(dim) + 2)
                    for name, dim in self.view_dims.items()
                }
            )
            if uses_view_sidecar_factors
            else None
        )
        view_gate_dim = identity_token_dim + self.factor_dim * len(self.view_names) + group_token_dim + 2
        self.view_gate = (
            _mlp(
                view_gate_dim,
                tuple(int(dim) for dim in config.semantic_moe_view_gate_hidden),
                len(self.view_names),
                dropout=dropout,
            )
            if uses_view_gate
            else None
        )
        if self.view_gate is not None:
            _zero_last_linear(self.view_gate)

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

    def _group_context_token(
        self,
        group_features: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.group_context is None:
            return None
        if group_features is None:
            raise ValueError("semantic group context requires group features")
        return self.group_context(self._group_context_features(group_features))

    def _semantic_factor_token(
        self,
        *,
        champion_embedding: torch.Tensor,
        role_embedding: torch.Tensor,
        build_embedding: torch.Tensor,
        fused_identity: torch.Tensor,
        context_features: torch.Tensor,
        group_token: torch.Tensor | None,
        confidence: torch.Tensor,
        log_support: torch.Tensor,
    ) -> torch.Tensor:
        token_parts = [
            champion_embedding,
            role_embedding,
            build_embedding,
            fused_identity,
            context_features,
        ]
        if group_token is not None:
            token_parts.append(group_token)
        token_parts.extend([confidence.unsqueeze(-1), log_support.unsqueeze(-1)])
        return torch.cat(token_parts, dim=-1)

    def _relationship_token(
        self,
        *,
        champion_embedding: torch.Tensor,
        role_embedding: torch.Tensor,
        build_embedding: torch.Tensor,
        fused_identity: torch.Tensor,
        semantic_factor: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                champion_embedding,
                role_embedding,
                build_embedding,
                fused_identity,
                semantic_factor,
            ],
            dim=-1,
        )

    def _route_experts(
        self,
        *,
        semantic_factor: torch.Tensor,
        fused_identity: torch.Tensor,
        router: nn.Module,
        experts: nn.ModuleList,
    ) -> dict[str, torch.Tensor]:
        router_input = torch.cat([semantic_factor, fused_identity], dim=-1)
        router_logits = router(router_input) / self.temperature
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1)
        route_probs = router_logits.new_zeros(router_logits.shape)
        route_probs.scatter_(-1, top_indices, top_weights)

        expert_outputs = torch.cat([expert(semantic_factor) for expert in experts], dim=-1)
        slot_delta_raw = (expert_outputs * route_probs).sum(dim=-1)

        expert_usage = route_probs.mean(dim=(0, 1))
        selected_fraction = (route_probs > 0.0).to(dtype=route_probs.dtype).mean(dim=(0, 1))
        target_usage = route_probs.new_full((self.num_experts,), 1.0 / float(self.num_experts))
        balance_loss = (expert_usage - target_usage).pow(2).mean()
        safe_probs = route_probs.clamp_min(1.0e-12)
        entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)
        max_entropy = router_logits.new_tensor(float(self.top_k)).log()
        entropy_loss = (max_entropy - entropy).clamp_min(0.0).mean()
        return {
            "slot_delta_raw": slot_delta_raw,
            "route_probs": route_probs,
            "top_indices": top_indices,
            "top_weights": top_weights,
            "expert_usage": expert_usage,
            "selected_fraction": selected_fraction,
            "router_entropy": entropy.mean(),
            "balance_loss": balance_loss,
            "entropy_loss": entropy_loss,
        }

    def _view_sidecar_factors(
        self,
        *,
        static: torch.Tensor,
        full_game: torch.Tensor,
        temporal: torch.Tensor,
        confidence: torch.Tensor,
        log_support: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self.view_sidecar_factors is None:
            raise ValueError("view sidecar factors are not enabled for this architecture")
        view_inputs = {
            "static": static,
            "full_game": full_game,
            "temporal": temporal,
        }
        factors: dict[str, torch.Tensor] = {}
        keep_fractions: list[torch.Tensor] = []
        for name in self.view_names:
            sidecar_token = torch.cat(
                [
                    view_inputs[name],
                    confidence.unsqueeze(-1),
                    log_support.unsqueeze(-1),
                ],
                dim=-1,
            )
            factor = self.view_sidecar_factors[name](sidecar_token)
            factor, keep_fraction = self._mask_context_tokens(factor)
            factors[name] = factor
            keep_fractions.append(keep_fraction)
        token_keep_fraction = torch.stack(keep_fractions).mean()
        return factors, token_keep_fraction

    def _view_weights(
        self,
        *,
        champion_embedding: torch.Tensor,
        role_embedding: torch.Tensor,
        build_embedding: torch.Tensor,
        fused_identity: torch.Tensor,
        view_factors: dict[str, torch.Tensor],
        group_token: torch.Tensor | None,
        confidence: torch.Tensor,
        log_support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.view_gate is None:
            raise ValueError("view gate is not enabled for this architecture")
        gate_parts = [
            champion_embedding,
            role_embedding,
            build_embedding,
            fused_identity,
            *(view_factors[name] for name in self.view_names),
        ]
        if group_token is not None:
            gate_parts.append(group_token)
        gate_parts.extend([confidence.unsqueeze(-1), log_support.unsqueeze(-1)])
        view_logits = self.view_gate(torch.cat(gate_parts, dim=-1)) / self.temperature
        if self.training and self.view_router_noise > 0.0:
            view_logits = view_logits + torch.randn_like(view_logits) * self.view_router_noise
        weights = torch.softmax(view_logits, dim=-1)
        top_weights, top_indices = weights.topk(self.view_top_k, dim=-1)

        usage = weights.mean(dim=(0, 1))
        selected_fraction = (weights > 0.0).to(dtype=weights.dtype).mean(dim=(0, 1))
        target_usage = weights.new_full((len(self.view_names),), 1.0 / float(len(self.view_names)))
        balance_loss = (usage - target_usage).pow(2).mean()
        safe_weights = weights.clamp_min(1.0e-12)
        entropy = -(safe_weights * safe_weights.log()).sum(dim=-1)
        max_entropy = view_logits.new_tensor(float(len(self.view_names))).log()
        entropy_loss = (max_entropy - entropy).clamp_min(0.0).mean()
        return {
            "weights": weights,
            "top_indices": top_indices,
            "top_weights": top_weights,
            "usage": usage,
            "selected_fraction": selected_fraction,
            "entropy": entropy.mean(),
            "balance_loss": balance_loss,
            "entropy_loss": entropy_loss,
        }

    def _moe_return(
        self,
        *,
        logit: torch.Tensor,
        slot_delta: torch.Tensor,
        expert_slot_delta: torch.Tensor,
        group_relationship: dict[str, torch.Tensor],
        route: dict[str, torch.Tensor],
        semantic_factor: torch.Tensor,
        token_keep_fraction: torch.Tensor,
        delta_l2_loss: torch.Tensor,
        additional_regularization_loss: torch.Tensor | None = None,
        extra: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.max_abs_slot_delta > 0.0:
            cap = float(self.max_abs_slot_delta)
            slot_delta = cap * torch.tanh(slot_delta / cap)
            logit = slot_delta[:, :5].mean(dim=1) - slot_delta[:, 5:].mean(dim=1)
        factor_orthogonality_loss, factor_variance_loss, factor_std_mean, factor_std_min = (
            self._factor_regularization(semantic_factor)
        )
        if additional_regularization_loss is None:
            additional_regularization_loss = logit.new_zeros(())
        regularization_loss = (
            self.balance_weight * route["balance_loss"]
            + self.entropy_weight * route["entropy_loss"]
            + self.factor_orthogonality_weight * factor_orthogonality_loss
            + self.factor_variance_weight * factor_variance_loss
            + self.delta_l2_weight * delta_l2_loss
            + self.group_relationship_l2_weight * group_relationship["coeff_l2_loss"]
            + additional_regularization_loss
        )
        out = {
            "semantic_moe_logit": logit,
            "semantic_moe_slot_delta": slot_delta,
            "semantic_moe_expert_logit": (
                expert_slot_delta[:, :5].mean(dim=1)
                - expert_slot_delta[:, 5:].mean(dim=1)
            ),
            "semantic_moe_group_relationship_logit": group_relationship["logit"],
            "semantic_moe_group_relationship_slot_delta": group_relationship["slot_delta"],
            "semantic_moe_router_probs": route["route_probs"],
            "semantic_moe_topk_indices": route["top_indices"],
            "semantic_moe_topk_weights": route["top_weights"],
            "semantic_moe_expert_usage": route["expert_usage"],
            "semantic_moe_expert_selected_fraction": route["selected_fraction"],
            "semantic_moe_router_entropy": route["router_entropy"],
            "semantic_moe_factor_norm": semantic_factor.norm(dim=-1).mean(),
            "semantic_moe_balance_loss": route["balance_loss"],
            "semantic_moe_entropy_loss": route["entropy_loss"],
            "semantic_moe_factor_orthogonality_loss": factor_orthogonality_loss,
            "semantic_moe_factor_variance_loss": factor_variance_loss,
            "semantic_moe_factor_std_mean": factor_std_mean,
            "semantic_moe_factor_std_min": factor_std_min,
            "semantic_moe_context_token_keep_fraction": token_keep_fraction,
            "semantic_moe_delta_l2_loss": delta_l2_loss,
            "semantic_moe_slot_delta_max_abs": slot_delta.detach().abs().max(),
            "semantic_moe_max_abs_slot_delta": logit.new_tensor(
                float(self.max_abs_slot_delta)
            ),
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
        if extra:
            out.update(extra)
        return out

    def _view_diagnostics(
        self,
        gate: dict[str, torch.Tensor],
        *,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "semantic_moe_view_usage": gate["usage"],
            "semantic_moe_view_selected_fraction": gate["selected_fraction"],
            "semantic_moe_view_entropy": gate["entropy"],
            "semantic_moe_view_balance_loss": gate["balance_loss"],
            "semantic_moe_view_entropy_loss": gate["entropy_loss"],
            "semantic_moe_view_top_k": reference.new_tensor(float(self.view_top_k)),
            "semantic_moe_convex_encoder_mix_enabled": reference.new_tensor(
                1.0 if self.architecture == "convex_encoder_mix" else 0.0
            ),
        }

    def _fused_outputs(
        self,
        *,
        champion_embedding: torch.Tensor,
        role_embedding: torch.Tensor,
        build_embedding: torch.Tensor,
        fused_identity: torch.Tensor,
        static: torch.Tensor,
        full_game: torch.Tensor,
        temporal: torch.Tensor,
        confidence: torch.Tensor,
        log_support: torch.Tensor,
        group_features: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        group_token = self._group_context_token(group_features)
        sidecar_token = torch.cat(
            [static, full_game, temporal, confidence.unsqueeze(-1), log_support.unsqueeze(-1)],
            dim=-1,
        )
        sidecar_factor = self.sidecar_factor(sidecar_token)
        sidecar_factor, token_keep_fraction = self._mask_context_tokens(sidecar_factor)
        context_features = self._context_features(sidecar_factor, confidence)
        token = self._semantic_factor_token(
            champion_embedding=champion_embedding,
            role_embedding=role_embedding,
            build_embedding=build_embedding,
            fused_identity=fused_identity,
            context_features=context_features,
            group_token=group_token,
            confidence=confidence,
            log_support=log_support,
        )
        semantic_factor = self.factor(token)
        group_relationship = self._group_relationship_outputs(
            identity_token=self._relationship_token(
                champion_embedding=champion_embedding,
                role_embedding=role_embedding,
                build_embedding=build_embedding,
                fused_identity=fused_identity,
                semantic_factor=semantic_factor,
            ),
            group_features=group_features,
            confidence=confidence,
        )
        route = self._route_experts(
            semantic_factor=semantic_factor,
            fused_identity=fused_identity,
            router=self.router,
            experts=self.experts,
        )
        slot_delta_raw = route["slot_delta_raw"]
        expert_slot_delta = self.response_scale * slot_delta_raw * confidence
        slot_delta = expert_slot_delta + group_relationship["slot_delta"]
        logit = slot_delta[:, :5].mean(dim=1) - slot_delta[:, 5:].mean(dim=1)
        return self._moe_return(
            logit=logit,
            slot_delta=slot_delta,
            expert_slot_delta=expert_slot_delta,
            group_relationship=group_relationship,
            route=route,
            semantic_factor=semantic_factor,
            token_keep_fraction=token_keep_fraction,
            delta_l2_loss=slot_delta_raw.pow(2).mean(),
        )

    def _convex_encoder_mix_outputs(
        self,
        *,
        champion_embedding: torch.Tensor,
        role_embedding: torch.Tensor,
        build_embedding: torch.Tensor,
        fused_identity: torch.Tensor,
        static: torch.Tensor,
        full_game: torch.Tensor,
        temporal: torch.Tensor,
        confidence: torch.Tensor,
        log_support: torch.Tensor,
        group_features: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        group_token = self._group_context_token(group_features)
        view_factors, token_keep_fraction = self._view_sidecar_factors(
            static=static,
            full_game=full_game,
            temporal=temporal,
            confidence=confidence,
            log_support=log_support,
        )
        view_gate = self._view_weights(
            champion_embedding=champion_embedding,
            role_embedding=role_embedding,
            build_embedding=build_embedding,
            fused_identity=fused_identity,
            view_factors=view_factors,
            group_token=group_token,
            confidence=confidence,
            log_support=log_support,
        )
        stacked_factors = torch.stack([view_factors[name] for name in self.view_names], dim=2)
        sidecar_factor = (stacked_factors * view_gate["weights"].unsqueeze(-1)).sum(dim=2)
        context_features = self._context_features(sidecar_factor, confidence)
        token = self._semantic_factor_token(
            champion_embedding=champion_embedding,
            role_embedding=role_embedding,
            build_embedding=build_embedding,
            fused_identity=fused_identity,
            context_features=context_features,
            group_token=group_token,
            confidence=confidence,
            log_support=log_support,
        )
        semantic_factor = self.factor(token)
        group_relationship = self._group_relationship_outputs(
            identity_token=self._relationship_token(
                champion_embedding=champion_embedding,
                role_embedding=role_embedding,
                build_embedding=build_embedding,
                fused_identity=fused_identity,
                semantic_factor=semantic_factor,
            ),
            group_features=group_features,
            confidence=confidence,
        )
        route = self._route_experts(
            semantic_factor=semantic_factor,
            fused_identity=fused_identity,
            router=self.router,
            experts=self.experts,
        )
        slot_delta_raw = route["slot_delta_raw"]
        expert_slot_delta = self.response_scale * slot_delta_raw * confidence
        slot_delta = expert_slot_delta + group_relationship["slot_delta"]
        logit = slot_delta[:, :5].mean(dim=1) - slot_delta[:, 5:].mean(dim=1)
        view_regularization = (
            self.view_balance_weight * view_gate["balance_loss"]
            + self.view_entropy_weight * view_gate["entropy_loss"]
        )
        return self._moe_return(
            logit=logit,
            slot_delta=slot_delta,
            expert_slot_delta=expert_slot_delta,
            group_relationship=group_relationship,
            route=route,
            semantic_factor=semantic_factor,
            token_keep_fraction=token_keep_fraction,
            delta_l2_loss=slot_delta_raw.pow(2).mean(),
            additional_regularization_loss=view_regularization,
            extra=self._view_diagnostics(view_gate, reference=logit),
        )

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

        common = {
            "champion_embedding": champion_embedding,
            "role_embedding": role_embedding,
            "build_embedding": build_embedding,
            "fused_identity": fused_identity,
            "static": static,
            "full_game": full_game,
            "temporal": temporal,
            "confidence": confidence,
            "log_support": log_support,
            "group_features": group_features,
        }
        return self._convex_encoder_mix_outputs(**common)


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
        self.head = _mlp(c.node_dim * 4, c.readout_hidden, 1, dropout=c.dropout)
        self.loadout_residual = (
            _mlp(
                int(c.loadout_feature_dim),
                c.loadout_residual_hidden,
                1,
                dropout=c.loadout_residual_dropout,
            )
            if int(c.loadout_feature_dim) > 0
            else None
        )
        self.patch_residual = (
            _mlp(
                int(c.patch_feature_dim),
                c.patch_residual_hidden,
                1,
                dropout=c.patch_residual_dropout,
            )
            if int(c.patch_feature_dim) > 0
            else None
        )
        if self.loadout_residual is not None:
            _zero_last_linear(self.loadout_residual)
        if self.patch_residual is not None:
            _zero_last_linear(self.patch_residual)

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

    def _game_feature_block(
        self,
        value: torch.Tensor | None,
        *,
        batch_size: int,
        dim: int,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if dim <= 0:
            return reference.new_zeros((batch_size, 0))
        if value is None:
            return reference.new_zeros((batch_size, dim))
        if value.ndim != 2 or value.shape[1] != dim:
            raise ValueError(f"{name} must have shape [batch, {dim}]")
        return value.to(dtype=reference.dtype)

    def _residual_feature_logit(
        self,
        head: nn.Module | None,
        value: torch.Tensor | None,
        *,
        batch_size: int,
        dim: int,
        reference: torch.Tensor,
        name: str,
        max_abs_logit: float | None = None,
    ) -> torch.Tensor:
        if head is None or dim <= 0:
            return reference.new_zeros((batch_size,))
        if value is None:
            return reference.new_zeros((batch_size,))
        features = self._game_feature_block(
            value,
            batch_size=batch_size,
            dim=dim,
            reference=reference,
            name=name,
        )
        logit = head(features).squeeze(-1)
        if max_abs_logit is not None and max_abs_logit > 0.0:
            scale = float(max_abs_logit)
            logit = scale * torch.tanh(logit / scale)
        return logit

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
        conf_1vx: torch.Tensor | None = None,
        log_count_1vx: torch.Tensor | None = None,
        identity_static_sidecar: torch.Tensor | None = None,
        identity_full_game_sidecar: torch.Tensor | None = None,
        identity_temporal_sidecar: torch.Tensor | None = None,
        identity_encoder_support: torch.Tensor | None = None,
        semantic_group_features: torch.Tensor | None = None,
        loadout_features: torch.Tensor | None = None,
        patch_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
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
        base_logit = self.head(torch.cat(head_parts, dim=-1)).squeeze(-1)
        batch_size = int(base_logit.shape[0])
        loadout_logit = self._residual_feature_logit(
            self.loadout_residual,
            loadout_features,
            batch_size=batch_size,
            dim=int(self.config.loadout_feature_dim),
            reference=base_logit,
            name="loadout_features",
        )
        patch_logit = self._residual_feature_logit(
            self.patch_residual,
            patch_features,
            batch_size=batch_size,
            dim=int(self.config.patch_feature_dim),
            reference=base_logit,
            name="patch_features",
            max_abs_logit=float(self.config.patch_residual_max_abs_logit),
        )
        feature_logit = loadout_logit + patch_logit
        context_logit = self._semantic_context_logit(
            identity_static_sidecar=identity_static_sidecar,
            identity_full_game_sidecar=identity_full_game_sidecar,
            identity_temporal_sidecar=identity_temporal_sidecar,
            identity_encoder_support=identity_encoder_support,
        )
        if context_logit is None:
            context_logit = base_logit.new_zeros(base_logit.shape)
        final_logit = base_logit + context_logit + feature_logit
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
            final_logit = base_logit + context_logit + feature_logit
            return {
                "base_logit": base_logit,
                "context_logit": context_logit,
                "loadout_logit": loadout_logit,
                "patch_logit": patch_logit,
                "feature_logit": feature_logit,
                "final_logit": final_logit,
                **moe_outputs,
            }
        return {
            "base_logit": base_logit,
            "context_logit": context_logit,
            "loadout_logit": loadout_logit,
            "patch_logit": patch_logit,
            "feature_logit": feature_logit,
            "final_logit": final_logit,
        }

    def forward(
        self,
        *,
        champion_id: torch.Tensor,
        build_id: torch.Tensor,
        mu_1vx: torch.Tensor,
        var_1vx: torch.Tensor,
        conf_1vx: torch.Tensor | None = None,
        log_count_1vx: torch.Tensor | None = None,
        identity_static_sidecar: torch.Tensor | None = None,
        identity_full_game_sidecar: torch.Tensor | None = None,
        identity_temporal_sidecar: torch.Tensor | None = None,
        identity_encoder_support: torch.Tensor | None = None,
        semantic_group_features: torch.Tensor | None = None,
        loadout_features: torch.Tensor | None = None,
        patch_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "champion_id": champion_id,
            "build_id": build_id,
            "mu_1vx": mu_1vx,
            "var_1vx": var_1vx,
        }
        optional = {
            "conf_1vx": conf_1vx,
            "log_count_1vx": log_count_1vx,
            "identity_static_sidecar": identity_static_sidecar,
            "identity_full_game_sidecar": identity_full_game_sidecar,
            "identity_temporal_sidecar": identity_temporal_sidecar,
            "identity_encoder_support": identity_encoder_support,
            "semantic_group_features": semantic_group_features,
            "loadout_features": loadout_features,
            "patch_features": patch_features,
        }
        inputs.update(optional)
        filtered = {key: value for key, value in inputs.items() if value is not None}
        if self.config.structural_antisymmetry:
            direct = self._forward_impl(**filtered)
            swapped = self._forward_impl(**swap_hgnn_inputs(filtered))
            scale = float(self.config.structural_antisymmetry_scale)
            base_logit = scale * (direct["base_logit"] - swapped["base_logit"])
            context_logit = scale * (direct["context_logit"] - swapped["context_logit"])
            loadout_logit = scale * (direct["loadout_logit"] - swapped["loadout_logit"])
            patch_logit = scale * (direct["patch_logit"] - swapped["patch_logit"])
            feature_logit = loadout_logit + patch_logit
            outputs = {
                "base_logit": base_logit,
                "context_logit": context_logit,
                "loadout_logit": loadout_logit,
                "patch_logit": patch_logit,
                "feature_logit": feature_logit,
                "final_logit": base_logit + context_logit + feature_logit,
            }
            if "semantic_moe_logit" in direct and "semantic_moe_logit" in swapped:
                for key in (
                    "semantic_moe_balance_loss",
                    "semantic_moe_entropy_loss",
                    "semantic_moe_delta_l2_loss",
                    "semantic_moe_slot_delta_max_abs",
                    "semantic_moe_max_abs_slot_delta",
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
                    "semantic_moe_view_entropy",
                    "semantic_moe_view_balance_loss",
                    "semantic_moe_view_entropy_loss",
                    "semantic_moe_view_top_k",
                    "semantic_moe_convex_encoder_mix_enabled",
                ):
                    if key in direct and key in swapped:
                        outputs[key] = 0.5 * (direct[key] + swapped[key])
                for key in (
                    "semantic_moe_view_usage",
                    "semantic_moe_view_selected_fraction",
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
        "team_slot_readout_hidden",
        "identity_encoder_sidecar_hidden",
        "semantic_context_hidden",
        "semantic_moe_factor_hidden",
        "semantic_moe_router_hidden",
        "semantic_moe_expert_hidden",
        "semantic_group_relationship_hidden",
        "loadout_residual_hidden",
        "patch_residual_hidden",
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
    if payload.get("model_type") != "hgnn":
        raise ValueError(f"checkpoint is not an HGNN artifact: {path}")
    if "state_dict" not in payload:
        raise ValueError(f"HGNN checkpoint is missing state_dict: {path}")
    config = _config_from_payload(payload)
    model = HGNNWinModel(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
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
    "resolve_device",
    "save_hgnn_model",
    "support_features",
    "swap_hgnn_inputs",
]
