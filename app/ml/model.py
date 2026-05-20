from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from app.ml.config import N_SIDES, SIDE_BLUE, SIDE_RED, ModelConfig
from app.ml.dataset import Vocab

# Fixed 10 player tokens (5 blue + 5 red).
N_PLAYER_TOKENS = 10
N_TEAM_TOKENS = 5
# Pooling splits the encoder's player tokens by side, pools each team
# independently into (blue_repr, red_repr), and builds the 5-way comparison
# (b, r, b-r, |b-r|, b*r) used as the head input. Blue/red swap symmetry is
# enforced at the head via antisymmetrization:
#   logit = head(b, r, b-r, |b-r|, b*r) - head(r, b, r-b, |b-r|, b*r)
# so logit(b, r) = -logit(r, b) exactly, by construction.


class _EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.gelu(self.linear1(x), approximate="tanh")))
        return self.dropout2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.dropout1(attn_out)
        x = x + self._ff_block(self.norm2(x))
        return x


class _Encoder(nn.Module):
    def __init__(self, layer: _EncoderLayer, n_layers: int):
        super().__init__()
        from copy import deepcopy

        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _MatchLevelMoEHead(nn.Module):
    """Learned match-level MoE head with a dense-head skip."""

    def __init__(
        self,
        *,
        match_dim: int,
        cfg: ModelConfig,
    ):
        super().__init__()
        if cfg.n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if cfg.moe_top_k <= 0:
            raise ValueError("moe_top_k must be positive")
        if cfg.router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive")
        if not 0.0 <= cfg.moe_dropout <= 1.0:
            raise ValueError("moe_dropout must be in [0, 1]")
        if cfg.expert_hidden <= 0:
            raise ValueError("expert_hidden must be positive")
        if cfg.router_hidden <= 0:
            raise ValueError("router_hidden must be positive")

        self.n_experts = cfg.n_experts
        self.top_k = min(cfg.moe_top_k, cfg.n_experts)
        self.router_temperature = cfg.router_temperature
        self.aux_loss_coef = cfg.moe_aux_loss_coef
        self.warmup_steps = cfg.moe_warmup_steps
        self.router_noise = cfg.moe_router_noise

        expert_input_dim = match_dim + 3
        self.experts = nn.ModuleList(
            self._make_expert(expert_input_dim, cfg.expert_hidden, cfg.moe_dropout)
            for _ in range(cfg.n_experts)
        )
        self.router = nn.Sequential(
            nn.LayerNorm(expert_input_dim),
            nn.Linear(expert_input_dim, cfg.router_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(cfg.moe_dropout),
            nn.Linear(cfg.router_hidden, cfg.n_experts),
        )
        router_final = cast(nn.Linear, self.router[-1])
        nn.init.zeros_(router_final.weight)
        nn.init.zeros_(router_final.bias)

    @staticmethod
    def _make_expert(
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        expert = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        final_layer = cast(nn.Linear, expert[-1])
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        return expert

    def forward(
        self,
        match_features: torch.Tensor,
        dense_score: torch.Tensor,
        route_logit: torch.Tensor,
        dense_routing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score, aux, _ = self._forward(
            match_features,
            dense_score,
            route_logit,
            dense_routing=dense_routing,
            return_diagnostics=False,
        )
        return score, aux

    def forward_with_diagnostics(
        self,
        match_features: torch.Tensor,
        dense_score: torch.Tensor,
        route_logit: torch.Tensor,
        dense_routing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        score, aux, diagnostics = self._forward(
            match_features,
            dense_score,
            route_logit,
            dense_routing=dense_routing,
            return_diagnostics=True,
        )
        return score, aux, cast(dict[str, torch.Tensor], diagnostics)

    def _forward(
        self,
        match_features: torch.Tensor,
        dense_score: torch.Tensor,
        route_logit: torch.Tensor,
        *,
        dense_routing: bool,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        route_logit = route_logit.float()
        baseline_prob = route_logit.sigmoid()
        scalar_features = torch.cat(
            [
                route_logit.unsqueeze(-1),
                baseline_prob.unsqueeze(-1),
                (baseline_prob - 0.5).abs().unsqueeze(-1),
            ],
            dim=-1,
        )

        expert_input = torch.cat(
            [match_features, scalar_features.to(match_features.dtype)],
            dim=-1,
        )
        routing_logits = self.router(expert_input).float() / self.router_temperature
        if self.training and not dense_routing and self.router_noise > 0.0:
            routing_logits = routing_logits + self.router_noise * torch.randn_like(
                routing_logits
            )
        full_probs = routing_logits.softmax(dim=-1)
        k = self.n_experts if dense_routing else self.top_k
        top_values, top_indices = routing_logits.topk(k, dim=-1)
        top_weights = top_values.softmax(dim=-1)

        # Switch-style load-balancing aux loss: P_i = mean router prob,
        # f_i = fraction of (token, slot) assignments to expert i.
        counts = full_probs.new_zeros(self.n_experts).scatter_add_(
            0, top_indices.reshape(-1), full_probs.new_ones(top_indices.numel())
        )
        f = counts / top_indices.numel()
        P = full_probs.mean(dim=0)
        aux_loss = self.aux_loss_coef * self.n_experts * (f * P).sum()

        selected_scores = expert_input.new_zeros(top_indices.shape)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = (top_indices == expert_id).nonzero(as_tuple=True)
            if rows.numel() > 0:
                selected_scores[rows, slots] = expert(expert_input[rows]).squeeze(
                    -1
                ).to(selected_scores.dtype)
        moe_score = (selected_scores * top_weights.to(selected_scores.dtype)).sum(
            dim=-1
        )
        diagnostics = None
        if return_diagnostics:
            expert_weights = routing_logits.new_zeros(routing_logits.shape).scatter(
                -1, top_indices, top_weights
            )
            expert_corrections = routing_logits.new_zeros(routing_logits.shape).scatter(
                -1, top_indices, (selected_scores.float() * top_weights)
            )
            diagnostics = {
                "expert_weights": expert_weights,
                "router_probs": full_probs,
                "expert_corrections": expert_corrections,
            }
        return dense_score + moe_score, aux_loss, diagnostics


class HybridTokenModel(nn.Module):
    """Champion set transformer over 10 player tokens.

    Each player token is the compositional sum of champion + role + build + side
    embeddings plus the 6002-derived historical profile projection for that
    token across scaling bins.
    """

    def __init__(
        self,
        vocab: Vocab,
        cfg: ModelConfig,
    ):
        super().__init__()
        d = cfg.d_model

        self.champ_emb = nn.Embedding(vocab.n_champions, d)
        self.role_emb = nn.Embedding(vocab.n_roles, d)
        self.build_emb = nn.Embedding(vocab.n_builds, d)
        self.side_emb = nn.Embedding(N_SIDES, d)
        profile_input_dim = vocab.n_profile_bins * vocab.n_profile_features
        if profile_input_dim <= 0:
            raise ValueError("profile_input_dim must be positive")
        self.profile_projection = nn.Sequential(
            nn.LayerNorm(profile_input_dim),
            nn.Linear(profile_input_dim, d, bias=False),
        )
        self._player_side: torch.Tensor
        self.register_buffer(
            "_player_side",
            torch.tensor([[SIDE_BLUE] * 5 + [SIDE_RED] * 5], dtype=torch.long),
            persistent=False,
        )

        self.input_norm = nn.LayerNorm(d)
        self.input_dropout = nn.Dropout(cfg.dropout)

        encoder_layer = _EncoderLayer(
            d_model=d,
            n_heads=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
        )
        self.encoder = _Encoder(encoder_layer, cfg.n_layers)

        self.pooling = cfg.pooling
        self.team_attention_pool = (
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1, bias=False))
            if cfg.pooling == "team_attention"
            else None
        )
        head_input_dim = d * 5
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, cfg.head_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.moe_head = (
            _MatchLevelMoEHead(match_dim=head_input_dim, cfg=cfg)
            if cfg.use_moe
            else None
        )

    def _player_tokens(
        self,
        champion_idx: torch.Tensor,
        role_idx: torch.Tensor,
        build_idx: torch.Tensor,
        player_profile: torch.Tensor,
    ) -> torch.Tensor:
        b = champion_idx.shape[0]
        side_idx = self._player_side.expand(b, -1)
        profile = player_profile.reshape(b, N_PLAYER_TOKENS, -1)
        profile = profile.to(self.champ_emb.weight.dtype)
        return (
            self.champ_emb(champion_idx)
            + self.role_emb(role_idx)
            + self.build_emb(build_idx)
            + self.side_emb(side_idx)
            + self.profile_projection(profile)
        )

    def _team_pool(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool 10 player tokens into separate blue/red team representations.

        `tokens` is [B, 10, d] with positions [0:5] carrying blue side and
        [5:10] carrying red (set by the fixed _player_side buffer during
        embedding). Returns (blue_repr, red_repr), both [B, d].
        """
        blue_tokens = tokens[:, :N_TEAM_TOKENS]
        red_tokens = tokens[:, N_TEAM_TOKENS:]
        if self.pooling == "team_attention":
            team_pool = cast(nn.Sequential, self.team_attention_pool)
            blue_weights = team_pool(blue_tokens).squeeze(-1).softmax(dim=-1)
            red_weights = team_pool(red_tokens).squeeze(-1).softmax(dim=-1)
            blue_repr = (blue_tokens * blue_weights.unsqueeze(-1)).sum(dim=1)
            red_repr = (red_tokens * red_weights.unsqueeze(-1)).sum(dim=1)
        else:
            blue_repr = blue_tokens.mean(dim=1)
            red_repr = red_tokens.mean(dim=1)
        return blue_repr, red_repr

    @staticmethod
    def _match_features(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """5-way comparison concat (first, second, first-second, |first-second|, first*second)."""
        diff = first - second
        return torch.cat(
            [first, second, diff, diff.abs(), first * second],
            dim=-1,
        )

    def forward(
        self,
        champion_idx: torch.Tensor,
        role_idx: torch.Tensor,
        build_idx: torch.Tensor,
        player_profile: torch.Tensor,
        dense_routing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, final_logit, aux_loss, _ = self._logit_parts(
            champion_idx,
            role_idx,
            build_idx,
            player_profile,
            dense_routing=dense_routing,
        )
        return final_logit, aux_loss

    def matched_diagnostic_tensors(
        self,
        champion_idx: torch.Tensor,
        role_idx: torch.Tensor,
        build_idx: torch.Tensor,
        player_profile: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        baseline_logit, final_logit, _, route_diagnostics = self._logit_parts(
            champion_idx,
            role_idx,
            build_idx,
            player_profile,
            include_route_weights=True,
        )
        output = {"baseline_logit": baseline_logit, "final_logit": final_logit}
        if route_diagnostics is not None:
            output.update(route_diagnostics)
        return output

    def _logit_parts(
        self,
        champion_idx: torch.Tensor,
        role_idx: torch.Tensor,
        build_idx: torch.Tensor,
        player_profile: torch.Tensor,
        *,
        dense_routing: bool = False,
        include_route_weights: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
    ]:
        player = self._player_tokens(
            champion_idx,
            role_idx,
            build_idx,
            player_profile,
        )
        x = self.input_dropout(self.input_norm(player))
        z = self.encoder(x)
        blue_repr, red_repr = self._team_pool(z)
        match_bvr = self._match_features(blue_repr, red_repr)
        match_rvb = self._match_features(red_repr, blue_repr)
        score_bvr = self.head(match_bvr).squeeze(-1)
        score_rvb = self.head(match_rvb).squeeze(-1)
        baseline_logit = score_bvr - score_rvb
        if self.moe_head is None:
            zero = baseline_logit.new_zeros(())
            return baseline_logit, baseline_logit, zero, None
        if include_route_weights:
            score_bvr, aux_bvr, bvr_diagnostics = (
                self.moe_head.forward_with_diagnostics(
                    match_bvr, score_bvr, baseline_logit, dense_routing
                )
            )
            score_rvb, aux_rvb, rvb_diagnostics = (
                self.moe_head.forward_with_diagnostics(
                    match_rvb, score_rvb, -baseline_logit, dense_routing
                )
            )
            route_diagnostics = {
                f"bvr_{key}": value for key, value in bvr_diagnostics.items()
            }
            route_diagnostics.update(
                {f"rvb_{key}": value for key, value in rvb_diagnostics.items()}
            )
        else:
            score_bvr, aux_bvr = self.moe_head(
                match_bvr, score_bvr, baseline_logit, dense_routing
            )
            score_rvb, aux_rvb = self.moe_head(
                match_rvb, score_rvb, -baseline_logit, dense_routing
            )
            route_diagnostics = None
        return (
            baseline_logit,
            score_bvr - score_rvb,
            aux_bvr + aux_rvb,
            route_diagnostics,
        )


__all__ = ["HybridTokenModel"]
