from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from app.ml.config import N_SIDES, SIDE_BLUE, SIDE_RED, ModelConfig
from app.ml.dataset import Vocab
from app.ml.utils.attention_diagnostics import (
    attention_example_slice,
    attention_layer_stats,
    summarise_attention_layers,
)

# Fixed 10 player tokens (5 blue + 5 red).
N_PLAYER_TOKENS = 10
N_TEAM_TOKENS = 5
# Pooling splits the encoder's player tokens by side, pools each team
# independently into (blue_repr, red_repr), and builds the 5-way comparison
# (b, r, b-r, |b-r|, b*r) used as the head input. Blue/red swap symmetry is
# enforced at the head via antisymmetrization:
#   logit = head(b, r, b-r, |b-r|, b*r) - head(r, b, r-b, |b-r|, b*r)
# so logit(b, r) = -logit(r, b) exactly, by construction.


class _DiagnosticEncoderLayer(nn.Module):
    """TransformerEncoderLayer-equivalent layer with optional attention stats.

    Normal training uses PyTorch's MultiheadAttention SDPA fast path; diagnostic
    collection keeps the manual attention path so attention maps remain
    inspectable.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        attention_dropout: float,
        head_dropout: float,
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
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.head_dropout = float(head_dropout)

    def _apply_head_dropout(self, attn_probs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.head_dropout <= 0.0:
            return attn_probs
        keep_prob = 1.0 - self.head_dropout
        keep = (
            torch.rand(
                attn_probs.shape[0],
                attn_probs.shape[1],
                1,
                1,
                device=attn_probs.device,
            )
            < keep_prob
        )
        return attn_probs * keep.to(attn_probs.dtype) / keep_prob

    def _self_attention_manual(
        self,
        x: torch.Tensor,
        collect_attention_diagnostics: bool,
        attention_diagnostics_sample_size: int | None,
        attention_player_token_count: int,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        batch, seq_len, d_model = x.shape
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = attn_logits.softmax(dim=-1)
        retained_attn_probs = F.dropout(
            attn_probs,
            p=float(self.self_attn.dropout),
            training=self.training,
        )
        retained_attn_probs = self._apply_head_dropout(retained_attn_probs)
        stats = None
        if collect_attention_diagnostics:
            stats = attention_layer_stats(
                attention_example_slice(attn_probs, attention_diagnostics_sample_size),
                player_token_count=attention_player_token_count,
            )

        out = retained_attn_probs @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.self_attn.out_proj(out), stats

    def _sa_block(
        self,
        x: torch.Tensor,
        collect_attention_diagnostics: bool,
        attention_diagnostics_sample_size: int | None,
        attention_player_token_count: int,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        use_manual_attention = collect_attention_diagnostics or (
            self.training and self.head_dropout > 0.0
        )
        if use_manual_attention:
            out, stats = self._self_attention_manual(
                x,
                collect_attention_diagnostics,
                attention_diagnostics_sample_size,
                attention_player_token_count,
            )
        else:
            out = self.self_attn(x, x, x, need_weights=False)[0]
            stats = None
        return self.dropout1(out), stats

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.gelu(self.linear1(x), approximate="tanh")))
        return self.dropout2(x)

    def forward(
        self,
        x: torch.Tensor,
        collect_attention_diagnostics: bool = False,
        attention_diagnostics_sample_size: int | None = None,
        attention_player_token_count: int = 0,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        sa_out, stats = self._sa_block(
            self.norm1(x),
            collect_attention_diagnostics,
            attention_diagnostics_sample_size,
            attention_player_token_count,
        )
        x = x + sa_out
        x = x + self._ff_block(self.norm2(x))
        return x, stats


class _DiagnosticEncoder(nn.Module):
    """TransformerEncoder-equivalent with optional per-layer diagnostics."""

    def __init__(
        self,
        layer: _DiagnosticEncoderLayer,
        n_layers: int,
    ):
        super().__init__()
        from copy import deepcopy

        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        collect_attention_diagnostics: bool = False,
        attention_diagnostics_sample_size: int | None = None,
        attention_player_token_count: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        layer_stats: list[dict[str, object]] = []
        for layer in self.layers:
            x, stats = layer(
                x,
                collect_attention_diagnostics=collect_attention_diagnostics,
                attention_diagnostics_sample_size=attention_diagnostics_sample_size,
                attention_player_token_count=attention_player_token_count,
            )
            if stats is not None:
                layer_stats.append(stats)
        if collect_attention_diagnostics:
            return x, summarise_attention_layers(layer_stats)
        return x


class HybridTokenModel(nn.Module):
    """Champion set transformer over 10 player tokens.

    Each player token is the compositional sum of champion + role + build + side
    embeddings. The model is PyTorch-native so current CUDA wheels can select
    optimized scaled-dot-product attention kernels for NVIDIA Blackwell GPUs.
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
        self.register_buffer(
            "_player_side",
            torch.tensor([[SIDE_BLUE] * 5 + [SIDE_RED] * 5], dtype=torch.long),
            persistent=False,
        )

        self.input_norm = nn.LayerNorm(d)
        self.input_dropout = nn.Dropout(cfg.dropout)

        encoder_layer = _DiagnosticEncoderLayer(
            d_model=d,
            n_heads=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            head_dropout=cfg.head_dropout,
        )
        self.encoder = _DiagnosticEncoder(
            encoder_layer,
            cfg.n_layers,
        )

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

    def _player_tokens(
        self,
        champion_idx: torch.Tensor,
        role_idx: torch.Tensor,
        build_idx: torch.Tensor,
    ) -> torch.Tensor:
        b = champion_idx.shape[0]
        side_idx = self._player_side.expand(b, -1)
        return (
            self.champ_emb(champion_idx)
            + self.role_emb(role_idx)
            + self.build_emb(build_idx)
            + self.side_emb(side_idx)
        )

    def _team_pool(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            blue_repr = torch.sum(blue_tokens * blue_weights.unsqueeze(-1), dim=1)
            red_repr = torch.sum(red_tokens * red_weights.unsqueeze(-1), dim=1)
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
        return_attention_diagnostics: bool = False,
        attention_diagnostics_sample_size: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        player = self._player_tokens(champion_idx, role_idx, build_idx)
        x = self.input_dropout(self.input_norm(player))
        encoded = self.encoder(
            x,
            collect_attention_diagnostics=return_attention_diagnostics,
            attention_diagnostics_sample_size=attention_diagnostics_sample_size,
            attention_player_token_count=N_PLAYER_TOKENS,
        )
        if return_attention_diagnostics:
            z, attention_diagnostics = encoded
        else:
            z = encoded
            attention_diagnostics = {}
        blue_repr, red_repr = self._team_pool(z)
        score_bvr = self.head(self._match_features(blue_repr, red_repr)).squeeze(-1)
        score_rvb = self.head(self._match_features(red_repr, blue_repr)).squeeze(-1)
        logits = score_bvr - score_rvb
        if return_attention_diagnostics:
            return logits, attention_diagnostics
        return logits


__all__ = ["HybridTokenModel"]
