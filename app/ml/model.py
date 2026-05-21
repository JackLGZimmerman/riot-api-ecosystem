from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from app.ml.cache_layout import PROFILE_FEATURE_COLUMNS
from app.ml.config import N_SIDES, SIDE_BLUE, SIDE_RED, ModelConfig
from app.ml.dataset import Vocab

# Fixed 10 player tokens (5 blue + 5 red).
N_PLAYER_TOKENS = 10
N_TEAM_TOKENS = 5
LOG_MATCHUPS_FEATURE = "log_matchups"
LOG_MATCHUPS_FEATURE_IDX = PROFILE_FEATURE_COLUMNS.index(LOG_MATCHUPS_FEATURE)
# Pooling splits the encoder's player tokens by side, pools each team
# independently into (blue_repr, red_repr), and builds the comparison
# (b, r, b-r, |b-r|, b*r, lane) used as the head input, where lane is an
# attention-weighted summary of the 5 same-role blue-vs-red token diffs.
# Blue/red swap symmetry is enforced at the head via antisymmetrization:
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


class _TemporalProfileEncoder(nn.Module):
    """Encodes a token's 6002 historical profile across scaling bins.

    Input is [B, T, n_bins, n_features] of per-bin aggregates. `log_matchups`
    is reserved strictly for reliability: n = expm1(log_matchups), confidence
    = n / (n + k). The other profile features are standardized by per-feature
    train mean/std (so absolute levels and incommensurate units survive,
    unlike a per-row LayerNorm), encoded by a shared MLP (the GELU lets it
    model nonlinear trajectory shape, not just linear early/late trends),
    tagged with a learned bin-position embedding, then pooled across bins by
    presence-masked attention.

    The output is two-stage: the attention pool gives a `level` summary, and
    a presence-masked `late - early` half-mean delta gives the trajectory
    direction the pool's weighted sum structurally cannot represent. Both are
    fused into one [B, T, d] token, so the temporal axis survives into the
    match transformer's cross-player attention. `delta_proj` is zero-init, so
    a fresh model starts identical to a level-only encoder and learns the
    delta contribution from there.
    """

    def __init__(
        self,
        n_bins: int,
        n_features: int,
        d: int,
        dropout: float,
        feature_mean: Sequence[float],
        feature_std: Sequence[float],
        confidence_prior_count: float,
    ):
        super().__init__()
        if n_bins <= 0 or n_features <= 1:
            raise ValueError("n_bins must be positive and n_features must exceed 1")
        if n_features != len(PROFILE_FEATURE_COLUMNS):
            raise ValueError(
                "n_features must match PROFILE_FEATURE_COLUMNS so "
                f"{LOG_MATCHUPS_FEATURE!r} can be used only as confidence"
            )
        if len(feature_mean) != n_features or len(feature_std) != n_features:
            raise ValueError("feature_mean/feature_std must have n_features entries")
        if confidence_prior_count <= 0.0:
            raise ValueError("confidence_prior_count must be positive")
        content_feature_idx = [
            idx for idx in range(n_features) if idx != LOG_MATCHUPS_FEATURE_IDX
        ]
        self.bin_mlp = nn.Sequential(
            nn.Linear(len(content_feature_idx), d),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )
        self.bin_pos_emb = nn.Embedding(n_bins, d)
        self.attn_score = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 1, bias=False),
        )
        # Bins [0:split) are "early", [split:) are "late"; the delta is the
        # late-minus-early half-mean. Zero-init projection keeps the encoder
        # bit-identical to a level-only pool at start of training.
        self.bin_split = n_bins // 2
        self.delta_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )
        delta_linear = cast(nn.Linear, self.delta_proj[-1])
        nn.init.zeros_(delta_linear.weight)
        nn.init.zeros_(delta_linear.bias)
        self._bin_idx: torch.Tensor
        self.register_buffer(
            "_bin_idx", torch.arange(n_bins, dtype=torch.long), persistent=False
        )
        self.log_matchups_feature_idx = LOG_MATCHUPS_FEATURE_IDX
        self._content_feature_idx: torch.Tensor
        self.register_buffer(
            "_content_feature_idx",
            torch.tensor(content_feature_idx, dtype=torch.long),
            persistent=False,
        )
        # Persistent so a loaded checkpoint carries its own standardization.
        self._feature_mean: torch.Tensor
        self._feature_std: torch.Tensor
        self.register_buffer(
            "_feature_mean", torch.tensor(feature_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "_feature_std", torch.tensor(feature_std, dtype=torch.float32)
        )
        self._confidence_prior_count: torch.Tensor
        self.register_buffer(
            "_confidence_prior_count",
            torch.tensor(float(confidence_prior_count), dtype=torch.float32),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean of `values` over the bin axis, counting only present bins.

        `values` is [B, T, k, d], `mask` is [B, T, k]. Tokens with no present
        bin in the half divide by a clamped count of 1 and so return zeros.
        """
        weight = mask.to(values.dtype)
        total = (values * weight.unsqueeze(-1)).sum(dim=-2)
        count = weight.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return total / count

    def _profile_confidence(
        self,
        profile: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        """Sample-count reliability from raw `log_matchups`, not content signal."""
        log_matchups = profile[..., self.log_matchups_feature_idx].clamp(
            min=0.0,
            max=20.0,
        )
        support = torch.expm1(log_matchups)
        prior_count = self._confidence_prior_count.to(dtype=support.dtype)
        confidence = support / (support + prior_count)
        confidence = confidence.clamp(min=0.0, max=1.0)
        return torch.where(present, confidence, torch.zeros_like(confidence))

    def forward(self, player_profile: torch.Tensor) -> torch.Tensor:
        profile = player_profile.to(self.bin_pos_emb.weight.dtype)
        # A bin is present iff any feature is non-zero; absent bins are
        # zero-filled by build_dataset, and any real bin has avg_gold > 0.
        # Presence is read before standardization shifts zero-fills off zero.
        present = profile.abs().sum(dim=-1) > 0
        confidence = self._profile_confidence(profile, present)
        content = profile.index_select(-1, self._content_feature_idx)
        content_mean = self._feature_mean.index_select(0, self._content_feature_idx)
        content_std = self._feature_std.index_select(0, self._content_feature_idx)
        standardized = (content - content_mean) / content_std
        bin_repr = self.bin_mlp(standardized) + self.bin_pos_emb(self._bin_idx)
        scores = self.attn_score(bin_repr).squeeze(-1)
        scores = scores + confidence.clamp_min(1e-6).log()
        scores = scores.masked_fill(~present, float("-inf"))
        # Tokens with no present bins yield all -inf; nan_to_num maps the
        # resulting nan weights to 0 so the pooled profile is exactly zero.
        weights = torch.nan_to_num(scores.softmax(dim=-1), nan=0.0)
        level_values = bin_repr * confidence.unsqueeze(-1)
        level = (level_values * weights.unsqueeze(-1)).sum(dim=-2)

        # Trajectory direction: a weighted bin sum cannot represent "scales
        # up" vs "falls off", so add an explicit late-minus-early delta. Each
        # half is a presence-masked mean; the delta is gated to zero unless
        # both halves have a present bin, so a one-sided profile (e.g. only
        # early bins) cannot fake a slope.
        split = self.bin_split
        early = self._masked_mean(bin_repr[..., :split, :], present[..., :split])
        late = self._masked_mean(bin_repr[..., split:, :], present[..., split:])
        both_present = present[..., :split].any(-1) & present[..., split:].any(-1)
        early_conf = self._masked_mean(
            confidence[..., :split].unsqueeze(-1),
            present[..., :split],
        ).squeeze(-1)
        late_conf = self._masked_mean(
            confidence[..., split:].unsqueeze(-1),
            present[..., split:],
        ).squeeze(-1)
        delta_conf = torch.minimum(early_conf, late_conf)
        delta = (late - early) * both_present.unsqueeze(-1) * delta_conf.unsqueeze(-1)
        return level + self.delta_proj(delta)


class HybridTokenModel(nn.Module):
    """Champion set transformer over 10 player tokens.

    Each player token is the compositional sum of champion + role + build + side
    embeddings plus the 6002-derived historical profile encoding for that
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
        self.profile_encoder = _TemporalProfileEncoder(
            n_bins=vocab.n_profile_bins,
            n_features=vocab.n_profile_features,
            d=d,
            dropout=cfg.dropout,
            feature_mean=vocab.profile_mean,
            feature_std=vocab.profile_std,
            confidence_prior_count=cfg.profile_confidence_prior_count,
        )
        # Scalar gate on the profile stream, init 1.0 (identity at start);
        # the learned value measures how much profile the model keeps.
        self.profile_gate = nn.Parameter(torch.ones(()))
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
        self.lane_attention_pool = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 1, bias=False)
        )
        head_input_dim = d * 6
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
        player_profile: torch.Tensor,
    ) -> torch.Tensor:
        b = champion_idx.shape[0]
        side_idx = self._player_side.expand(b, -1)
        return (
            self.champ_emb(champion_idx)
            + self.role_emb(role_idx)
            + self.build_emb(build_idx)
            + self.side_emb(side_idx)
            + self.profile_gate * self.profile_encoder(player_profile)
        )

    def _lane_pool(self, lane_diff: torch.Tensor) -> torch.Tensor:
        """Attention-weighted summary of the 5 same-role blue-vs-red diffs.

        `lane_diff[:, i]` is the encoded blue role-i token minus the red
        role-i token. Team pooling washes out same-role matchups; a mean
        pool here would just reproduce the `blue_repr - red_repr` diff, so
        the weights are learned and input-dependent. Output [B, d].
        """
        weights = self.lane_attention_pool(lane_diff).squeeze(-1).softmax(dim=-1)
        return (lane_diff * weights.unsqueeze(-1)).sum(dim=1)

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
    def _match_features(
        first: torch.Tensor, second: torch.Tensor, lane: torch.Tensor
    ) -> torch.Tensor:
        """6-way concat: first, second, first-second, |first-second|,
        first*second, and the same-role lane-matchup summary."""
        diff = first - second
        return torch.cat(
            [first, second, diff, diff.abs(), first * second, lane],
            dim=-1,
        )

    def forward(
        self,
        champion_idx: torch.Tensor,
        role_idx: torch.Tensor,
        build_idx: torch.Tensor,
        player_profile: torch.Tensor,
    ) -> torch.Tensor:
        player = self._player_tokens(
            champion_idx,
            role_idx,
            build_idx,
            player_profile,
        )
        x = self.input_dropout(self.input_norm(player))
        z = self.encoder(x)
        blue_repr, red_repr = self._team_pool(z)
        blue_tokens = z[:, :N_TEAM_TOKENS]
        red_tokens = z[:, N_TEAM_TOKENS:]
        lane_bvr = self._lane_pool(blue_tokens - red_tokens)
        lane_rvb = self._lane_pool(red_tokens - blue_tokens)
        match_bvr = self._match_features(blue_repr, red_repr, lane_bvr)
        match_rvb = self._match_features(red_repr, blue_repr, lane_rvb)
        score_bvr = self.head(match_bvr).squeeze(-1)
        score_rvb = self.head(match_rvb).squeeze(-1)
        return score_bvr - score_rvb


__all__ = ["HybridTokenModel"]
