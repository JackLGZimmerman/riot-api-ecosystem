from __future__ import annotations

import torch
from torch import nn

from app.ml.config import (
    N_SIDES,
    N_TOKEN_TYPES,
    SIDE_BLUE,
    SIDE_RED,
    TOKEN_TYPE_PLAYER,
    ModelConfig,
)
from app.ml.dataset import InteractionLayout, Vocab


class HybridTokenModel(nn.Module):
    """Hybrid champion + interaction transformer.

    Each game becomes a sequence of:
      - 10 champion-role-build player tokens (5 blue, 5 red)
      - N_INTERACTION_TOKENS interaction tokens carrying matchup/synergy scores

    A learnable [CLS] token attends across the full sequence and produces the
    blue-win logit. Interaction tokens fuse a token-type embedding, a side
    embedding (blue/red/cross), per-slot role embeddings, and a small linear
    projection of the (centered_win_rate, reliability) scalars.
    """

    def __init__(
        self,
        vocab: Vocab,
        layout: InteractionLayout,
        cfg: ModelConfig,
    ):
        super().__init__()
        d = cfg.d_model

        self.champ_emb = nn.Embedding(vocab.n_champions, d)
        self.role_emb = nn.Embedding(vocab.n_roles, d)
        self.build_emb = nn.Embedding(vocab.n_builds, d)
        self.side_emb = nn.Embedding(N_SIDES, d)
        self.type_emb = nn.Embedding(N_TOKEN_TYPES, d)
        self.score_proj = nn.Linear(2, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.register_buffer(
            "_player_side",
            torch.tensor(
                [[SIDE_BLUE] * 5 + [SIDE_RED] * 5], dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "_player_type",
            torch.full((1, 10), TOKEN_TYPE_PLAYER, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_interaction_types", layout.types.unsqueeze(0), persistent=False
        )
        self.register_buffer(
            "_interaction_sides", layout.sides.unsqueeze(0), persistent=False
        )
        self.register_buffer(
            "_interaction_roles", layout.roles.unsqueeze(0), persistent=False
        )

        self.input_norm = nn.LayerNorm(d)
        self.input_dropout = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.head_hidden),
            nn.GELU(),
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
        type_idx = self._player_type.expand(b, -1)
        return (
            self.champ_emb(champion_idx)
            + self.role_emb(role_idx)
            + self.build_emb(build_idx)
            + self.side_emb(side_idx)
            + self.type_emb(type_idx)
        )

    def _interaction_tokens(
        self,
        score: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        b = score.shape[0]
        types = self._interaction_types.expand(b, -1)
        sides = self._interaction_sides.expand(b, -1)
        roles = self._interaction_roles.expand(b, -1, -1)

        type_e = self.type_emb(types)
        side_e = self.side_emb(sides)
        # Sum role embeddings across the N_ROLE_SLOTS slots; UNK_INDEX=0 maps
        # to the unused-slot embedding row, which the model is free to drive
        # toward zero contribution during training.
        role_e = self.role_emb(roles).sum(dim=2)

        scalars = torch.stack([score, reliability], dim=-1)  # (B, K, 2)
        score_e = self.score_proj(scalars)
        return type_e + side_e + role_e + score_e

    def forward(
        self,
        champion_idx: torch.Tensor,  # (B, 10)
        role_idx: torch.Tensor,  # (B, 10)
        build_idx: torch.Tensor,  # (B, 10)
        interaction_score: torch.Tensor,  # (B, K)
        interaction_reliability: torch.Tensor,  # (B, K)
    ) -> torch.Tensor:  # (B,) blue_win logits
        player = self._player_tokens(champion_idx, role_idx, build_idx)
        interaction = self._interaction_tokens(
            interaction_score, interaction_reliability
        )
        b = player.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, player, interaction], dim=1)
        x = self.input_dropout(self.input_norm(x))
        z = self.encoder(x)
        return self.head(z[:, 0]).squeeze(-1)


__all__ = ["HybridTokenModel"]
