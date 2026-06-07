"""Dedicated temporal autoencoder over per-identity (bucket, metric) tensors.

Separate from the full-game `full_game_encoder` branch: it reconstructs the
standardised temporal tensor produced by
`app/classification/embeddings/temporal.py`, learning a compact latent for each
identity's stat trajectory. Reconstruction is mask-aware so buckets a short-lived
identity never reached do not create loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.classification.encoder_common import (
    _batchnorm_single_row_safe,
    _latent_decorrelation_loss,
    _latent_summary,
    _require_positive,
    _require_unit_interval,
    _resolve_device,
)

logger = logging.getLogger(__name__)

TemporalArchitecture = Literal["flat", "tcn", "gru", "transformer"]
SUPPORTED_TEMPORAL_ARCHITECTURES: frozenset[str] = frozenset(
    {"flat", "tcn", "gru", "transformer"}
)


def _id_vocab(keys: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    champ = np.asarray([int(k[0]) for k in keys], dtype=np.int64)
    pos_labels = sorted({str(k[1]) for k in keys})
    build_labels = sorted({str(k[2]) for k in keys})
    pos_idx = {label: i for i, label in enumerate(pos_labels)}
    build_idx = {label: i for i, label in enumerate(build_labels)}
    pos = np.asarray([pos_idx[str(k[1])] for k in keys], dtype=np.int64)
    build = np.asarray([build_idx[str(k[2])] for k in keys], dtype=np.int64)
    return champ, pos, build


class TemporalDataset(Dataset):
    def __init__(self, values: np.ndarray, mask: np.ndarray, keys: list[tuple]) -> None:
        self.values = torch.from_numpy(values.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.float32))
        champ, pos, build = _id_vocab(keys)
        self.champ = torch.from_numpy(champ)
        self.pos = torch.from_numpy(pos)
        self.build = torch.from_numpy(build)
        self.champ_vocab = int(champ.max()) + 1
        self.pos_vocab = int(pos.max()) + 1
        self.build_vocab = int(build.max()) + 1

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, i: int):
        return self.values[i], self.mask[i], self.champ[i], self.pos[i], self.build[i]


@dataclass
class TemporalAEConfig:
    metric_embed_dim: int = 96
    latent_dim: int = 416
    champ_embed_dim: int = 16
    pos_embed_dim: int = 4
    build_embed_dim: int = 8
    hidden: int = 1536
    dropout: float = 0.02
    # Off by default: a 2-seed sweep showed decoder-side latent dropout strictly
    # hurts this branch (worse masked MSE and lower effective rank). Kept as an
    # ablation lever for parity with the full-game/static encoders.
    latent_dropout: float = 0.0
    zero_unobserved_input: bool = True
    mask_as_input: bool = False
    architecture: TemporalArchitecture = "flat"
    tcn_layers: int = 2
    tcn_kernel_size: int = 3
    transformer_layers: int = 2
    transformer_heads: int = 4

    def __post_init__(self) -> None:
        _require_positive(
            self,
            (
                "metric_embed_dim",
                "latent_dim",
                "champ_embed_dim",
                "pos_embed_dim",
                "build_embed_dim",
                "hidden",
                "tcn_layers",
                "tcn_kernel_size",
                "transformer_layers",
                "transformer_heads",
            ),
        )
        _require_unit_interval(self, "dropout")
        _require_unit_interval(self, "latent_dropout")
        _require_temporal_architecture(self.architecture)
        if self.tcn_kernel_size % 2 == 0:
            raise ValueError("tcn_kernel_size must be odd so bucket length is preserved")


class _TemporalTcnBackbone(nn.Module):
    def __init__(self, width: int, *, layers: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        blocks: list[nn.Module] = []
        for _ in range(layers):
            blocks.extend(
                (
                    nn.Conv1d(width, width, kernel_size=kernel_size, padding=padding),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: (batch, buckets, width)
        encoded = self.net(sequence.transpose(1, 2)).transpose(1, 2)
        return encoded


class _TemporalTransformerBackbone(nn.Module):
    def __init__(
        self,
        n_buckets: int,
        width: int,
        *,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        heads = _attention_heads(width, heads)
        self.position = nn.Parameter(torch.zeros(1, n_buckets, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=max(width * 4, 32),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = sequence + self.position
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
            all_padded = key_padding_mask.all(dim=1)
            if bool(all_padded.any()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_padded] = False
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class TemporalAutoencoder(nn.Module):
    def __init__(
        self,
        n_buckets: int,
        n_metric: int,
        champ_vocab: int,
        pos_vocab: int,
        build_vocab: int,
        cfg: TemporalAEConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or TemporalAEConfig()
        self.n_buckets, self.n_metric = n_buckets, n_metric
        c = self.cfg

        # Shared per-bucket metric embedding (applied to every bucket's vector).
        self.metric_input_dim = n_metric + (1 if c.mask_as_input else 0)
        self.metric_embed = nn.Linear(self.metric_input_dim, c.metric_embed_dim)
        # The latent always carries the full (champion, role, build) identity.
        self.champ_embed = nn.Embedding(champ_vocab, c.champ_embed_dim)
        self.pos_embed = nn.Embedding(pos_vocab, c.pos_embed_dim)
        self.build_embed = nn.Embedding(build_vocab, c.build_embed_dim)

        self.temporal_backbone: nn.Module | None
        if c.architecture == "flat":
            self.temporal_backbone = None
            sequence_feature_dim = n_buckets * c.metric_embed_dim
        elif c.architecture == "tcn":
            self.temporal_backbone = _TemporalTcnBackbone(
                c.metric_embed_dim,
                layers=c.tcn_layers,
                kernel_size=c.tcn_kernel_size,
                dropout=c.dropout,
            )
            sequence_feature_dim = c.metric_embed_dim * 2
        elif c.architecture == "gru":
            self.temporal_backbone = nn.GRU(
                input_size=c.metric_embed_dim,
                hidden_size=c.metric_embed_dim,
                num_layers=1,
                batch_first=True,
            )
            sequence_feature_dim = c.metric_embed_dim * 2
        elif c.architecture == "transformer":
            self.temporal_backbone = _TemporalTransformerBackbone(
                n_buckets,
                c.metric_embed_dim,
                layers=c.transformer_layers,
                heads=c.transformer_heads,
                dropout=c.dropout,
            )
            sequence_feature_dim = c.metric_embed_dim * 2
        else:  # pragma: no cover - guarded by TemporalAEConfig
            raise ValueError(f"unsupported temporal architecture: {c.architecture}")

        enc_in = (
            sequence_feature_dim
            + c.champ_embed_dim
            + c.pos_embed_dim
            + c.build_embed_dim
        )
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, c.latent_dim),
        )
        self.latent_norm = nn.BatchNorm1d(c.latent_dim)
        self.latent_dropout = nn.Dropout(c.latent_dropout)
        self.decoder = nn.Sequential(
            nn.Linear(c.latent_dim, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, n_buckets * n_metric),
        )

    def encode(self, x, champ, pos, build, mask=None):
        if self.cfg.zero_unobserved_input and mask is not None:
            x = x * mask.to(dtype=x.dtype).unsqueeze(-1)
        if self.cfg.mask_as_input:
            if mask is None:
                mask_channel = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
            else:
                mask_channel = mask.to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, mask_channel.unsqueeze(-1)], dim=-1)
        b = x.shape[0]
        sequence = self.metric_embed(x)
        if self.cfg.architecture == "flat":
            h = sequence.reshape(b, -1)  # (b, buckets*embed)
        elif self.cfg.architecture == "tcn":
            assert isinstance(self.temporal_backbone, _TemporalTcnBackbone)
            h = _masked_sequence_pool(self.temporal_backbone(sequence), mask)
        elif self.cfg.architecture == "gru":
            assert isinstance(self.temporal_backbone, nn.GRU)
            encoded, _hidden = self.temporal_backbone(sequence)
            h = _masked_sequence_pool(encoded, mask)
        elif self.cfg.architecture == "transformer":
            assert isinstance(self.temporal_backbone, _TemporalTransformerBackbone)
            h = _masked_sequence_pool(self.temporal_backbone(sequence, mask), mask)
        else:  # pragma: no cover - guarded by TemporalAEConfig
            raise ValueError(f"unsupported temporal architecture: {self.cfg.architecture}")
        h = torch.cat(
            [h, self.champ_embed(champ), self.pos_embed(pos), self.build_embed(build)],
            dim=-1,
        )
        z = self.encoder(h)
        return _batchnorm_single_row_safe(z, self.latent_norm)

    def forward(self, x, champ, pos, build, mask=None):
        z = self.encode(x, champ, pos, build, mask)
        # Decoder sees a dropout-corrupted latent during training, while `encode`
        # still yields the clean latent for extraction. This spreads
        # reconstruction pressure across more independent latent dimensions.
        recon = self.decoder(self.latent_dropout(z)).reshape(
            x.shape[0], self.n_buckets, self.n_metric
        )
        return recon, z


def masked_mse(recon, target, mask) -> torch.Tensor:
    # mask is (b, buckets); broadcast over the metric axis.
    w = mask.unsqueeze(-1)
    sq = (recon - target) ** 2 * w
    denom = w.sum() * target.shape[-1]
    return sq.sum() / torch.clamp(denom, min=1.0)


def _masked_sequence_pool(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.cat([sequence.mean(dim=1), sequence.amax(dim=1)], dim=-1)
    weights = mask.to(dtype=sequence.dtype).unsqueeze(-1)
    observed = weights.sum(dim=1)
    denom = observed.clamp_min(1.0)
    mean = (sequence * weights).sum(dim=1) / denom
    masked = sequence.masked_fill(weights <= 0.0, torch.finfo(sequence.dtype).min)
    max_values = masked.amax(dim=1)
    max_values = torch.where(observed > 0.0, max_values, torch.zeros_like(max_values))
    return torch.cat([mean, max_values], dim=-1)


def _require_temporal_architecture(value: str) -> None:
    if value not in SUPPORTED_TEMPORAL_ARCHITECTURES:
        known = ", ".join(sorted(SUPPORTED_TEMPORAL_ARCHITECTURES))
        raise ValueError(f"architecture must be one of: {known}")


def _attention_heads(width: int, requested: int) -> int:
    if width % requested == 0:
        return requested
    for heads in range(min(requested, width), 0, -1):
        if width % heads == 0:
            return heads
    return 1


def train_temporal(
    tensors,
    *,
    epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    cfg: TemporalAEConfig | None = None,
    latent_decorrelation_weight: float = 0.0,
    seed: int = 0,
) -> tuple[TemporalAutoencoder, list[dict]]:
    if latent_decorrelation_weight < 0.0:
        raise ValueError("latent_decorrelation_weight must be non-negative")
    torch.manual_seed(seed)
    device = _resolve_device(device)
    ds = TemporalDataset(tensors.values, tensors.mask, tensors.keys)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = TemporalAutoencoder(
        tensors.values.shape[1],
        tensors.values.shape[2],
        ds.champ_vocab,
        ds.pos_vocab,
        ds.build_vocab,
        cfg,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        total, total_recon, total_decorr, nb = 0.0, 0.0, 0.0, 0
        for x, mask, champ, pos, build in loader:
            x, mask = x.to(device), mask.to(device)
            champ, pos, build = champ.to(device), pos.to(device), build.to(device)
            recon, latent = model(x, champ, pos, build, mask)
            recon_loss = masked_mse(recon, x, mask)
            decorr_loss = _latent_decorrelation_loss(latent)
            loss = recon_loss + latent_decorrelation_weight * decorr_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            total_recon += recon_loss.item()
            total_decorr += decorr_loss.item()
            nb += 1
        denom = max(nb, 1)
        history.append(
            {
                "epoch": epoch,
                "loss": total / denom,
                "masked_mse": total_recon / denom,
                "latent_decorrelation_loss": total_decorr / denom,
            }
        )
    return model, history


@torch.no_grad()
def extract_temporal_latents(
    model, tensors, device: str | torch.device = "cpu"
) -> np.ndarray:
    device = _resolve_device(device)
    was_training = model.training
    model.to(device)
    model.eval()
    ds = TemporalDataset(tensors.values, tensors.mask, tensors.keys)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    out: list[np.ndarray] = []
    for x, _mask, champ, pos, build in loader:
        z = model.encode(
            x.to(device),
            champ.to(device),
            pos.to(device),
            build.to(device),
            _mask.to(device),
        )
        out.append(z.cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(out, axis=0)


@torch.no_grad()
def evaluate_temporal_autoencoder(
    model, tensors, device: str | torch.device = "cpu", *, batch_size: int = 2048
) -> dict[str, float]:
    """Mask-aware reconstruction plus latent-grouping diagnostics on clean input."""
    device = _resolve_device(device)
    was_training = model.training
    model.to(device)
    model.eval()
    ds = TemporalDataset(tensors.values, tensors.mask, tensors.keys)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    total_squared_error = 0.0
    total_values = 0.0
    latents: list[np.ndarray] = []
    for x, mask, champ, pos, build in loader:
        x, mask = x.to(device), mask.to(device)
        champ, pos, build = champ.to(device), pos.to(device), build.to(device)
        recon, latent = model(x, champ, pos, build, mask)
        w = mask.unsqueeze(-1)
        total_squared_error += float((((recon - x) ** 2) * w).sum().cpu())
        total_values += float(w.sum().cpu()) * x.shape[-1]
        latents.append(latent.cpu().numpy())
    if was_training:
        model.train()
    if total_values == 0.0:
        raise ValueError("tensors must contain at least one observed bucket")
    return {
        "masked_mse": total_squared_error / total_values,
        "rows": float(len(ds)),
        **_latent_summary(np.concatenate(latents, axis=0)),
    }
