"""Dedicated temporal autoencoder over per-identity (bucket, metric) tensors.

Separate from the full-game `champion_semantics` autoencoder: it reconstructs the
standardised temporal tensor produced by
`app/classification/embeddings/temporal.py`, learning a compact latent for each
identity's stat trajectory. Reconstruction is mask-aware so buckets a short-lived
identity never reached do not create loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


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
    metric_embed_dim: int = 24
    latent_dim: int = 96
    champ_embed_dim: int = 16
    pos_embed_dim: int = 4
    build_embed_dim: int = 8
    hidden: int = 256
    dropout: float = 0.05


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
        self.metric_embed = nn.Linear(n_metric, c.metric_embed_dim)
        self.champ_embed = nn.Embedding(champ_vocab, c.champ_embed_dim)
        self.pos_embed = nn.Embedding(pos_vocab, c.pos_embed_dim)
        self.build_embed = nn.Embedding(build_vocab, c.build_embed_dim)

        enc_in = (
            n_buckets * c.metric_embed_dim
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
        self.decoder = nn.Sequential(
            nn.Linear(c.latent_dim, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, n_buckets * n_metric),
        )

    def encode(self, x, champ, pos, build):
        b = x.shape[0]
        h = self.metric_embed(x).reshape(b, -1)  # (b, buckets*embed)
        h = torch.cat(
            [h, self.champ_embed(champ), self.pos_embed(pos), self.build_embed(build)],
            dim=-1,
        )
        z = self.encoder(h)
        return self.latent_norm(z)

    def forward(self, x, champ, pos, build):
        z = self.encode(x, champ, pos, build)
        recon = self.decoder(z).reshape(x.shape[0], self.n_buckets, self.n_metric)
        return recon, z


def masked_mse(recon, target, mask) -> torch.Tensor:
    # mask is (b, buckets); broadcast over the metric axis.
    w = mask.unsqueeze(-1)
    sq = (recon - target) ** 2 * w
    denom = w.sum() * target.shape[-1]
    return sq.sum() / torch.clamp(denom, min=1.0)


def train_temporal(
    tensors,
    *,
    epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = "cpu",
    cfg: TemporalAEConfig | None = None,
    seed: int = 0,
) -> tuple[TemporalAutoencoder, list[dict]]:
    torch.manual_seed(seed)
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
        total, nb = 0.0, 0
        for x, mask, champ, pos, build in loader:
            x, mask = x.to(device), mask.to(device)
            champ, pos, build = champ.to(device), pos.to(device), build.to(device)
            recon, _ = model(x, champ, pos, build)
            loss = masked_mse(recon, x, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        history.append({"epoch": epoch, "loss": total / max(nb, 1)})
    return model, history


@torch.no_grad()
def extract_temporal_latents(model, tensors, device: str = "cpu") -> np.ndarray:
    model.eval()
    ds = TemporalDataset(tensors.values, tensors.mask, tensors.keys)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    out: list[np.ndarray] = []
    for x, _mask, champ, pos, build in loader:
        z = model.encode(
            x.to(device), champ.to(device), pos.to(device), build.to(device)
        )
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)
