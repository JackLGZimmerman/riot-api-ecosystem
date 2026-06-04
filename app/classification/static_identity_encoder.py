"""Static identity autoencoder for deterministic champion dictionary features.

This branch is champion-level. Champion base stats are a function of champion
only, so role and build never parametrise the encoding: there are no role/build
inputs and no role/build reconstruction. It also rejects empirical priors, win
rates, matchup rates, and support-count features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.classification.embeddings.static_champion import (
    build_static_matrix,
)
from app.classification.encoder_common import (
    LatentNorm,
    _batchnorm_single_row_safe,
    _id_array,
    _latent_decorrelation_loss,
    _latent_norm,
    _latent_summary,
    _mlp,
    _normalise_positive_dims,
    _require_columns,
    _require_latent_norm,
    _require_positive,
    _require_unit_interval,
    _resolve_amp,
    _resolve_device,
)

StaticLatentNorm = LatentNorm

FORBIDDEN_STATIC_INPUT_SUBSTRINGS = (
    "win_rate",
    "winrate",
    "blue_win",
    "matchups",
    "matchup_",
    "synergy",
    "prior",
    "_cnt",
    "_count",
    "count_",
)
FORBIDDEN_STATIC_INPUT_COLUMNS = frozenset(
    {
        "count",
        "cnt",
        "support",
        "p1_cnt",
        "m1v1_cnt",
        "s2vx_cnt",
        "win_rate",
        "blue_win",
        "matchups",
        "matchup_1v1",
        "synergy_2vx",
    }
)


@dataclass(frozen=True)
class StaticIdentityConfig:
    continuous_dim: int
    latent_dim: int = 128
    hidden_dims: tuple[int, ...] = (192, 96)
    decoder_hidden_dims: tuple[int, ...] = (96, 192)
    dropout: float = 0.0
    latent_dropout: float = 0.0
    latent_norm: StaticLatentNorm = "batch"

    def __post_init__(self) -> None:
        _require_positive(self, ("continuous_dim", "latent_dim"))
        _normalise_positive_dims(self, ("hidden_dims", "decoder_hidden_dims"))
        _require_unit_interval(self, "dropout")
        _require_unit_interval(self, "latent_dropout")
        _require_latent_norm(self.latent_norm)


class StaticIdentityDataset(Dataset[dict[str, torch.Tensor]]):
    """One row per champion of deterministic static identity features."""

    def __init__(
        self,
        frame: pd.DataFrame,
        continuous_columns: Sequence[str] | None = None,
        *,
        champion_col: str = "champion_id",
    ) -> None:
        _require_columns(frame, (champion_col,))
        columns = _continuous_columns(frame, continuous_columns, champion_col)
        validate_static_input_columns(columns)
        self.champion_id = torch.as_tensor(_id_array(frame, champion_col), dtype=torch.long)
        self.continuous = torch.as_tensor(_continuous_array(frame, columns), dtype=torch.float32)
        self.continuous_columns = tuple(columns)

    def __len__(self) -> int:
        return int(self.continuous.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "champion_id": self.champion_id[index],
            "continuous": self.continuous[index],
        }


class StaticIdentityAutoencoder(nn.Module):
    """Champion-level autoencoder over deterministic static stat vectors."""

    def __init__(self, config: StaticIdentityConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.encoder = _mlp(c.continuous_dim, c.hidden_dims, c.latent_dim, dropout=c.dropout)
        self.latent_norm = _latent_norm(c.latent_dim, c.latent_norm)
        self.latent_dropout = nn.Dropout(c.latent_dropout)
        self.decoder = _mlp(
            c.latent_dim,
            c.decoder_hidden_dims,
            c.continuous_dim,
            dropout=c.dropout,
        )

    def encode(self, continuous: torch.Tensor) -> torch.Tensor:
        _validate_continuous(continuous, self.config.continuous_dim)
        latent = self.encoder(continuous)
        return _batchnorm_single_row_safe(latent, self.latent_norm)

    def forward(self, continuous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(continuous)
        return self.decoder(self.latent_dropout(latent)), latent


def static_identity_frame(
    champion_ids: Sequence[int],
    *,
    clip_value: float | None = 8.0,
) -> pd.DataFrame:
    """Per-champion standardised static-stat frame (one row per unique champion)."""
    unique = np.unique(np.asarray(list(champion_ids), dtype=np.int64))
    if unique.size == 0:
        raise ValueError("champion_ids must not be empty")
    static_matrix, names = build_static_matrix(unique, clip_value=clip_value)
    frame = pd.DataFrame(static_matrix, columns=list(names))
    frame.insert(0, "champion_id", unique)
    return frame


def train_static_autoencoder(
    model: StaticIdentityAutoencoder,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    *,
    epochs: int = 20,
    lr: float = 1.0e-3,
    device: str | torch.device = "cpu",
    noise_std: float = 0.0,
    mask_prob: float = 0.0,
    latent_decorrelation_weight: float = 5.0e-4,
    weight_decay: float = 0.0,
    amp: bool = True,
) -> list[dict[str, float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if latent_decorrelation_weight < 0.0:
        raise ValueError("latent_decorrelation_weight must be non-negative")
    device = _resolve_device(device)
    amp_enabled = _resolve_amp(amp, device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    history: list[dict[str, float]] = []
    non_blocking = device.type == "cuda"
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "continuous_loss": 0.0, "latent_decorrelation_loss": 0.0}
        rows = 0
        for batch in dataloader:
            batch = _batch_to_device(batch, device, non_blocking=non_blocking)
            corrupted = _corrupt_continuous(
                batch["continuous"],
                noise_std=noise_std,
                mask_prob=mask_prob,
            )
            context = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
            optimizer.zero_grad(set_to_none=True)
            with context:
                reconstruction, latent = model(corrupted)
                continuous_loss = nn.functional.mse_loss(
                    reconstruction.float(),
                    batch["continuous"].float(),
                )
                decorrelation_loss = _latent_decorrelation_loss(latent)
                loss = continuous_loss + latent_decorrelation_weight * decorrelation_loss
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            n = int(batch["continuous"].shape[0])
            rows += n
            totals["loss"] += float(loss.detach().cpu()) * n
            totals["continuous_loss"] += float(continuous_loss.detach().cpu()) * n
            totals["latent_decorrelation_loss"] += float(decorrelation_loss.detach().cpu()) * n
        denom = max(rows, 1)
        history.append({"epoch": float(epoch), **{key: value / denom for key, value in totals.items()}})
    return history


def evaluate_static_autoencoder(
    model: StaticIdentityAutoencoder,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    device = _resolve_device(device)
    was_training = model.training
    model.to(device)
    model.eval()
    total_squared_error = 0.0
    total_values = 0
    total_rows = 0
    latents: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    champion_ids: list[np.ndarray] = []
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for batch in dataloader:
            batch = _batch_to_device(batch, device, non_blocking=non_blocking)
            reconstruction, latent = model(batch["continuous"])
            error = reconstruction - batch["continuous"]
            total_squared_error += float(torch.sum(error.square()).detach().cpu())
            total_values += int(error.numel())
            total_rows += int(batch["continuous"].shape[0])
            latents.append(latent.detach().cpu().numpy())
            reconstructions.append(reconstruction.detach().cpu().numpy())
            truths.append(batch["continuous"].detach().cpu().numpy())
            champion_ids.append(batch["champion_id"].detach().cpu().numpy())
    if was_training:
        model.train()
    if total_rows == 0 or total_values == 0:
        raise ValueError("dataloader must yield at least one row")
    latent_matrix = np.concatenate(latents, axis=0)
    recovery = _champion_recovery_accuracy(
        np.concatenate(reconstructions, axis=0),
        np.concatenate(truths, axis=0),
        np.concatenate(champion_ids, axis=0),
    )
    return {
        "mse": total_squared_error / total_values,
        "rows": float(total_rows),
        "champion_recovery_accuracy": recovery,
        **_latent_summary(latent_matrix),
    }


@torch.no_grad()
def extract_static_latents(
    model: StaticIdentityAutoencoder,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    device = _resolve_device(device)
    was_training = model.training
    model.to(device)
    model.eval()
    champion_ids: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    non_blocking = device.type == "cuda"
    for batch in dataloader:
        batch = _batch_to_device(batch, device, non_blocking=non_blocking)
        latent = model.encode(batch["continuous"])
        champion_ids.append(batch["champion_id"].detach().cpu().numpy())
        latents.append(latent.detach().cpu().numpy())
    if was_training:
        model.train()
    latent_columns = [f"static_latent_{i}" for i in range(model.config.latent_dim)]
    if not latents:
        return pd.DataFrame(columns=["champion_id", *latent_columns])
    latent_matrix = np.concatenate(latents, axis=0)
    out = {"champion_id": np.concatenate(champion_ids, axis=0)}
    for i, name in enumerate(latent_columns):
        out[name] = latent_matrix[:, i]
    return pd.DataFrame(out, columns=["champion_id", *latent_columns])


def validate_static_input_columns(columns: Sequence[str]) -> None:
    rejected: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if lowered in FORBIDDEN_STATIC_INPUT_COLUMNS:
            rejected.append(str(column))
            continue
        if any(part in lowered for part in FORBIDDEN_STATIC_INPUT_SUBSTRINGS):
            rejected.append(str(column))
    if rejected:
        raise ValueError(
            "static encoder inputs cannot include empirical priors, win rates, "
            f"or support counts: {sorted(set(rejected))}"
        )


def _champion_recovery_accuracy(
    reconstruction: np.ndarray,
    truth: np.ndarray,
    champion_ids: np.ndarray,
) -> float:
    """Fraction of champions whose reconstruction is nearest their own stats.

    Static stats are a fixed champion dictionary, so the encoding only matters if
    the original champion is recoverable from it: for each champion the decoded
    static vector should be closer to that champion's true stats than to any
    other champion's.
    """
    if reconstruction.shape[0] < 2:
        return 1.0
    recon = reconstruction.astype(np.float64, copy=False)
    true = truth.astype(np.float64, copy=False)
    recon_sq = np.sum(recon * recon, axis=1, keepdims=True)
    true_sq = np.sum(true * true, axis=1, keepdims=True)
    distances = recon_sq + true_sq.T - 2.0 * recon @ true.T
    nearest = np.argmin(distances, axis=1)
    return float(np.mean(champion_ids[nearest] == champion_ids))


def _continuous_columns(
    frame: pd.DataFrame,
    columns: Sequence[str] | None,
    champion_col: str,
) -> tuple[str, ...]:
    if columns is None:
        columns = tuple(column for column in frame.columns if column != champion_col)
    if isinstance(columns, str):
        raise ValueError("continuous_columns must be a sequence of column names")
    out = tuple(str(column) for column in columns)
    if not out:
        raise ValueError("continuous_columns must not be empty")
    _require_columns(frame, out)
    return out


def _continuous_array(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32, copy=True)
    if values.ndim != 2:
        raise ValueError("continuous static features must be a 2-D matrix")
    if not np.isfinite(values).all():
        raise ValueError("continuous static features contain non-finite values")
    return values


def _validate_continuous(values: torch.Tensor, width: int) -> None:
    if values.ndim != 2:
        raise ValueError("continuous must have shape [batch, continuous_dim]")
    if values.shape[1] != width:
        raise ValueError(f"continuous width must be {width}, got {int(values.shape[1])}")
    if not values.is_floating_point():
        raise ValueError("continuous must be a floating point tensor")


def _batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        "champion_id": batch["champion_id"].to(device=device, dtype=torch.long, non_blocking=non_blocking),
        "continuous": batch["continuous"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
    }


def _corrupt_continuous(values: torch.Tensor, *, noise_std: float, mask_prob: float) -> torch.Tensor:
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if not 0.0 <= mask_prob <= 1.0:
        raise ValueError("mask_prob must be between 0 and 1")
    if noise_std == 0.0 and mask_prob == 0.0:
        return values
    out = values.clone()
    if noise_std > 0.0:
        out = out + torch.randn_like(out) * noise_std
    if mask_prob > 0.0:
        out = out * (torch.rand_like(out) >= mask_prob)
    return out


__all__ = [
    "FORBIDDEN_STATIC_INPUT_COLUMNS",
    "FORBIDDEN_STATIC_INPUT_SUBSTRINGS",
    "StaticIdentityAutoencoder",
    "StaticIdentityConfig",
    "StaticIdentityDataset",
    "evaluate_static_autoencoder",
    "extract_static_latents",
    "static_identity_frame",
    "train_static_autoencoder",
    "validate_static_input_columns",
]
