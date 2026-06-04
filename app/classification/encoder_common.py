"""Shared building blocks for the classification autoencoders.

Single source of truth for the small torch/numpy helpers, latent diagnostics,
and frozen-config validators that the full-game, static-identity, and temporal
encoders all need. These previously lived in `full_game_encoder` and were
re-implemented verbatim in the sibling encoders.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

LatentNorm = Literal["batch", "layer", "none"]


# --- torch runtime ---------------------------------------------------------


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_amp(amp: bool, device: torch.device) -> bool:
    return bool(amp and device.type == "cuda")


def _resolve_pin_memory(pin_memory: bool | None, device: torch.device) -> bool:
    if pin_memory is not None:
        return bool(pin_memory)
    return device.type == "cuda"


# --- network builders ------------------------------------------------------


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _latent_norm(latent_dim: int, kind: LatentNorm) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(latent_dim)
    if kind == "layer":
        return nn.LayerNorm(latent_dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError("latent_norm must be one of: 'batch', 'layer', 'none'")


def _batchnorm_single_row_safe(latent: torch.Tensor, norm: nn.Module) -> torch.Tensor:
    """Apply `norm`, but feed a single-row training batch through running stats.

    `BatchNorm1d` cannot compute batch statistics over one row, so a trailing
    1-row training batch would otherwise crash; fall back to the stored running
    statistics for that case only.
    """
    if isinstance(norm, nn.BatchNorm1d) and norm.training and latent.shape[0] == 1:
        return nn.functional.batch_norm(
            latent,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            eps=norm.eps,
        )
    return norm(latent)


# --- dataframe helpers -----------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _id_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = frame[column].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must contain numeric integer IDs") from exc
    if values.ndim != 1:
        raise ValueError(f"{column} must be a 1-D column")
    if not np.isfinite(values).all():
        raise ValueError(f"{column} contains non-finite IDs")
    if np.any(values < 0.0) or np.any(values != np.floor(values)):
        raise ValueError(f"{column} must contain non-negative integer IDs")
    return values.astype(np.int64, copy=False)


# --- latent diagnostics ----------------------------------------------------


def _latent_summary(latent_matrix: np.ndarray) -> dict[str, float]:
    if latent_matrix.ndim != 2:
        raise ValueError("latent_matrix must be 2-D")
    latent_std = latent_matrix.std(axis=0)
    centered = latent_matrix - latent_matrix.mean(axis=0, keepdims=True)
    if latent_matrix.shape[0] < 2:
        eigenvalues = np.zeros(latent_matrix.shape[1], dtype=np.float64)
    else:
        covariance = np.cov(centered, rowvar=False)
        covariance = np.atleast_2d(covariance)
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    eigen_sum = float(eigenvalues.sum())
    if eigen_sum > 0.0:
        probabilities = eigenvalues[eigenvalues > 0.0] / eigen_sum
        effective_rank = float(
            np.exp(-(probabilities * np.log(probabilities)).sum())
        )
        squared_sum = float(np.square(eigenvalues).sum())
        participation_rank = (
            float((eigen_sum * eigen_sum) / squared_sum)
            if squared_sum > 0.0
            else 0.0
        )
    else:
        effective_rank = 0.0
        participation_rank = 0.0

    if latent_matrix.shape[0] < 2 or latent_matrix.shape[1] <= 1:
        mean_abs_corr = 0.0
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(centered, rowvar=False)
        off_diagonal = ~np.eye(corr.shape[0], dtype=bool)
        mean_abs_corr = float(np.nanmean(np.abs(corr[off_diagonal])))
        if not np.isfinite(mean_abs_corr):
            mean_abs_corr = 0.0

    return {
        "latent_active_dims": float(np.count_nonzero(latent_std > 1.0e-6)),
        "latent_mean_std": float(latent_std.mean()),
        "latent_max_std": float(latent_std.max()),
        "latent_effective_rank": effective_rank,
        "latent_participation_rank": participation_rank,
        "latent_mean_abs_corr": mean_abs_corr,
    }


def _latent_decorrelation_loss(latent: torch.Tensor) -> torch.Tensor:
    if latent.shape[0] < 2 or latent.shape[1] < 2:
        return latent.new_tensor(0.0)
    values = latent.float()
    values = values - values.mean(dim=0, keepdim=True)
    values = values / values.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    ).clamp_min(1.0e-6)
    corr = values.T @ values / values.shape[0]
    corr = corr - torch.diag(torch.diag(corr))
    return corr.square().mean()


# --- frozen-config validators ----------------------------------------------


def _require_positive(config: Any, names: Sequence[str]) -> None:
    for name in names:
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")


def _normalise_positive_dims(config: Any, names: Sequence[str]) -> None:
    for name in names:
        dims = tuple(int(dim) for dim in getattr(config, name))
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"{name} must contain only positive dimensions")
        object.__setattr__(config, name, dims)


def _require_unit_interval(config: Any, name: str) -> None:
    if not 0.0 <= float(getattr(config, name)) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _require_latent_norm(value: str) -> None:
    if value not in {"batch", "layer", "none"}:
        raise ValueError("latent_norm must be one of: 'batch', 'layer', 'none'")
