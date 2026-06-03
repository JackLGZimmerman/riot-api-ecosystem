"""Full-game identity metrics autoencoder."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.classification.embeddings.config import (
    DERIVED_METRIC_FUNCS,
    raw_and_derived_metric_names,
)
from app.classification.embeddings.context_features import CONTEXT_FEATURE_NAMES

_NON_PROFILE_METRIC_COLUMNS = frozenset({"matchups"})
DEFAULT_METRICS_HIDDEN_DIMS = (320, 160)
DEFAULT_METRIC_NOISE_STD = 0.003
DEFAULT_LATENT_DECORRELATION_WEIGHT = 5.0e-4
DEFAULT_LATENT_DROPOUT = 0.10
LatentNorm = Literal["batch", "layer", "none"]


@dataclass(frozen=True)
class FullGameSemanticConfig:
    n_champions: int
    n_teampositions: int
    n_builds: int
    metrics_dim: int
    latent_dim: int = 640
    champion_embedding_dim: int = 16
    teamposition_embedding_dim: int = 4
    build_embedding_dim: int = 8
    metrics_embedding_dim: int = 160
    metrics_hidden_dims: tuple[int, ...] = DEFAULT_METRICS_HIDDEN_DIMS
    fusion_hidden_dims: tuple[int, ...] = (128,)
    decoder_hidden_dims: tuple[int, ...] = (512, 384)
    dropout: float = 0.0
    latent_dropout: float = DEFAULT_LATENT_DROPOUT
    latent_norm: LatentNorm = "batch"

    def __post_init__(self) -> None:
        for name in (
            "n_champions",
            "n_teampositions",
            "n_builds",
            "metrics_dim",
            "latent_dim",
            "champion_embedding_dim",
            "teamposition_embedding_dim",
            "build_embedding_dim",
            "metrics_embedding_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("metrics_hidden_dims", "fusion_hidden_dims", "decoder_hidden_dims"):
            dims = tuple(int(dim) for dim in getattr(self, name))
            if any(dim <= 0 for dim in dims):
                raise ValueError(f"{name} must contain only positive dimensions")
            object.__setattr__(self, name, dims)
        if not 0.0 <= float(self.dropout) <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if not 0.0 <= float(self.latent_dropout) <= 1.0:
            raise ValueError("latent_dropout must be between 0 and 1")
        if self.latent_norm not in {"batch", "layer", "none"}:
            raise ValueError("latent_norm must be one of: 'batch', 'layer', 'none'")


class FullGameProfileDataset(Dataset[dict[str, torch.Tensor]]):
    """Rows of normalized full-game identity metrics for autoencoder training."""

    def __init__(
        self,
        frame: pd.DataFrame,
        metric_columns: Sequence[str] | None = None,
        *,
        champion_col: str = "champion_id",
        teamposition_col: str = "teamposition_id",
        build_col: str = "build_id",
    ) -> None:
        metric_columns = _metric_columns(metric_columns)
        _require_columns(frame, (champion_col, teamposition_col, build_col))
        self.champion_id = torch.as_tensor(_id_array(frame, champion_col), dtype=torch.long)
        self.teamposition_id = torch.as_tensor(_id_array(frame, teamposition_col), dtype=torch.long)
        self.build_id = torch.as_tensor(_id_array(frame, build_col), dtype=torch.long)
        self.metrics = torch.as_tensor(_metrics_array(frame, metric_columns), dtype=torch.float32)
        self.metric_columns = tuple(metric_columns)

    def __len__(self) -> int:
        return int(self.metrics.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "champion_id": self.champion_id[index],
            "teamposition_id": self.teamposition_id[index],
            "build_id": self.build_id[index],
            "metrics": self.metrics[index],
        }


class FullGameEncoder(nn.Module):
    """Encode the full `(champion, role, build)` identity and its metrics.

    The latent always fuses champion/role/build identity embeddings with the
    normalized full-game behavioral metric vector; this branch operates at the
    `(champion, role, build)` grain by construction.
    """

    def __init__(self, config: FullGameSemanticConfig) -> None:
        super().__init__()
        self.config = config
        self.champion_embedding = nn.Embedding(
            config.n_champions,
            config.champion_embedding_dim,
        )
        self.teamposition_embedding = nn.Embedding(
            config.n_teampositions,
            config.teamposition_embedding_dim,
        )
        self.build_embedding = nn.Embedding(config.n_builds, config.build_embedding_dim)
        self.metrics_encoder = _mlp(
            config.metrics_dim,
            config.metrics_hidden_dims,
            config.metrics_embedding_dim,
            dropout=config.dropout,
        )
        fusion_dim = (
            config.metrics_embedding_dim
            + config.champion_embedding_dim
            + config.teamposition_embedding_dim
            + config.build_embedding_dim
        )
        self.fusion = _mlp(
            fusion_dim,
            config.fusion_hidden_dims,
            config.latent_dim,
            dropout=config.dropout,
        )
        self.latent_norm = _latent_norm(config.latent_dim, config.latent_norm)

    def forward(
        self,
        champion_id: torch.Tensor,
        teamposition_id: torch.Tensor,
        build_id: torch.Tensor,
        metrics: torch.Tensor,
    ) -> torch.Tensor:
        _validate_metrics(metrics, self.config.metrics_dim)
        batch_size = metrics.shape[0]
        _validate_ids("champion_id", champion_id, batch_size, self.config.n_champions)
        _validate_ids(
            "teamposition_id",
            teamposition_id,
            batch_size,
            self.config.n_teampositions,
        )
        _validate_ids("build_id", build_id, batch_size, self.config.n_builds)

        fused = torch.cat(
            [
                self.champion_embedding(champion_id),
                self.teamposition_embedding(teamposition_id),
                self.build_embedding(build_id),
                self.metrics_encoder(metrics),
            ],
            dim=-1,
        )
        latent = self.fusion(fused)
        if (
            isinstance(self.latent_norm, nn.BatchNorm1d)
            and self.training
            and latent.shape[0] == 1
        ):
            return nn.functional.batch_norm(
                latent,
                self.latent_norm.running_mean,
                self.latent_norm.running_var,
                self.latent_norm.weight,
                self.latent_norm.bias,
                training=False,
                eps=self.latent_norm.eps,
            )
        return self.latent_norm(latent)


class FullGameAutoencoder(nn.Module):
    def __init__(self, config: FullGameSemanticConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = FullGameEncoder(config)
        self.latent_dropout = nn.Dropout(config.latent_dropout)
        self.decoder = _mlp(
            config.latent_dim,
            config.decoder_hidden_dims,
            config.metrics_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        champion_id: torch.Tensor,
        teamposition_id: torch.Tensor,
        build_id: torch.Tensor,
        metrics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(champion_id, teamposition_id, build_id, metrics)
        return self.decoder(self.latent_dropout(latent)), latent


def train_autoencoder(
    model: FullGameAutoencoder,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    *,
    epochs: int = 20,
    lr: float = 1.0e-3,
    device: str | torch.device = "cpu",
    noise_std: float = DEFAULT_METRIC_NOISE_STD,
    mask_prob: float = 0.0,
    latent_decorrelation_weight: float = DEFAULT_LATENT_DECORRELATION_WEIGHT,
    weight_decay: float = 0.0,
    amp: bool = True,
) -> list[dict[str, float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if not 0.0 <= mask_prob <= 1.0:
        raise ValueError("mask_prob must be between 0 and 1")
    if latent_decorrelation_weight < 0.0:
        raise ValueError("latent_decorrelation_weight must be non-negative")

    device = _resolve_device(device)
    amp_enabled = _resolve_amp(amp, device)
    _configure_cuda_fast_math(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    non_blocking = device.type == "cuda"
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_decorrelation_loss = 0.0
        total_rows = 0
        for batch in dataloader:
            batch = _batch_to_device(batch, device, non_blocking=non_blocking)
            clean_metrics = batch["metrics"]
            metric_input = _corrupt_metrics(
                clean_metrics,
                noise_std=noise_std,
                mask_prob=mask_prob,
            )

            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.amp.autocast("cuda") if amp_enabled else nullcontext()
            )
            with context:
                reconstruction, latent = model(
                    batch["champion_id"],
                    batch["teamposition_id"],
                    batch["build_id"],
                    metric_input,
                )
            reconstruction_loss = loss_fn(reconstruction.float(), clean_metrics)
            decorrelation_loss = _latent_decorrelation_loss(latent)
            loss = reconstruction_loss + latent_decorrelation_weight * decorrelation_loss
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = int(clean_metrics.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_reconstruction_loss += (
                float(reconstruction_loss.detach().cpu()) * batch_size
            )
            total_decorrelation_loss += (
                float(decorrelation_loss.detach().cpu()) * batch_size
            )
            total_rows += batch_size

        if total_rows == 0:
            raise ValueError("dataloader must yield at least one row")
        history.append(
            {
                "epoch": float(epoch),
                "loss": total_loss / total_rows,
                "reconstruction_loss": total_reconstruction_loss / total_rows,
                "latent_decorrelation_loss": total_decorrelation_loss / total_rows,
            }
        )

    return history


def train_from_dataframe_or_csv(
    data: pd.DataFrame | str | Path,
    metric_columns: Sequence[str] | None = None,
    *,
    champion_col: str = "champion_id",
    teamposition_col: str = "teamposition_id",
    build_col: str = "build_id",
    config: FullGameSemanticConfig | None = None,
    batch_size: int | str = 1024,
    shuffle: bool = True,
    epochs: int = 20,
    lr: float = 1.0e-3,
    device: str | torch.device = "cpu",
    noise_std: float = DEFAULT_METRIC_NOISE_STD,
    mask_prob: float = 0.0,
    latent_decorrelation_weight: float = DEFAULT_LATENT_DECORRELATION_WEIGHT,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    amp: bool = True,
    max_batch_size: int | None = None,
) -> tuple[FullGameAutoencoder, list[dict[str, float]]]:
    frame = _read_frame(data)
    device = _resolve_device(device)
    batch_size_request = _resolve_batch_size_request(batch_size)
    metric_columns = _metric_columns(metric_columns)
    if latent_decorrelation_weight < 0.0:
        raise ValueError("latent_decorrelation_weight must be non-negative")
    if config is None:
        config = FullGameSemanticConfig(
            n_champions=_infer_vocab_size(frame, champion_col),
            n_teampositions=_infer_vocab_size(frame, teamposition_col),
            n_builds=_infer_vocab_size(frame, build_col),
            metrics_dim=len(metric_columns),
        )
    elif config.metrics_dim != len(metric_columns):
        raise ValueError("config.metrics_dim must match the number of metric columns")
    _validate_frame_id_range(frame, champion_col, config.n_champions)
    _validate_frame_id_range(frame, teamposition_col, config.n_teampositions)
    _validate_frame_id_range(frame, build_col, config.n_builds)

    dataset = FullGameProfileDataset(
        frame,
        metric_columns,
        champion_col=champion_col,
        teamposition_col=teamposition_col,
        build_col=build_col,
    )
    model = FullGameAutoencoder(config)
    resolved_batch_size = _resolve_train_batch_size(
        model,
        dataset,
        batch_size_request,
        device,
        amp=amp,
        latent_decorrelation_weight=latent_decorrelation_weight,
        max_batch_size=max_batch_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=_resolve_pin_memory(pin_memory, device),
        persistent_workers=num_workers > 0,
    )
    history = train_autoencoder(
        model,
        dataloader,
        epochs=epochs,
        lr=lr,
        device=device,
        noise_std=noise_std,
        mask_prob=mask_prob,
        latent_decorrelation_weight=latent_decorrelation_weight,
        weight_decay=weight_decay,
        amp=amp,
    )
    for row in history:
        row["batch_size"] = float(resolved_batch_size)
    return model, history


def full_game_metric_columns(*, include_context: bool = True) -> tuple[str, ...]:
    """Default full-game metric set for full-game autoencoder training.

    The default is the complete per-identity catalogue: raw profile metrics,
    derived ratios/differences, and team-participation/role-matchup context
    features. The context features are not derivable from plain per-identity
    rows, so the input frame must already carry the (smoothed) context columns
    built with ``EmbeddingConfig(include_context_features=True)``.

    Pass ``include_context=False`` for the legacy profile-only 155-column
    surface.
    """
    base = raw_and_derived_metric_names()
    if include_context:
        return (*base, *CONTEXT_FEATURE_NAMES)
    return base


def evaluate_autoencoder(
    model: FullGameAutoencoder,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: str | torch.device,
    *,
    neighbor_k: int | None = None,
    max_neighbor_rows: int = 2048,
) -> dict[str, float]:
    """Evaluate clean reconstruction plus latent activity/grouping diagnostics."""
    if neighbor_k is not None and neighbor_k <= 0:
        raise ValueError("neighbor_k must be positive when set")
    if max_neighbor_rows <= 1:
        raise ValueError("max_neighbor_rows must be greater than 1")
    device = _resolve_device(device)
    was_training = model.training
    model.to(device)
    model.eval()
    non_blocking = device.type == "cuda"

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_values = 0
    total_rows = 0
    latents: list[np.ndarray] = []
    clean_metric_rows: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _batch_to_device(batch, device, non_blocking=non_blocking)
            reconstruction, latent = model(
                batch["champion_id"],
                batch["teamposition_id"],
                batch["build_id"],
                batch["metrics"],
            )
            error = reconstruction - batch["metrics"]
            total_squared_error += float(torch.sum(error.square()).detach().cpu())
            total_absolute_error += float(torch.sum(error.abs()).detach().cpu())
            total_values += int(error.numel())
            total_rows += int(batch["metrics"].shape[0])
            latents.append(latent.detach().cpu().numpy())
            if neighbor_k is not None:
                clean_metric_rows.append(batch["metrics"].detach().cpu().numpy())

    if was_training:
        model.train()
    if total_rows == 0 or total_values == 0:
        raise ValueError("dataloader must yield at least one row")

    latent_matrix = np.concatenate(latents, axis=0)
    latent_summary = _latent_summary(latent_matrix)
    if neighbor_k is not None:
        metric_matrix = np.concatenate(clean_metric_rows, axis=0)
        latent_summary.update(
            _semantic_neighborhood_summary(
                metric_matrix,
                latent_matrix,
                neighbor_k=neighbor_k,
                max_rows=max_neighbor_rows,
            )
        )
    return {
        "mse": total_squared_error / total_values,
        "mae": total_absolute_error / total_values,
        "rows": float(total_rows),
        **latent_summary,
    }


def extract_full_game_latents(
    model: FullGameAutoencoder,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: str | torch.device,
) -> pd.DataFrame:
    device = _resolve_device(device)
    was_training = model.training
    model.to(device)
    model.eval()
    non_blocking = device.type == "cuda"

    ids: dict[str, list[np.ndarray]] = {
        "champion_id": [],
        "teamposition_id": [],
        "build_id": [],
    }
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _batch_to_device(batch, device, non_blocking=non_blocking)
            latent = model.encoder(
                batch["champion_id"],
                batch["teamposition_id"],
                batch["build_id"],
                batch["metrics"],
            )
            for name in ids:
                ids[name].append(batch[name].detach().cpu().numpy())
            latents.append(latent.detach().cpu().numpy())

    if was_training:
        model.train()

    columns = ["champion_id", "teamposition_id", "build_id"]
    latent_columns = [f"latent_{i}" for i in range(model.config.latent_dim)]
    if not latents:
        return pd.DataFrame(columns=[*columns, *latent_columns])

    latent_matrix = np.concatenate(latents, axis=0)
    out = {
        "champion_id": np.concatenate(ids["champion_id"], axis=0),
        "teamposition_id": np.concatenate(ids["teamposition_id"], axis=0),
        "build_id": np.concatenate(ids["build_id"], axis=0),
    }
    for i, name in enumerate(latent_columns):
        out[name] = latent_matrix[:, i]
    return pd.DataFrame(out, columns=[*columns, *latent_columns])


def find_max_train_batch_size(
    model: FullGameAutoencoder,
    dataset: Dataset[dict[str, torch.Tensor]],
    device: str | torch.device = "auto",
    *,
    amp: bool = True,
    latent_decorrelation_weight: float = DEFAULT_LATENT_DECORRELATION_WEIGHT,
    max_batch_size: int | None = None,
) -> int:
    """Probe the largest one-step training batch that fits on the target device."""
    device = _resolve_device(device)
    if latent_decorrelation_weight < 0.0:
        raise ValueError("latent_decorrelation_weight must be non-negative")
    if len(dataset) <= 0:
        raise ValueError("dataset must contain at least one row")
    limit = (
        len(dataset)
        if max_batch_size is None
        else min(int(max_batch_size), len(dataset))
    )
    if limit <= 0:
        raise ValueError("max_batch_size must be positive when set")
    if device.type != "cuda":
        return limit

    amp_enabled = _resolve_amp(amp, device)
    _configure_cuda_fast_math(device)
    was_training = model.training
    model.to(device)
    model.train()

    last_good = 0
    candidate = 1
    while candidate <= limit:
        if _train_probe_fits(
            model,
            candidate,
            device,
            amp_enabled,
            latent_decorrelation_weight,
        ):
            last_good = candidate
            if candidate == limit:
                break
            candidate = min(candidate * 2, limit)
        else:
            break

    if last_good == 0:
        if was_training:
            model.train()
        else:
            model.eval()
        raise RuntimeError("CUDA memory cannot fit even batch_size=1")
    if last_good < limit:
        low = last_good + 1
        high = candidate - 1
        while low <= high:
            midpoint = (low + high) // 2
            if _train_probe_fits(
                model,
                midpoint,
                device,
                amp_enabled,
                latent_decorrelation_weight,
            ):
                last_good = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1

    if was_training:
        model.train()
    else:
        model.eval()
    torch.cuda.empty_cache()
    return last_good


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


def _semantic_neighborhood_summary(
    metric_matrix: np.ndarray,
    latent_matrix: np.ndarray,
    *,
    neighbor_k: int,
    max_rows: int,
) -> dict[str, float]:
    if metric_matrix.ndim != 2 or latent_matrix.ndim != 2:
        raise ValueError("metric_matrix and latent_matrix must be 2-D")
    if metric_matrix.shape[0] != latent_matrix.shape[0]:
        raise ValueError("metric_matrix and latent_matrix row counts must match")
    rows = metric_matrix.shape[0]
    if rows < 2:
        return {
            "latent_metric_neighbor_k": 0.0,
            "latent_metric_neighbor_recall": 0.0,
            "latent_metric_distance_corr": 0.0,
        }

    if rows > max_rows:
        sample_idx = np.linspace(0, rows - 1, num=max_rows, dtype=np.int64)
        metric_matrix = metric_matrix[sample_idx]
        latent_matrix = latent_matrix[sample_idx]
        rows = max_rows

    k = min(int(neighbor_k), rows - 1)
    metric_distances = _squared_distance_matrix(metric_matrix)
    latent_distances = _squared_distance_matrix(latent_matrix)
    metric_neighbors = _nearest_neighbor_indices(metric_distances, k)
    latent_neighbors = _nearest_neighbor_indices(latent_distances, k)
    recall = np.mean(
        [
            len(set(metric_neighbors[i]).intersection(latent_neighbors[i])) / k
            for i in range(rows)
        ]
    )

    tri = np.triu_indices(rows, k=1)
    metric_pair_distances = np.sqrt(metric_distances[tri])
    latent_pair_distances = np.sqrt(latent_distances[tri])
    if metric_pair_distances.std() == 0.0 or latent_pair_distances.std() == 0.0:
        distance_corr = 0.0
    else:
        distance_corr = float(
            np.corrcoef(metric_pair_distances, latent_pair_distances)[0, 1]
        )
        if not np.isfinite(distance_corr):
            distance_corr = 0.0

    return {
        "latent_metric_neighbor_k": float(k),
        "latent_metric_neighbor_recall": float(recall),
        "latent_metric_distance_corr": distance_corr,
    }


def _squared_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    values = matrix.astype(np.float32, copy=False)
    squared_norms = np.sum(values * values, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * values @ values.T
    return np.maximum(distances, 0.0)


def _nearest_neighbor_indices(distances: np.ndarray, k: int) -> np.ndarray:
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    return np.argpartition(distances, kth=k - 1, axis=1)[:, :k]


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


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _latent_norm(latent_dim: int, kind: LatentNorm) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(latent_dim)
    if kind == "layer":
        return nn.LayerNorm(latent_dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError("latent_norm must be one of: 'batch', 'layer', 'none'")


def _metric_columns(metric_columns: Sequence[str] | None) -> tuple[str, ...]:
    if metric_columns is None:
        return full_game_metric_columns()
    if isinstance(metric_columns, str):
        raise ValueError("metric_columns must be a sequence of column names")
    columns = tuple(metric_columns)
    if not columns:
        raise ValueError("metric_columns must not be empty")
    non_profile = sorted(set(columns) & _NON_PROFILE_METRIC_COLUMNS)
    if non_profile:
        raise ValueError(
            "metric_columns cannot include non-profile metadata columns: "
            f"{non_profile}"
        )
    return columns


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


def _metrics_array(frame: pd.DataFrame, metric_columns: Sequence[str]) -> np.ndarray:
    metric_values = _FrameMetricValues(frame)
    columns = [_metric_array(frame, metric_values, column) for column in metric_columns]
    values = np.stack(columns, axis=-1).astype(np.float32, copy=False)
    if values.ndim != 2:
        raise ValueError("metrics must be a 2-D matrix")
    if not np.isfinite(values).all():
        raise ValueError("metrics contain non-finite values")
    return values


class _FrameMetricValues(Mapping[str, np.ndarray]):
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, metric: str) -> np.ndarray:
        if metric not in self._cache:
            self._cache[metric] = _source_metric_array(self._frame, metric)
        return self._cache[metric]

    def __iter__(self):
        return iter(self._frame.columns)

    def __len__(self) -> int:
        return len(self._frame.columns)


def _metric_array(
    frame: pd.DataFrame,
    metric_values: _FrameMetricValues,
    column: str,
) -> np.ndarray:
    if _has_source_metric(frame, column):
        return _source_metric_array(frame, column)
    if column not in DERIVED_METRIC_FUNCS:
        raise ValueError(f"Missing required columns: ['{column}']")
    try:
        values = DERIVED_METRIC_FUNCS[column](metric_values).astype(np.float32)
    except KeyError as exc:
        missing = str(exc.args[0]) if exc.args else "unknown"
        raise ValueError(
            f"Cannot derive metric '{column}' because source metric '{missing}' is missing"
        ) from exc
    if values.ndim != 1:
        raise ValueError(f"{column} must resolve to a 1-D metric column")
    if len(values) != len(frame):
        raise ValueError(f"{column} row count must match frame")
    return values


def _has_source_metric(frame: pd.DataFrame, metric: str) -> bool:
    return metric in frame.columns or f"smoothed_{metric}" in frame.columns


def _source_metric_array(frame: pd.DataFrame, metric: str) -> np.ndarray:
    source = metric if metric in frame.columns else f"smoothed_{metric}"
    if source not in frame.columns:
        raise KeyError(metric)
    try:
        values = frame[source].to_numpy(dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must contain numeric metric values") from exc
    if values.ndim != 1:
        raise ValueError(f"{source} must be a 1-D metric column")
    return values


def _infer_vocab_size(frame: pd.DataFrame, column: str) -> int:
    _require_columns(frame, (column,))
    values = _id_array(frame, column)
    if values.size == 0:
        raise ValueError(f"{column} cannot infer vocab size from an empty frame")
    return int(values.max()) + 1


def _validate_frame_id_range(frame: pd.DataFrame, column: str, vocab_size: int) -> None:
    _require_columns(frame, (column,))
    values = _id_array(frame, column)
    if values.size == 0:
        return
    max_id = int(values.max())
    if max_id >= vocab_size:
        raise ValueError(f"{column} IDs must be in [0, {vocab_size})")


def _validate_metrics(metrics: torch.Tensor, metrics_dim: int) -> None:
    if metrics.ndim != 2:
        raise ValueError("metrics must have shape [batch, metrics_dim]")
    if metrics.shape[1] != metrics_dim:
        raise ValueError(
            f"metrics width must be {metrics_dim}, got {int(metrics.shape[1])}"
        )
    if not metrics.is_floating_point():
        raise ValueError("metrics must be a floating point tensor")


def _validate_ids(
    name: str,
    values: torch.Tensor,
    batch_size: int,
    vocab_size: int,
) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape [batch]")
    if values.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must match metrics")
    if values.dtype != torch.long:
        raise ValueError(f"{name} must be a LongTensor")
    if values.numel() == 0:
        return
    if values.device.type != "cpu":
        return
    min_id = int(values.min())
    max_id = int(values.max())
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(f"{name} IDs must be in [0, {vocab_size})")


def _batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        "champion_id": batch["champion_id"].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        ),
        "teamposition_id": batch["teamposition_id"].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        ),
        "build_id": batch["build_id"].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        ),
        "metrics": batch["metrics"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ),
    }


def _corrupt_metrics(
    metrics: torch.Tensor,
    *,
    noise_std: float,
    mask_prob: float,
) -> torch.Tensor:
    if noise_std == 0.0 and mask_prob == 0.0:
        return metrics
    out = metrics.clone()
    if noise_std > 0.0:
        out = out + torch.randn_like(out) * noise_std
    if mask_prob > 0.0:
        keep = torch.rand_like(out) >= mask_prob
        out = out * keep
    return out


def _read_frame(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.read_csv(Path(data))


def _resolve_batch_size_request(batch_size: int | str) -> int | str:
    if isinstance(batch_size, str):
        value = batch_size.strip().lower()
        if value == "auto":
            return "auto"
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError("batch_size must be a positive integer or 'auto'") from exc
    else:
        parsed = int(batch_size)
    if parsed <= 0:
        raise ValueError("batch_size must be positive")
    return parsed


def _resolve_train_batch_size(
    model: FullGameAutoencoder,
    dataset: Dataset[dict[str, torch.Tensor]],
    batch_size: int | str,
    device: torch.device,
    *,
    amp: bool,
    latent_decorrelation_weight: float,
    max_batch_size: int | None,
) -> int:
    if batch_size == "auto":
        return find_max_train_batch_size(
            model,
            dataset,
            device,
            amp=amp,
            latent_decorrelation_weight=latent_decorrelation_weight,
            max_batch_size=max_batch_size,
        )
    return int(batch_size)


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


def _configure_cuda_fast_math(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _train_probe_fits(
    model: FullGameAutoencoder,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    latent_decorrelation_weight: float,
) -> bool:
    try:
        model.zero_grad(set_to_none=True)
        champion_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        teamposition_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        build_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        metrics = torch.randn(
            batch_size,
            model.config.metrics_dim,
            dtype=torch.float32,
            device=device,
        )
        target = torch.randn_like(metrics)
        context = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
        with context:
            reconstruction, latent = model(
                champion_id,
                teamposition_id,
                build_id,
                metrics,
            )
        loss = nn.functional.mse_loss(reconstruction.float(), target)
        if latent_decorrelation_weight > 0.0:
            loss = loss + (
                latent_decorrelation_weight * _latent_decorrelation_loss(latent)
            )
        loss.backward()
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return False
        raise
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a full-game identity autoencoder")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metric-columns", nargs="*")
    metric_scope = parser.add_mutually_exclusive_group()
    metric_scope.add_argument(
        "--include-context",
        dest="include_context",
        action="store_true",
        help=(
            "Use the complete 215-column metric set with team-participation +"
            " role-matchup context features. This is the default when"
            " --metric-columns is omitted."
        ),
    )
    metric_scope.add_argument(
        "--profile-only",
        dest="include_context",
        action="store_false",
        help="Use only the legacy 155 raw+derived profile metrics.",
    )
    parser.add_argument("--champion-col", default="champion_id")
    parser.add_argument("--teamposition-col", default="teamposition_id")
    parser.add_argument("--build-col", default="build_id")
    parser.add_argument("--latent-output", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        default="1024",
        help="Positive integer batch size, or 'auto' to probe the largest CUDA training batch that fits.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help="Optional upper bound for --batch-size auto.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--latent-dim", type=int, default=640)
    parser.add_argument("--champion-embedding-dim", type=int, default=16)
    parser.add_argument("--teamposition-embedding-dim", type=int, default=4)
    parser.add_argument("--build-embedding-dim", type=int, default=8)
    parser.add_argument("--metrics-embedding-dim", type=int, default=160)
    parser.add_argument(
        "--latent-dropout",
        type=float,
        default=DEFAULT_LATENT_DROPOUT,
        help="Drop latent decoder inputs during training to spread reconstruction signal.",
    )
    parser.add_argument(
        "--metrics-hidden-dims",
        type=int,
        nargs="*",
        help="Hidden dimensions for the metrics MLP encoder.",
    )
    parser.add_argument("--noise-std", type=float, default=DEFAULT_METRIC_NOISE_STD)
    parser.add_argument("--mask-prob", type=float, default=0.0)
    parser.add_argument(
        "--latent-decorrelation-weight",
        type=float,
        default=DEFAULT_LATENT_DECORRELATION_WEIGHT,
        help="Small penalty on correlated latent dimensions; set to 0 to disable.",
    )
    parser.add_argument(
        "--latent-norm",
        choices=("batch", "layer", "none"),
        default="batch",
        help="Normalization applied to the fused latent vector.",
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        default=10,
        help="Evaluate metric/latent nearest-neighbor recall with this k.",
    )
    parser.add_argument(
        "--max-neighbor-rows",
        type=int,
        default=2048,
        help="Maximum rows sampled for nearest-neighbor latent diagnostics.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable CUDA automatic mixed precision.",
    )
    pin_memory = parser.add_mutually_exclusive_group()
    pin_memory.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Force DataLoader pinned host memory.",
    )
    pin_memory.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable DataLoader pinned host memory.",
    )
    parser.set_defaults(amp=True, pin_memory=None, include_context=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    frame = pd.read_csv(args.csv)
    metric_columns = (
        _metric_columns(args.metric_columns)
        if args.metric_columns
        else full_game_metric_columns(include_context=args.include_context)
    )
    config = FullGameSemanticConfig(
        n_champions=_infer_vocab_size(frame, args.champion_col),
        n_teampositions=_infer_vocab_size(frame, args.teamposition_col),
        n_builds=_infer_vocab_size(frame, args.build_col),
        metrics_dim=len(metric_columns),
        latent_dim=args.latent_dim,
        champion_embedding_dim=args.champion_embedding_dim,
        teamposition_embedding_dim=args.teamposition_embedding_dim,
        build_embedding_dim=args.build_embedding_dim,
        metrics_embedding_dim=args.metrics_embedding_dim,
        latent_dropout=args.latent_dropout,
        metrics_hidden_dims=(
            tuple(args.metrics_hidden_dims)
            if args.metrics_hidden_dims is not None
            else DEFAULT_METRICS_HIDDEN_DIMS
        ),
        latent_norm=args.latent_norm,
    )
    device = _resolve_device(args.device)
    model, history = train_from_dataframe_or_csv(
        frame,
        metric_columns,
        champion_col=args.champion_col,
        teamposition_col=args.teamposition_col,
        build_col=args.build_col,
        config=config,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        noise_std=args.noise_std,
        mask_prob=args.mask_prob,
        latent_decorrelation_weight=args.latent_decorrelation_weight,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        amp=args.amp,
        max_batch_size=args.max_batch_size,
    )
    print(
        f"batch_size={int(history[-1]['batch_size'])} "
        f"final_loss={history[-1]['loss']:.6f} "
        f"reconstruction_loss={history[-1]['reconstruction_loss']:.6f} "
        f"latent_decorrelation_loss={history[-1]['latent_decorrelation_loss']:.6f}"
    )

    eval_dataset = FullGameProfileDataset(
        frame,
        metric_columns,
        champion_col=args.champion_col,
        teamposition_col=args.teamposition_col,
        build_col=args.build_col,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=int(history[-1]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=_resolve_pin_memory(args.pin_memory, device),
        persistent_workers=args.num_workers > 0,
    )
    evaluation = evaluate_autoencoder(
        model,
        eval_dataloader,
        device,
        neighbor_k=args.neighbor_k,
        max_neighbor_rows=args.max_neighbor_rows,
    )
    print_parts = [
        f"clean_mse={evaluation['mse']:.6f}",
        f"clean_mae={evaluation['mae']:.6f}",
        f"latent_active_dims={int(evaluation['latent_active_dims'])}",
        f"latent_effective_rank={evaluation['latent_effective_rank']:.2f}",
    ]
    if "latent_metric_neighbor_recall" in evaluation:
        print_parts.append(
            f"neighbor_recall@{int(evaluation['latent_metric_neighbor_k'])}="
            f"{evaluation['latent_metric_neighbor_recall']:.3f}"
        )
        print_parts.append(
            f"distance_corr={evaluation['latent_metric_distance_corr']:.3f}"
        )
    print(" ".join(print_parts))

    if args.latent_output is not None:
        latents = extract_full_game_latents(model, eval_dataloader, device)
        args.latent_output.parent.mkdir(parents=True, exist_ok=True)
        latents.to_csv(args.latent_output, index=False)
        print(f"wrote_latents={args.latent_output}")


if __name__ == "__main__":
    main()
