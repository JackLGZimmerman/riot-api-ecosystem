"""Build a compact three-encoder sidecar artifact for HGNN semantic context.

The artifact is fit from train-only classification aggregates and contains one
row per observed `(championid, teamposition, build)` identity:

* static champion latents
* full-game champion/role/build latents
* temporal champion/role/build latents

Run with:
    python -m app.ml.build_encoder_sidecar --output app/ml/data/experiments/semantic_identity_sidecar_compact.npz
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.classification.encoder_common import _latent_summary as _latent_matrix_summary
from app.classification.embeddings.config import (
    ALL_METRICS,
    DERIVED_METRIC_FUNCS,
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.context_features import CONTEXT_FEATURE_NAMES
from app.classification.embeddings.load import LevelRows, load_all
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.temporal import build_temporal_tensors
from app.classification.full_game_encoder import (
    FullGameProfileDataset,
    FullGameSemanticConfig,
    OUTCOME_METRIC_COLUMNS,
    evaluate_autoencoder,
    extract_full_game_latents,
    train_from_dataframe_or_csv,
)
from app.classification.static_identity_encoder import (
    StaticIdentityAutoencoder,
    StaticIdentityConfig,
    StaticIdentityDataset,
    evaluate_static_autoencoder,
    extract_static_latents,
    static_identity_frame,
    train_static_autoencoder,
)
from app.classification.temporal_autoencoder import (
    SUPPORTED_TEMPORAL_ARCHITECTURES,
    TemporalAEConfig,
    evaluate_temporal_autoencoder,
    extract_temporal_latents,
    train_temporal,
)
from app.core.logging.logger import setup_logging_config
from app.core.utils.common import resolve_device_str as _resolve_device
from app.ml.encoder_sidecar import (
    EncoderSidecarLookup,
    build_encoder_sidecar_metadata,
    feature_hash,
    save_encoder_sidecar,
)
from app.ml.semantic_group_features import (
    SEMANTIC_GROUP_FEATURE_INDEX,
    SEMANTIC_GROUP_FEATURE_NAMES,
    build_identity_context_raw_from_metrics,
    build_semantic_group_features,
)
from app.core.utils.smoothing import apply_hierarchical_shrinkage

logger = logging.getLogger(__name__)

SIDECAR_WIDTH_PROFILES = ("compact", "standalone")
FULL_GAME_SUPPORT_WEIGHTING_MODES = ("none", "log1p")
FULL_GAME_SEMANTIC_TARGET_MODES = ("none", "soft_v2", "all_v2")
FULL_GAME_LATENT_EXPORT_MODES = ("autoencoder", "semantic_targets", "pca_whitened")
FULL_GAME_INPUT_SURFACES = (
    "full",
    "profile_only",
    "context_only",
    "raw_only",
    "derived_only",
    "raw_context",
)
FULL_GAME_IDENTITY_MODES = ("normal", "disabled")
MULTIVIEW_ALIGNMENT_OBJECTIVES = ("none", "vicreg", "barlow")
SOFT_V2_SEMANTIC_TARGET_NAMES = (
    "true_damage",
    "hard_cc_reliability",
    "frontline_intensity",
    "range_pressure",
    "burst_pressure",
    "scaling_pressure",
    "sustain_protection",
    "mixed_damage",
)


def _identity_frame(matrix) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    positions = sorted({str(key[1]) for key in matrix.keys})
    builds = sorted({str(key[2]) for key in matrix.keys})
    pos_idx = {label: idx for idx, label in enumerate(positions)}
    build_idx = {label: idx for idx, label in enumerate(builds)}
    frame = pd.DataFrame(matrix.matrix, columns=list(matrix.feature_names))
    frame.insert(0, "champion_id", [int(key[0]) for key in matrix.keys])
    frame.insert(1, "teamposition_id", [pos_idx[str(key[1])] for key in matrix.keys])
    frame.insert(2, "build_id", [build_idx[str(key[2])] for key in matrix.keys])
    return frame, pos_idx, build_idx


def _raw_identity_frame(
    rows: LevelRows,
    keys: list[tuple[int, str, str]],
    *,
    pos_idx: dict[str, int],
    build_idx: dict[str, int],
) -> pd.DataFrame:
    row_by_key = {
        tuple(rows.columns[column][idx] for column in rows.key_columns): idx
        for idx in range(rows.n)
    }
    row_indices = []
    for key in keys:
        source_idx = row_by_key.get((int(key[0]), str(key[1]), str(key[2])))
        if source_idx is None:
            raise ValueError(f"raw smoothed row missing identity key: {key}")
        row_indices.append(source_idx)
    idx_array = np.asarray(row_indices, dtype=np.int64)
    data: dict[str, np.ndarray] = {
        "champion_id": np.asarray([int(key[0]) for key in keys], dtype=np.int32),
        "teamposition_id": np.asarray(
            [pos_idx[str(key[1])] for key in keys],
            dtype=np.int32,
        ),
        "build_id": np.asarray(
            [build_idx[str(key[2])] for key in keys],
            dtype=np.int32,
        ),
    }
    metric_values: dict[str, np.ndarray] = {}
    for name in ALL_METRICS:
        source = f"smoothed_{name}"
        if source in rows.columns:
            values = rows.columns[source][idx_array].astype(np.float32)
            metric_values[name] = values
            data[name] = values
    for name, func in DERIVED_METRIC_FUNCS.items():
        try:
            data[name] = func(metric_values).astype(np.float32)
        except KeyError:
            continue
    for name in CONTEXT_FEATURE_NAMES:
        source = f"smoothed_{name}"
        if source in rows.columns:
            data[name] = rows.columns[source][idx_array].astype(np.float32)
    return pd.DataFrame(data)


def _train_static(
    champion_ids: np.ndarray,
    *,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[dict[int, np.ndarray], tuple[str, ...], dict[str, Any]]:
    torch.manual_seed(seed)
    frame = static_identity_frame(champion_ids)
    dataset = StaticIdentityDataset(frame)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = StaticIdentityAutoencoder(
        StaticIdentityConfig(
            continuous_dim=len(dataset.continuous_columns),
            latent_dim=latent_dim,
            latent_norm="batch",
        )
    )
    history = train_static_autoencoder(
        model,
        loader,
        epochs=epochs,
        device=device,
        noise_std=0.002,
        latent_decorrelation_weight=5.0e-4,
    )
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluation = evaluate_static_autoencoder(model, eval_loader, device=device)
    latents = extract_static_latents(model, eval_loader, device=device)
    latent_cols = [col for col in latents.columns if col.startswith("static_latent_")]
    by_champion = {
        int(row["champion_id"]): row[latent_cols].to_numpy(dtype=np.float32, copy=True)
        for _, row in latents.iterrows()
    }
    return by_champion, tuple(dataset.continuous_columns), {
        "history_last": history[-1],
        "evaluation": evaluation,
        "config": asdict(model.config),
    }


def _train_full_game(
    frame: pd.DataFrame,
    metric_columns: tuple[str, ...],
    *,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    width_profile: str,
    sample_weight: np.ndarray | None,
    support_weighting: str,
    semantic_targets: np.ndarray | None,
    semantic_target_names: tuple[str, ...],
    semantic_target_mode: str,
    semantic_target_weight: float,
    identity_mode: str,
    allow_outcome_metrics: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    config = full_game_sidecar_config(
        frame,
        metric_columns,
        latent_dim=latent_dim,
        width_profile=width_profile,
        semantic_target_dim=0 if semantic_targets is None else semantic_targets.shape[1],
        identity_mode=identity_mode,
    )
    model, history = train_from_dataframe_or_csv(
        frame,
        metric_columns,
        config=config,
        batch_size=batch_size,
        epochs=epochs,
        lr=1.0e-3,
        device=device,
        noise_std=0.002,
        mask_prob=0.0,
        latent_decorrelation_weight=5.0e-4,
        num_workers=0,
        amp=True,
        sample_weight=sample_weight,
        semantic_targets=semantic_targets,
        semantic_loss_weight=semantic_target_weight,
        allow_outcome_metrics=allow_outcome_metrics,
    )
    dataset = FullGameProfileDataset(
        frame,
        metric_columns,
        semantic_targets=semantic_targets,
        allow_outcome_metrics=allow_outcome_metrics,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluation = evaluate_autoencoder(model, loader, device=device)
    latents = extract_full_game_latents(model, loader, device=device)
    latent_cols = [col for col in latents.columns if col.startswith("latent_")]
    return latents[latent_cols].to_numpy(dtype=np.float32, copy=True), {
        "history_last": history[-1],
        "evaluation": evaluation,
        "config": asdict(model.config),
        "latent_export": "autoencoder",
        "support_weighting": support_weighting,
        "sample_weight_summary": _sample_weight_summary(sample_weight),
        "semantic_target_mode": semantic_target_mode,
        "semantic_target_names": list(semantic_target_names),
        "semantic_target_weight": semantic_target_weight,
        "semantic_target_summary": _semantic_target_summary(semantic_targets),
        "identity_mode": identity_mode,
    }


def _train_temporal(
    *,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    mask_as_input: bool,
    zero_unobserved_input: bool,
    architecture: str,
    width_profile: str,
) -> tuple[dict[tuple[int, str, str], np.ndarray], tuple[str, ...], dict[str, Any]]:
    tensors = build_temporal_tensors(EmbeddingConfig(split="train"), use_cache=True)
    config = temporal_sidecar_config(
        latent_dim=latent_dim,
        mask_as_input=mask_as_input,
        zero_unobserved_input=zero_unobserved_input,
        architecture=architecture,
        width_profile=width_profile,
    )
    model, history = train_temporal(
        tensors,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        cfg=config,
        latent_decorrelation_weight=5.0e-4,
        seed=seed,
    )
    evaluation = evaluate_temporal_autoencoder(model, tensors, device=device, batch_size=batch_size)
    latent_matrix = extract_temporal_latents(model, tensors, device=device)
    by_key = {
        (int(key[0]), str(key[1]), str(key[2])): latent_matrix[idx].astype(np.float32, copy=True)
        for idx, key in enumerate(tensors.keys)
    }
    return by_key, tuple(tensors.metric_names), {
        "history_last": history[-1],
        "evaluation": evaluation,
        "config": asdict(config),
        "rows": len(tensors.keys),
    }


def full_game_sidecar_config(
    frame: pd.DataFrame,
    metric_columns: tuple[str, ...],
    *,
    latent_dim: int,
    width_profile: str,
    semantic_target_dim: int = 0,
    identity_mode: str = "normal",
) -> FullGameSemanticConfig:
    """Return the full-game encoder config used for sidecar export."""
    _require_width_profile(width_profile)
    base = {
        "n_champions": int(frame["champion_id"].max()) + 1,
        "n_teampositions": int(frame["teamposition_id"].max()) + 1,
        "n_builds": int(frame["build_id"].max()) + 1,
        "metrics_dim": len(metric_columns),
        "latent_dim": latent_dim,
        "semantic_target_dim": int(semantic_target_dim),
        "latent_norm": "batch",
        "identity_mode": identity_mode,
    }
    if width_profile == "standalone":
        return FullGameSemanticConfig(**base)
    return FullGameSemanticConfig(
        **base,
        metrics_embedding_dim=min(96, max(32, latent_dim)),
        metrics_hidden_dims=(192, 96),
        fusion_hidden_dims=(96,),
        decoder_hidden_dims=(128, 96),
        latent_dropout=0.05,
    )


def temporal_sidecar_config(
    *,
    latent_dim: int,
    mask_as_input: bool,
    zero_unobserved_input: bool = True,
    architecture: str,
    width_profile: str,
) -> TemporalAEConfig:
    """Return the temporal encoder config used for sidecar export."""
    _require_width_profile(width_profile)
    if width_profile == "standalone":
        return TemporalAEConfig(
            latent_dim=latent_dim,
            mask_as_input=mask_as_input,
            zero_unobserved_input=zero_unobserved_input,
            architecture=architecture,
        )
    return TemporalAEConfig(
        metric_embed_dim=min(48, max(16, latent_dim)),
        latent_dim=latent_dim,
        hidden=512,
        dropout=0.02,
        mask_as_input=mask_as_input,
        zero_unobserved_input=zero_unobserved_input,
        architecture=architecture,
    )


def _require_width_profile(value: str) -> None:
    if value not in SIDECAR_WIDTH_PROFILES:
        known = ", ".join(SIDECAR_WIDTH_PROFILES)
        raise ValueError(f"width_profile must be one of: {known}")


def select_full_game_metric_columns(
    metric_columns: tuple[str, ...],
    *,
    surface: str,
    allow_outcome_metrics: bool = False,
) -> tuple[str, ...]:
    """Return the requested full-game input surface while preserving matrix order."""
    _require_full_game_input_surface(surface)
    all_columns = tuple(
        str(name)
        for name in metric_columns
        if allow_outcome_metrics or str(name) not in OUTCOME_METRIC_COLUMNS
    )
    if surface == "full":
        selected = all_columns
    else:
        raw = frozenset(
            name
            for name in ALL_METRICS
            if allow_outcome_metrics or name not in OUTCOME_METRIC_COLUMNS
        )
        derived = frozenset(DERIVED_METRIC_FUNCS)
        context = frozenset(CONTEXT_FEATURE_NAMES)
        if surface == "profile_only":
            allowed = raw | derived
        elif surface == "context_only":
            allowed = context
        elif surface == "raw_only":
            allowed = raw
        elif surface == "derived_only":
            allowed = derived
        elif surface == "raw_context":
            allowed = raw | context
        else:  # pragma: no cover - guarded by _require_full_game_input_surface
            allowed = frozenset()
        selected = tuple(name for name in all_columns if name in allowed)
    if not selected:
        raise ValueError(f"full-game input surface {surface!r} selected no metrics")
    return selected


def _require_full_game_input_surface(value: str) -> None:
    if value not in FULL_GAME_INPUT_SURFACES:
        known = ", ".join(FULL_GAME_INPUT_SURFACES)
        raise ValueError(f"full_game_input_surface must be one of: {known}")


def _require_full_game_identity_mode(value: str) -> None:
    if value not in FULL_GAME_IDENTITY_MODES:
        known = ", ".join(FULL_GAME_IDENTITY_MODES)
        raise ValueError(f"full_game_identity_mode must be one of: {known}")


def full_game_sample_weight(
    support: np.ndarray,
    *,
    mode: str,
) -> np.ndarray | None:
    """Return optional train weights derived from support counts."""
    _require_support_weighting_mode(mode)
    if mode == "none":
        return None
    values = np.asarray(support, dtype=np.float32)
    weights = np.log1p(np.maximum(values, 0.0)).astype(np.float32)
    positive = weights > 0.0
    if not bool(positive.any()):
        return np.ones_like(weights, dtype=np.float32)
    weights = weights / float(weights[positive].mean())
    return weights


def full_game_semantic_targets(
    frame: pd.DataFrame,
    *,
    build_vocab: tuple[str, ...],
    mode: str,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """Build optional train-only soft semantic targets for full-game latents."""
    _require_semantic_target_mode(mode)
    if mode == "none":
        return None, tuple()
    context_raw = build_identity_context_raw_from_metrics(frame)
    repeated_context = np.repeat(context_raw[:, None, :], 10, axis=1)
    champions = np.repeat(
        frame["champion_id"].to_numpy(dtype=np.int64, copy=True)[:, None],
        10,
        axis=1,
    )
    builds = np.repeat(
        frame["build_id"].to_numpy(dtype=np.int64, copy=True)[:, None],
        10,
        axis=1,
    )
    semantic_features = build_semantic_group_features(
        context_raw=repeated_context,
        champion_id=champions,
        build_id=builds,
        build_vocab=build_vocab,
    )[:, 0, :]
    target_names = (
        tuple(SEMANTIC_GROUP_FEATURE_NAMES)
        if mode == "all_v2"
        else SOFT_V2_SEMANTIC_TARGET_NAMES
    )
    indices = [SEMANTIC_GROUP_FEATURE_INDEX[name] for name in target_names]
    return semantic_features[:, indices].astype(np.float32, copy=True), target_names


def semantic_targets_as_latents(
    semantic_targets: np.ndarray,
    *,
    latent_dim: int,
) -> np.ndarray:
    """Project semantic targets directly into a fixed-width latent block."""
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    targets = np.asarray(semantic_targets, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] <= 0:
        raise ValueError("semantic_targets must be a non-empty 2-D matrix")
    if not np.isfinite(targets).all():
        raise ValueError("semantic_targets must contain finite values")
    mean = targets.mean(axis=0, keepdims=True)
    std = np.maximum(targets.std(axis=0, keepdims=True), 1.0e-6)
    normalized = (targets - mean) / std
    latents = np.zeros((targets.shape[0], int(latent_dim)), dtype=np.float32)
    width = min(int(latent_dim), normalized.shape[1])
    latents[:, :width] = normalized[:, :width]
    return latents


def full_game_pca_latents(
    frame: pd.DataFrame,
    metric_columns: tuple[str, ...],
    *,
    latent_dim: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project full-game metric inputs into a deterministic PCA-whitened block."""
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if not metric_columns:
        raise ValueError("metric_columns must be non-empty")
    metrics = frame.loc[:, list(metric_columns)].to_numpy(dtype=np.float32, copy=True)
    if metrics.ndim != 2 or metrics.shape[0] == 0:
        raise ValueError("full-game metrics must be a non-empty 2-D matrix")
    if not np.isfinite(metrics).all():
        raise ValueError("full-game metrics contain non-finite values")
    centered = metrics.astype(np.float64, copy=False)
    centered = centered - centered.mean(axis=0, keepdims=True)
    _, singular_values, components_t = np.linalg.svd(centered, full_matrices=False)
    width = min(int(latent_dim), int(components_t.shape[0]))
    scores = centered @ components_t[:width].T
    score_mean = scores.mean(axis=0, keepdims=True)
    score_std = np.maximum(scores.std(axis=0, keepdims=True), 1.0e-6)
    normalized = (scores - score_mean) / score_std
    latents = np.zeros((metrics.shape[0], int(latent_dim)), dtype=np.float32)
    latents[:, :width] = normalized.astype(np.float32, copy=False)
    variance = singular_values**2
    total_variance = float(variance.sum())
    explained = (
        float(variance[:width].sum() / total_variance)
        if total_variance > 0.0
        else 0.0
    )
    return latents, {
        "pca_active_dims": float(width),
        "pca_explained_variance_ratio": explained,
        "pca_top_singular_value": float(singular_values[0])
        if singular_values.size
        else 0.0,
    }


def align_multiview_latents(
    *,
    static_latents: np.ndarray,
    full_game_latents: np.ndarray,
    temporal_latents: np.ndarray,
    support: np.ndarray,
    objective: str,
    alignment_weight: float,
    alignment_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply a small multi-view alignment transform while preserving sidecar widths."""
    _require_multiview_alignment_objective(objective)
    if objective == "none":
        return static_latents, full_game_latents, temporal_latents, {"objective": "none"}
    if alignment_weight <= 0.0:
        raise ValueError("multiview alignment weight must be positive when enabled")
    if alignment_dim <= 0:
        raise ValueError("multiview alignment dim must be positive")
    if epochs <= 0:
        raise ValueError("multiview alignment epochs must be positive")
    if batch_size <= 0:
        raise ValueError("multiview alignment batch size must be positive")
    if lr <= 0.0:
        raise ValueError("multiview alignment lr must be positive")

    arrays = (
        np.asarray(static_latents, dtype=np.float32),
        np.asarray(full_game_latents, dtype=np.float32),
        np.asarray(temporal_latents, dtype=np.float32),
    )
    n_rows = int(arrays[0].shape[0])
    if any(values.ndim != 2 or values.shape[0] != n_rows for values in arrays):
        raise ValueError("all multiview latent arrays must be 2-D with matching rows")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("multiview latent arrays must contain finite values")

    common_dim = min(int(alignment_dim), *(int(values.shape[1]) for values in arrays))
    if common_dim <= 0:
        raise ValueError("multiview alignment common dim must be positive")

    torch.manual_seed(seed)
    resolved_device = torch.device(_resolve_device(device))
    standardized, stats = zip(
        *(_standardize_latent_matrix(values) for values in arrays),
        strict=True,
    )
    sample_weight = _multiview_sample_weight(support, n_rows=n_rows)
    dataset = TensorDataset(
        *(torch.as_tensor(values, dtype=torch.float32) for values in standardized),
        torch.as_tensor(sample_weight, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), n_rows),
        shuffle=True,
    )
    transforms = torch.nn.ModuleList(
        torch.nn.Linear(int(values.shape[1]), int(values.shape[1]), bias=False)
        for values in arrays
    ).to(resolved_device)
    with torch.no_grad():
        for layer in transforms:
            layer.weight.copy_(
                torch.eye(
                    layer.weight.shape[0],
                    layer.weight.shape[1],
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
            )
    optimizer = torch.optim.Adam(transforms.parameters(), lr=float(lr))
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        totals = {
            "loss": 0.0,
            "anchor_loss": 0.0,
            "alignment_loss": 0.0,
            "variance_loss": 0.0,
            "covariance_loss": 0.0,
            "rows": 0.0,
        }
        for batch in loader:
            *view_batch, weights = batch
            views = [values.to(resolved_device) for values in view_batch]
            weights = weights.to(resolved_device)
            transformed = [
                layer(values) for layer, values in zip(transforms, views, strict=True)
            ]
            anchor_loss = sum(
                _weighted_row_mse(pred, target, weights)
                for pred, target in zip(transformed, views, strict=True)
            ) / float(len(transformed))
            prefixes = [values[:, :common_dim] for values in transformed]
            if objective == "vicreg":
                alignment_loss, variance_loss, covariance_loss = _vicreg_terms(
                    prefixes,
                    weights,
                )
                objective_loss = (
                    alignment_loss + 0.1 * variance_loss + 0.01 * covariance_loss
                )
            else:
                alignment_loss = _barlow_terms(prefixes)
                variance_loss = alignment_loss.new_tensor(0.0)
                covariance_loss = alignment_loss.new_tensor(0.0)
                objective_loss = alignment_loss
            loss = anchor_loss + float(alignment_weight) * objective_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            rows = float(weights.numel())
            totals["loss"] += float(loss.detach().cpu()) * rows
            totals["anchor_loss"] += float(anchor_loss.detach().cpu()) * rows
            totals["alignment_loss"] += float(alignment_loss.detach().cpu()) * rows
            totals["variance_loss"] += float(variance_loss.detach().cpu()) * rows
            totals["covariance_loss"] += float(covariance_loss.detach().cpu()) * rows
            totals["rows"] += rows

        denom = max(totals.pop("rows"), 1.0)
        history.append(
            {
                "epoch": float(epoch),
                **{key: value / denom for key, value in totals.items()},
            }
        )

    aligned_arrays: list[np.ndarray] = []
    with torch.no_grad():
        for layer, values, (mean, std) in zip(
            transforms,
            standardized,
            stats,
            strict=True,
        ):
            tensor = torch.as_tensor(values, dtype=torch.float32, device=resolved_device)
            aligned = layer(tensor).cpu().numpy().astype(np.float32, copy=False)
            aligned_arrays.append(aligned * std + mean)

    summary = {
        "objective": objective,
        "alignment_weight": float(alignment_weight),
        "alignment_dim": float(alignment_dim),
        "common_dim": float(common_dim),
        "epochs": float(epochs),
        "batch_size": float(batch_size),
        "lr": float(lr),
        "history_last": history[-1],
        "sample_weight_summary": _sample_weight_summary(sample_weight),
        "latent_before": {
            "static": _latent_matrix_summary(arrays[0]),
            "full_game": _latent_matrix_summary(arrays[1]),
            "temporal": _latent_matrix_summary(arrays[2]),
        },
        "latent_after": {
            "static": _latent_matrix_summary(aligned_arrays[0]),
            "full_game": _latent_matrix_summary(aligned_arrays[1]),
            "temporal": _latent_matrix_summary(aligned_arrays[2]),
        },
    }
    return aligned_arrays[0], aligned_arrays[1], aligned_arrays[2], summary


def _require_support_weighting_mode(value: str) -> None:
    if value not in FULL_GAME_SUPPORT_WEIGHTING_MODES:
        known = ", ".join(FULL_GAME_SUPPORT_WEIGHTING_MODES)
        raise ValueError(f"support_weighting must be one of: {known}")


def _require_semantic_target_mode(value: str) -> None:
    if value not in FULL_GAME_SEMANTIC_TARGET_MODES:
        known = ", ".join(FULL_GAME_SEMANTIC_TARGET_MODES)
        raise ValueError(f"semantic_target_mode must be one of: {known}")


def _require_latent_export_mode(value: str) -> None:
    if value not in FULL_GAME_LATENT_EXPORT_MODES:
        known = ", ".join(FULL_GAME_LATENT_EXPORT_MODES)
        raise ValueError(f"full_game_latent_export must be one of: {known}")


def _require_multiview_alignment_objective(value: str) -> None:
    if value not in MULTIVIEW_ALIGNMENT_OBJECTIVES:
        known = ", ".join(MULTIVIEW_ALIGNMENT_OBJECTIVES)
        raise ValueError(f"multiview_alignment_objective must be one of: {known}")


def _standardize_latent_matrix(
    values: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    matrix = np.asarray(values, dtype=np.float32)
    mean = matrix.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
    std = np.maximum(matrix.std(axis=0, keepdims=True), 1.0e-6).astype(
        np.float32,
        copy=False,
    )
    return ((matrix - mean) / std).astype(np.float32, copy=False), (mean, std)


def _multiview_sample_weight(support: np.ndarray, *, n_rows: int) -> np.ndarray:
    values = np.asarray(support, dtype=np.float32).reshape(-1)
    if values.shape[0] != n_rows:
        raise ValueError("support must have one value per multiview latent row")
    weights = np.log1p(np.maximum(values, 0.0)).astype(np.float32, copy=False)
    weights[~np.isfinite(weights)] = 0.0
    positive = weights > 0.0
    if not bool(positive.any()):
        return np.ones(n_rows, dtype=np.float32)
    weights = weights / float(weights[positive].mean())
    return weights.astype(np.float32, copy=False)


def _weighted_row_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    row_loss = (pred - target).square().mean(dim=1)
    denom = torch.clamp(weight.sum(), min=1.0e-6)
    return (row_loss * weight).sum() / denom


def _view_pairs(values: list[torch.Tensor]) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (values[left], values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


def _vicreg_terms(
    views: list[torch.Tensor],
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = _view_pairs(views)
    alignment = sum(
        _weighted_row_mse(left, right, weight) for left, right in pairs
    ) / float(len(pairs))
    variance = sum(_variance_loss(view) for view in views) / float(len(views))
    covariance = sum(_covariance_loss(view) for view in views) / float(len(views))
    return alignment, variance, covariance


def _variance_loss(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return values.new_tensor(0.0)
    std = torch.sqrt(values.var(dim=0, unbiased=False) + 1.0e-4)
    return torch.relu(1.0 - std).mean()


def _covariance_loss(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return values.new_tensor(0.0)
    centered = values - values.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / float(max(values.shape[0] - 1, 1))
    return _off_diagonal(cov).square().sum() / float(values.shape[1])


def _barlow_terms(views: list[torch.Tensor]) -> torch.Tensor:
    losses = []
    for left, right in _view_pairs(views):
        left_norm = (left - left.mean(dim=0, keepdim=True)) / torch.clamp(
            left.std(dim=0, unbiased=False, keepdim=True),
            min=1.0e-4,
        )
        right_norm = (right - right.mean(dim=0, keepdim=True)) / torch.clamp(
            right.std(dim=0, unbiased=False, keepdim=True),
            min=1.0e-4,
        )
        corr = left_norm.T @ right_norm / float(max(left_norm.shape[0], 1))
        on_diag = torch.diagonal(corr).add(-1.0).square().sum()
        off_diag = _off_diagonal(corr).square().sum()
        losses.append(on_diag + 0.005 * off_diag)
    return sum(losses) / float(len(losses))


def _off_diagonal(values: torch.Tensor) -> torch.Tensor:
    rows, cols = values.shape
    if rows != cols:
        raise ValueError("off-diagonal helper expects a square matrix")
    if rows <= 1:
        return values.new_empty((0,))
    return values.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


def _sample_weight_summary(sample_weight: np.ndarray | None) -> dict[str, float | str]:
    if sample_weight is None:
        return {"mode": "none"}
    values = np.asarray(sample_weight, dtype=np.float32)
    return {
        "mode": "provided",
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "zero_rate": float(np.mean(values <= 0.0)),
    }


def _semantic_target_summary(
    semantic_targets: np.ndarray | None,
) -> dict[str, float | str]:
    if semantic_targets is None:
        return {"mode": "none"}
    values = np.asarray(semantic_targets, dtype=np.float32)
    return {
        "mode": "provided",
        "rows": float(values.shape[0]),
        "dims": float(values.shape[1]),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def _reuse_static_full_game_blocks(
    path: Path,
    *,
    champion_id: np.ndarray,
    teamposition: np.ndarray,
    build: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    sidecar = EncoderSidecarLookup.load(path)
    if not np.array_equal(sidecar.champion_id.astype(np.int32, copy=False), champion_id):
        raise ValueError(f"reused sidecar champion_id rows do not match: {path}")
    if not np.array_equal(sidecar.teamposition.astype(str), teamposition.astype(str)):
        raise ValueError(f"reused sidecar teamposition rows do not match: {path}")
    if not np.array_equal(sidecar.build.astype(str), build.astype(str)):
        raise ValueError(f"reused sidecar build rows do not match: {path}")
    return (
        sidecar.static_latents.astype(np.float32, copy=True),
        sidecar.full_game_latents.astype(np.float32, copy=True),
        sidecar.support.astype(np.float32, copy=True),
        copy.deepcopy(sidecar.metadata),
    )


def _metadata_with_reused_static_full_game(
    base_metadata: dict[str, Any],
    *,
    base_sidecar_path: Path,
    temporal_features: tuple[str, ...],
    temporal_summary: dict[str, Any],
    rows: int,
    temporal_missing: int,
    elapsed_seconds: float,
    device: str,
) -> dict[str, Any]:
    metadata = copy.deepcopy(base_metadata)
    metadata.setdefault("feature_hashes", {})
    metadata["feature_hashes"]["temporal"] = feature_hash(temporal_features)
    metadata.setdefault("split_metadata", {})
    metadata["split_metadata"].update({"fit_split": "train", "source_split": "train"})
    metadata.setdefault("encoder_configs", {})
    metadata["encoder_configs"]["temporal"] = temporal_summary["config"]
    metadata.setdefault("export", {})
    metadata["export"].update(
        {
            "kind": "compact_hgnn_semantic_sidecar",
            "device": device,
            "rows": rows,
            "temporal_missing_rows": temporal_missing,
            "elapsed_seconds": elapsed_seconds,
            "reused_static_full_game_from": str(base_sidecar_path),
        }
    )
    metadata["temporal_encoder_ablation"] = {
        "kind": "replace_temporal_only",
        "reused_static_full_game_from": str(base_sidecar_path),
        "temporal_config": temporal_summary["config"],
        "temporal_evaluation": temporal_summary.get("evaluation", {}),
    }
    return metadata


def build_sidecar(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    _require_latent_export_mode(args.full_game_latent_export)
    _require_full_game_input_surface(args.full_game_input_surface)
    _require_full_game_identity_mode(args.full_game_identity_mode)
    _require_multiview_alignment_objective(args.multiview_alignment_objective)
    if args.full_game_semantic_target_weight < 0.0:
        raise ValueError("full_game_semantic_target_weight must be non-negative")
    if args.multiview_alignment_weight < 0.0:
        raise ValueError("multiview_alignment_weight must be non-negative")
    if (
        args.multiview_alignment_objective == "none"
        and args.multiview_alignment_weight > 0.0
    ):
        raise ValueError(
            "--multiview-alignment-weight requires "
            "--multiview-alignment-objective"
        )
    if (
        args.full_game_semantic_target_mode == "none"
        and args.full_game_semantic_target_weight > 0.0
    ):
        raise ValueError(
            "full_game_semantic_target_weight requires "
            "--full-game-semantic-target-mode"
        )
    if (
        args.full_game_latent_export == "semantic_targets"
        and args.full_game_semantic_target_mode == "none"
    ):
        raise ValueError(
            "--full-game-latent-export semantic_targets requires "
            "--full-game-semantic-target-mode"
        )
    device = _resolve_device(args.device)
    logger.info("Building train-only classification matrices")
    embedding_config = EmbeddingConfig(split="train")
    smoothed_rows = apply_hierarchical_shrinkage(
        load_all(embedding_config),
        embedding_config,
    )
    matrices = build_all_matrices(smoothed_rows, embedding_config)
    baseline = matrices[IdentityType.BASELINE]
    identity_frame, pos_idx, build_idx = _identity_frame(baseline)
    build_vocab = tuple(label for label, _idx in sorted(build_idx.items(), key=lambda item: item[1]))
    full_metric_columns = tuple(str(name) for name in baseline.feature_names)
    metric_columns = select_full_game_metric_columns(
        full_metric_columns,
        surface=args.full_game_input_surface,
        allow_outcome_metrics=args.full_game_allow_outcome_metrics,
    )
    keys = [(int(key[0]), str(key[1]), str(key[2])) for key in baseline.keys]
    raw_identity_frame = _raw_identity_frame(
        smoothed_rows[IdentityType.BASELINE],
        keys,
        pos_idx=pos_idx,
        build_idx=build_idx,
    )
    champion_id = np.asarray([key[0] for key in keys], dtype=np.int32)
    teamposition = np.asarray([key[1] for key in keys])
    build = np.asarray([key[2] for key in keys])
    support = baseline.matchups.astype(np.float32, copy=False)

    reused_metadata: dict[str, Any] | None = None
    if args.reuse_static_full_game_from is None:
        logger.info("Training static sidecar encoder rows=%d dim=%d", len(set(champion_id)), args.static_latent_dim)
        static_by_champion, static_features, static_summary = _train_static(
            champion_id,
            latent_dim=args.static_latent_dim,
            epochs=args.static_epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        static_latents = np.stack([static_by_champion[int(champ)] for champ in champion_id])

        logger.info("Training full-game sidecar encoder rows=%d dim=%d", len(keys), args.full_game_latent_dim)
        semantic_targets, semantic_target_names = full_game_semantic_targets(
            raw_identity_frame,
            build_vocab=build_vocab,
            mode=args.full_game_semantic_target_mode,
        )
        if args.full_game_latent_export == "semantic_targets":
            if semantic_targets is None:
                raise ValueError("semantic target export requires semantic targets")
            full_game_latents = semantic_targets_as_latents(
                semantic_targets,
                latent_dim=args.full_game_latent_dim,
            )
            full_game_summary = {
                "history_last": {
                    "epoch": 0.0,
                    "loss": 0.0,
                    "reconstruction_loss": 0.0,
                    "semantic_loss": 0.0,
                    "batch_size": float(args.batch_size),
                },
                "evaluation": {
                    "rows": float(full_game_latents.shape[0]),
                    "semantic_direct_dims": float(semantic_targets.shape[1]),
                    **_latent_matrix_summary(full_game_latents),
                },
                "config": {
                    "latent_dim": args.full_game_latent_dim,
                    "latent_export": args.full_game_latent_export,
                    "metrics_dim": len(metric_columns),
                    "input_surface": args.full_game_input_surface,
                    "identity_mode": args.full_game_identity_mode,
                },
                "latent_export": args.full_game_latent_export,
                "support_weighting": args.full_game_support_weighting,
                "sample_weight_summary": _sample_weight_summary(
                    full_game_sample_weight(
                        support,
                        mode=args.full_game_support_weighting,
                    )
                ),
                "semantic_target_mode": args.full_game_semantic_target_mode,
                "semantic_target_names": list(semantic_target_names),
                "semantic_target_weight": args.full_game_semantic_target_weight,
                "semantic_target_summary": _semantic_target_summary(semantic_targets),
                "input_surface": args.full_game_input_surface,
                "identity_mode": args.full_game_identity_mode,
            }
        elif args.full_game_latent_export == "pca_whitened":
            full_game_latents, pca_summary = full_game_pca_latents(
                identity_frame,
                metric_columns,
                latent_dim=args.full_game_latent_dim,
            )
            full_game_summary = {
                "history_last": {
                    "epoch": 0.0,
                    "loss": 0.0,
                    "reconstruction_loss": 0.0,
                    "semantic_loss": 0.0,
                    "batch_size": float(args.batch_size),
                },
                "evaluation": {
                    "rows": float(full_game_latents.shape[0]),
                    **pca_summary,
                    **_latent_matrix_summary(full_game_latents),
                },
                "config": {
                    "latent_dim": args.full_game_latent_dim,
                    "latent_export": args.full_game_latent_export,
                    "metrics_dim": len(metric_columns),
                    "input_surface": args.full_game_input_surface,
                    "identity_mode": args.full_game_identity_mode,
                },
                "latent_export": args.full_game_latent_export,
                "support_weighting": args.full_game_support_weighting,
                "sample_weight_summary": _sample_weight_summary(
                    full_game_sample_weight(
                        support,
                        mode=args.full_game_support_weighting,
                    )
                ),
                "semantic_target_mode": args.full_game_semantic_target_mode,
                "semantic_target_names": list(semantic_target_names),
                "semantic_target_weight": args.full_game_semantic_target_weight,
                "semantic_target_summary": _semantic_target_summary(semantic_targets),
                "input_surface": args.full_game_input_surface,
                "identity_mode": args.full_game_identity_mode,
            }
        else:
            full_game_latents, full_game_summary = _train_full_game(
                identity_frame,
                metric_columns,
                latent_dim=args.full_game_latent_dim,
                epochs=args.full_game_epochs,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed + 1,
                width_profile=args.full_game_width_profile,
                sample_weight=full_game_sample_weight(
                    support,
                    mode=args.full_game_support_weighting,
                ),
                support_weighting=args.full_game_support_weighting,
                semantic_targets=semantic_targets,
                semantic_target_names=semantic_target_names,
                semantic_target_mode=args.full_game_semantic_target_mode,
                semantic_target_weight=args.full_game_semantic_target_weight,
                identity_mode=args.full_game_identity_mode,
                allow_outcome_metrics=args.full_game_allow_outcome_metrics,
            )
            full_game_summary["input_surface"] = args.full_game_input_surface
    else:
        logger.info(
            "Reusing static/full-game sidecar blocks from %s",
            args.reuse_static_full_game_from,
        )
        static_latents, full_game_latents, support, reused_metadata = (
            _reuse_static_full_game_blocks(
                args.reuse_static_full_game_from,
                champion_id=champion_id,
                teamposition=teamposition,
                build=build,
            )
        )
        static_features = tuple()
        static_summary = {
            "reused_from": str(args.reuse_static_full_game_from),
            "config": reused_metadata.get("encoder_configs", {}).get("static", {}),
        }
        full_game_summary = {
            "reused_from": str(args.reuse_static_full_game_from),
            "config": reused_metadata.get("encoder_configs", {}).get("full_game", {}),
        }

    logger.info(
        "Training temporal sidecar encoder dim=%d architecture=%s",
        args.temporal_latent_dim,
        args.temporal_architecture,
    )
    temporal_by_key, temporal_features, temporal_summary = _train_temporal(
        latent_dim=args.temporal_latent_dim,
        epochs=args.temporal_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 2,
        mask_as_input=bool(args.temporal_mask_as_input),
        zero_unobserved_input=bool(args.temporal_zero_unobserved_input),
        architecture=args.temporal_architecture,
        width_profile=args.temporal_width_profile,
    )
    temporal_latents = np.zeros((len(keys), args.temporal_latent_dim), dtype=np.float32)
    temporal_missing = 0
    for idx, key in enumerate(keys):
        latent = temporal_by_key.get(key)
        if latent is None:
            temporal_missing += 1
            continue
        temporal_latents[idx] = latent

    static_latents, full_game_latents, temporal_latents, multiview_summary = (
        align_multiview_latents(
            static_latents=static_latents,
            full_game_latents=full_game_latents,
            temporal_latents=temporal_latents,
            support=support,
            objective=args.multiview_alignment_objective,
            alignment_weight=args.multiview_alignment_weight,
            alignment_dim=args.multiview_alignment_dim,
            epochs=args.multiview_alignment_epochs,
            batch_size=args.batch_size,
            lr=args.multiview_alignment_lr,
            seed=args.seed + 3,
            device=device,
        )
    )

    elapsed_seconds = time.monotonic() - started
    source_outcome_metrics = tuple(
        name for name in full_metric_columns if name in OUTCOME_METRIC_COLUMNS
    )
    included_outcome_metrics = tuple(
        name for name in metric_columns if name in OUTCOME_METRIC_COLUMNS
    )
    excluded_outcome_metrics = (
        ()
        if args.full_game_allow_outcome_metrics
        else source_outcome_metrics
    )
    if reused_metadata is None:
        metadata = build_encoder_sidecar_metadata(
            static_features=static_features,
            full_game_features=metric_columns,
            temporal_features=temporal_features,
            split_metadata={
                "fit_split": "train",
                "source_split": "train",
            },
            encoder_configs={
                "static": static_summary["config"],
                "full_game": full_game_summary["config"],
                "temporal": temporal_summary["config"],
            },
            extra={
                "static_encoder": {"source": "deterministic champion dictionary"},
                "full_game_encoder": {
                    "latent_export": full_game_summary.get("latent_export", "autoencoder"),
                    "support_weighting": full_game_summary.get("support_weighting"),
                    "sample_weight_summary": full_game_summary.get(
                        "sample_weight_summary",
                    ),
                    "semantic_target_mode": full_game_summary.get(
                        "semantic_target_mode",
                    ),
                    "semantic_target_names": full_game_summary.get(
                        "semantic_target_names",
                    ),
                    "semantic_target_weight": full_game_summary.get(
                        "semantic_target_weight",
                    ),
                    "semantic_target_summary": full_game_summary.get(
                        "semantic_target_summary",
                    ),
                    "input_surface": args.full_game_input_surface,
                    "metric_count": len(metric_columns),
                    "allow_outcome_metrics": args.full_game_allow_outcome_metrics,
                    "included_outcome_metrics": included_outcome_metrics,
                    "excluded_outcome_metrics": excluded_outcome_metrics,
                    "identity_mode": args.full_game_identity_mode,
                },
                "export": {
                    "kind": "compact_hgnn_semantic_sidecar",
                    "device": device,
                    "rows": len(keys),
                    "temporal_missing_rows": temporal_missing,
                    "elapsed_seconds": elapsed_seconds,
                    "width_profiles": {
                        "full_game": args.full_game_width_profile,
                        "temporal": args.temporal_width_profile,
                    },
                    "full_game_input_surface": args.full_game_input_surface,
                    "full_game_identity_mode": args.full_game_identity_mode,
                    "full_game_allow_outcome_metrics": args.full_game_allow_outcome_metrics,
                    "temporal_zero_unobserved_input": args.temporal_zero_unobserved_input,
                },
                "multiview_alignment": multiview_summary,
            },
        )
    else:
        metadata = _metadata_with_reused_static_full_game(
            reused_metadata,
            base_sidecar_path=args.reuse_static_full_game_from,
            temporal_features=temporal_features,
            temporal_summary=temporal_summary,
            rows=len(keys),
            temporal_missing=temporal_missing,
            elapsed_seconds=elapsed_seconds,
            device=device,
        )
        metadata["multiview_alignment"] = multiview_summary
    out = save_encoder_sidecar(
        args.output,
        champion_id=champion_id,
        teamposition=teamposition,
        build=build,
        static_latents=static_latents,
        full_game_latents=full_game_latents,
        temporal_latents=temporal_latents,
        support=support,
        metadata=metadata,
    )
    summary = {
        "output": str(out),
        "rows": len(keys),
        "dims": {
            "static": args.static_latent_dim,
            "full_game": args.full_game_latent_dim,
            "temporal": args.temporal_latent_dim,
            "total": args.static_latent_dim + args.full_game_latent_dim + args.temporal_latent_dim,
        },
        "temporal_missing_rows": temporal_missing,
        "width_profiles": {
            "full_game": args.full_game_width_profile,
            "temporal": args.temporal_width_profile,
        },
        "full_game_latent_export": args.full_game_latent_export,
        "full_game_input_surface": args.full_game_input_surface,
        "full_game_identity_mode": args.full_game_identity_mode,
        "full_game_allow_outcome_metrics": args.full_game_allow_outcome_metrics,
        "full_game_included_outcome_metrics": included_outcome_metrics,
        "full_game_excluded_outcome_metrics": excluded_outcome_metrics,
        "multiview_alignment": multiview_summary,
        "temporal_zero_unobserved_input": args.temporal_zero_unobserved_input,
        "static": static_summary,
        "full_game": full_game_summary,
        "temporal": temporal_summary,
        "elapsed_seconds": elapsed_seconds,
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Wrote encoder sidecar: %s", out)
    if args.summary_output is not None:
        logger.info("Wrote summary: %s", args.summary_output)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--static-latent-dim", type=int, default=16)
    parser.add_argument("--full-game-latent-dim", type=int, default=64)
    parser.add_argument("--temporal-latent-dim", type=int, default=64)
    parser.add_argument(
        "--reuse-static-full-game-from",
        type=Path,
        help=(
            "Existing sidecar artifact whose static/full-game blocks, keys, "
            "support, and metadata should be reused while retraining temporal."
        ),
    )
    parser.add_argument("--static-epochs", type=int, default=500)
    parser.add_argument("--full-game-epochs", type=int, default=200)
    parser.add_argument("--temporal-epochs", type=int, default=200)
    parser.add_argument(
        "--full-game-width-profile",
        choices=SIDECAR_WIDTH_PROFILES,
        default="compact",
        help=(
            "Full-game encoder capacity profile. 'compact' preserves the "
            "current HGNN sidecar recipe; 'standalone' uses the larger "
            "full_game_encoder.py defaults."
        ),
    )
    parser.add_argument(
        "--full-game-support-weighting",
        choices=FULL_GAME_SUPPORT_WEIGHTING_MODES,
        default="none",
        help=(
            "Optional reconstruction loss weighting from identity support. "
            "'log1p' uses normalized log1p(matchups) as confidence weights."
        ),
    )
    parser.add_argument(
        "--full-game-semantic-target-mode",
        choices=FULL_GAME_SEMANTIC_TARGET_MODES,
        default="none",
        help=(
            "Optional train-only semantic auxiliary targets for the full-game "
            "latent. 'soft_v2' uses the continuous promoted audit axes; "
            "'all_v2' uses the full semantic feature schema."
        ),
    )
    parser.add_argument(
        "--full-game-semantic-target-weight",
        type=float,
        default=0.0,
        help="Loss weight for optional full-game semantic auxiliary targets.",
    )
    parser.add_argument(
        "--full-game-latent-export",
        choices=FULL_GAME_LATENT_EXPORT_MODES,
        default="autoencoder",
        help=(
            "How to populate the full-game latent block. 'autoencoder' uses the "
            "trained full-game encoder; 'semantic_targets' writes z-scored "
            "semantic target axes directly into the fixed-width latent block; "
            "'pca_whitened' writes deterministic PCA-whitened metric scores."
        ),
    )
    parser.add_argument(
        "--full-game-input-surface",
        choices=FULL_GAME_INPUT_SURFACES,
        default="full",
        help=(
            "Metric surface passed to the full-game encoder. 'full' is the "
            "current 214-column non-outcome production surface; narrower options isolate "
            "raw, derived, and context feature families."
        ),
    )
    parser.add_argument(
        "--full-game-allow-outcome-metrics",
        action="store_true",
        help=(
            "Opt in to oracle-style outcome/prior metrics such as historical win "
            "rate. Production semantic sidecars exclude these by default."
        ),
    )
    parser.add_argument(
        "--full-game-identity-mode",
        choices=FULL_GAME_IDENTITY_MODES,
        default="normal",
        help=(
            "Use normal champion/role/build embeddings or disable them to test "
            "whether metric semantics still calibrate without identity shortcuts."
        ),
    )
    parser.add_argument(
        "--multiview-alignment-objective",
        choices=MULTIVIEW_ALIGNMENT_OBJECTIVES,
        default="none",
        help=(
            "Optional post-encoder multi-view objective over static, full-game, "
            "and temporal latents. Keeps exported sidecar block widths unchanged."
        ),
    )
    parser.add_argument(
        "--multiview-alignment-weight",
        type=float,
        default=0.0,
        help="Loss weight for optional VICReg/Barlow-style multi-view alignment.",
    )
    parser.add_argument(
        "--multiview-alignment-dim",
        type=int,
        default=16,
        help="Shared prefix width used for multi-view alignment losses.",
    )
    parser.add_argument(
        "--multiview-alignment-epochs",
        type=int,
        default=300,
        help="Epoch budget for the lightweight post-encoder alignment transform.",
    )
    parser.add_argument(
        "--multiview-alignment-lr",
        type=float,
        default=5.0e-3,
        help="Learning rate for the lightweight post-encoder alignment transform.",
    )
    parser.add_argument(
        "--temporal-width-profile",
        choices=SIDECAR_WIDTH_PROFILES,
        default="compact",
        help=(
            "Temporal encoder capacity profile. 'compact' preserves the "
            "current HGNN sidecar recipe; 'standalone' uses the larger "
            "temporal_autoencoder.py defaults."
        ),
    )
    parser.add_argument(
        "--temporal-architecture",
        choices=sorted(SUPPORTED_TEMPORAL_ARCHITECTURES),
        default="flat",
        help="Temporal encoder backbone used before the latent projection.",
    )
    parser.add_argument(
        "--temporal-mask-as-input",
        action="store_true",
        help="Append the observed-bucket mask as a temporal encoder input channel.",
    )
    temporal_zero_group = parser.add_mutually_exclusive_group()
    temporal_zero_group.add_argument(
        "--temporal-zero-unobserved-input",
        dest="temporal_zero_unobserved_input",
        action="store_true",
        default=True,
        help="Zero temporal bucket inputs that were not observed before encoding.",
    )
    temporal_zero_group.add_argument(
        "--no-temporal-zero-unobserved-input",
        dest="temporal_zero_unobserved_input",
        action="store_false",
        help="Leave unobserved temporal bucket values untouched before encoding.",
    )
    return parser.parse_args(argv)


def main() -> None:
    setup_logging_config()
    build_sidecar(_parse_args())


if __name__ == "__main__":
    main()
