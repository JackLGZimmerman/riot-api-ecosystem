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
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from app.classification.embeddings.config import EmbeddingConfig, IdentityType
from app.classification.embeddings.pipeline import build_metric_matrices
from app.classification.embeddings.temporal import build_temporal_tensors
from app.classification.full_game_encoder import (
    FullGameProfileDataset,
    FullGameSemanticConfig,
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
    TemporalAEConfig,
    evaluate_temporal_autoencoder,
    extract_temporal_latents,
    train_temporal,
)
from app.core.logging.logger import setup_logging_config
from app.ml.encoder_sidecar import build_encoder_sidecar_metadata, save_encoder_sidecar

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    config = FullGameSemanticConfig(
        n_champions=int(frame["champion_id"].max()) + 1,
        n_teampositions=int(frame["teamposition_id"].max()) + 1,
        n_builds=int(frame["build_id"].max()) + 1,
        metrics_dim=len(metric_columns),
        latent_dim=latent_dim,
        metrics_embedding_dim=min(96, max(32, latent_dim)),
        metrics_hidden_dims=(192, 96),
        fusion_hidden_dims=(96,),
        decoder_hidden_dims=(128, 96),
        latent_dropout=0.05,
        latent_norm="batch",
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
    )
    dataset = FullGameProfileDataset(frame, metric_columns)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluation = evaluate_autoencoder(model, loader, device=device)
    latents = extract_full_game_latents(model, loader, device=device)
    latent_cols = [col for col in latents.columns if col.startswith("latent_")]
    return latents[latent_cols].to_numpy(dtype=np.float32, copy=True), {
        "history_last": history[-1],
        "evaluation": evaluation,
        "config": asdict(model.config),
    }


def _train_temporal(
    *,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[dict[tuple[int, str, str], np.ndarray], tuple[str, ...], dict[str, Any]]:
    tensors = build_temporal_tensors(EmbeddingConfig(split="train"), use_cache=True)
    config = TemporalAEConfig(
        metric_embed_dim=min(48, max(16, latent_dim)),
        latent_dim=latent_dim,
        hidden=512,
        dropout=0.02,
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


def build_sidecar(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    device = _resolve_device(args.device)
    logger.info("Building train-only classification matrices")
    matrices = build_metric_matrices(EmbeddingConfig(split="train"))
    baseline = matrices[IdentityType.BASELINE]
    identity_frame, _pos_idx, _build_idx = _identity_frame(baseline)
    metric_columns = tuple(str(name) for name in baseline.feature_names)
    keys = [(int(key[0]), str(key[1]), str(key[2])) for key in baseline.keys]
    champion_id = np.asarray([key[0] for key in keys], dtype=np.int32)
    teamposition = np.asarray([key[1] for key in keys])
    build = np.asarray([key[2] for key in keys])

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
    full_game_latents, full_game_summary = _train_full_game(
        identity_frame,
        metric_columns,
        latent_dim=args.full_game_latent_dim,
        epochs=args.full_game_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 1,
    )

    logger.info("Training temporal sidecar encoder dim=%d", args.temporal_latent_dim)
    temporal_by_key, temporal_features, temporal_summary = _train_temporal(
        latent_dim=args.temporal_latent_dim,
        epochs=args.temporal_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 2,
    )
    temporal_latents = np.zeros((len(keys), args.temporal_latent_dim), dtype=np.float32)
    temporal_missing = 0
    for idx, key in enumerate(keys):
        latent = temporal_by_key.get(key)
        if latent is None:
            temporal_missing += 1
            continue
        temporal_latents[idx] = latent

    support = baseline.matchups.astype(np.float32, copy=False)
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
            "export": {
                "kind": "compact_hgnn_semantic_sidecar",
                "device": device,
                "rows": len(keys),
                "temporal_missing_rows": temporal_missing,
                "elapsed_seconds": time.monotonic() - started,
            },
        },
    )
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
        "static": static_summary,
        "full_game": full_game_summary,
        "temporal": temporal_summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Wrote encoder sidecar: %s", out)
    if args.summary_output is not None:
        logger.info("Wrote summary: %s", args.summary_output)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--static-latent-dim", type=int, default=16)
    parser.add_argument("--full-game-latent-dim", type=int, default=64)
    parser.add_argument("--temporal-latent-dim", type=int, default=64)
    parser.add_argument("--static-epochs", type=int, default=80)
    parser.add_argument("--full-game-epochs", type=int, default=80)
    parser.add_argument("--temporal-epochs", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    setup_logging_config()
    build_sidecar(_parse_args())


if __name__ == "__main__":
    main()
