from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.core.config.settings import PROJECT_ROOT

ML_DATA_DIR = PROJECT_ROOT / "app" / "ml" / "data"
CACHE_DIR = ML_DATA_DIR / "cache"
# Base for preserved hyper-parameter sweep runs; live training writes to
# ML_DATA_DIR directly. Sweeps set TrainConfig.checkpoint_dir to a run subdir.
CHECKPOINT_DIR = ML_DATA_DIR / "checkpoints"

PARTICIPANT_TABLE = "game_data_filtered.participant_stats"
ITEM_VALUE_TABLE = "game_data_filtered.participant_item_value_totals"
SPLIT_TABLE = "game_data_filtered.ml_game_split"
PLAYER_PIVOT_TABLE = "game_data_filtered.ml_game_player_pivot"

POSITIONS: tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")

SIDE_BLUE = 1
SIDE_RED = 2
N_SIDES = 4

# Reserve index 0 for unknown / padding across all categorical embeddings.
UNK_INDEX = 0


@dataclass(frozen=True)
class DatasetConfig:
    cache_dir: Path = CACHE_DIR
    max_games: int | None = None
    build_chunk_games: int = 150_000
    val_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.15
    attention_dropout: float = 0.10
    head_dropout: float = 0.0
    pooling: Literal["cls", "mean", "attention", "concat_cls_mean", "gated"] = "gated"
    head_hidden: int = 256


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16_384
    gradient_accumulation_steps: int = 1
    epochs: int = 100
    lr: float = 5e-5
    weight_decay: float = 5e-3
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    compile_mode: Literal["none", "default", "reduce-overhead", "max-autotune"] = (
        "reduce-overhead"
    )
    target_min: float = 0.15
    target_max: float = 0.85
    warmup_steps: int = 125
    grad_clip: float = 0.0
    log_interval: int = 40
    # Core metrics (loss/accuracy/AUC/Brier/ECE/central-band, train+val) log every
    # epoch. The interval below only gates the heavy sampled attention + full
    # prediction-bucket diagnostics.
    attention_diagnostics_interval: int = 40
    attention_diagnostics_batch_size: int = 256
    attention_diagnostics_eval_samples: int = 1024
    # Held-in train subset evaluated every epoch through the validation path so
    # train-vs-val gaps are directly comparable. 0 disables it (sweeps).
    train_monitor_samples: int = 50_000
    checkpoint_dir: Path = ML_DATA_DIR
    metrics_dir: Path = ML_DATA_DIR
    metrics_file: str = "metrics.jsonl"
    latest_metrics_file: str = "metrics_latest.json"
    tensorboard_dir: str | None = "tensorboard"
    # When false, TensorBoard only receives the curated metric families. Enable
    # to additionally mirror every JSONL scalar under raw/<event>/<field>.
    tensorboard_raw_mirror: bool = False
    device: str = "cuda"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    seed: int = 42
