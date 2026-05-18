from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.core.config.settings import PROJECT_ROOT

ML_DATA_DIR = PROJECT_ROOT / "app" / "ml" / "data"
CACHE_DIR = ML_DATA_DIR / "cache"
# Preserved sweep runs; live training uses ML_DATA_DIR directly.
CHECKPOINT_DIR = ML_DATA_DIR / "checkpoints"

PARTICIPANT_TABLE = "game_data_filtered.participant_stats"
ITEM_VALUE_TABLE = "game_data_filtered.participant_item_value_totals"
SPLIT_TABLE = "game_data_filtered.ml_game_split"
PLAYER_PIVOT_TABLE = "game_data_filtered.ml_game_player_pivot"

POSITIONS: tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")

SIDE_BLUE = 0
SIDE_RED = 1
N_SIDES = 2


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
    pooling: Literal[
        "team_mean",
        "team_attention",
    ] = "team_mean"
    head_hidden: int = 256


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16_384
    epochs: int = 500
    lr: float = 2e-4
    weight_decay: float = 5e-3
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    compile_mode: Literal["none", "default", "reduce-overhead", "max-autotune"] = (
        "reduce-overhead"
    )
    target_min: float = 0.05
    target_max: float = 0.95
    warmup_steps: int = 125
    # Smooth heavy-tail decay after warmup; see README "Learning-Rate Schedule".
    lr_center_epoch: int = 10
    lr_sharpness: float = 8.0
    lr_tail_strength: float = 0.5
    lr_eta_min_ratio: float = 0.01
    grad_clip: float = 0.0
    log_interval: int = 10
    # Early-stop on val_loss after N epochs without improvement; 0 disables.
    early_stop_patience: int = 0
    # When False, skip the final reload-best + test eval block (sweeps).
    run_final_test: bool = True
    # Gates only the heavy sampled-attention + prediction-bucket diagnostics;
    # core metrics log every epoch regardless.
    attention_diagnostics_interval: int = 10
    attention_diagnostics_batch_size: int = 256
    attention_diagnostics_eval_samples: int = 1024
    # Graduated prediction-band table (5%/1%/0.1% bins) emitted on heavy
    # diagnostic epochs + final test as a `prediction_bands` event.
    prediction_bands_enabled: bool = True
    # Held-in train subset evaluated every epoch through the validation path so
    # train-vs-val gaps are directly comparable. 0 disables (sweeps).
    train_monitor_samples: int = 50_000
    checkpoint_dir: Path = ML_DATA_DIR
    metrics_dir: Path = ML_DATA_DIR
    metrics_file: str = "metrics.jsonl"
    latest_metrics_file: str = "metrics_latest.json"
    tensorboard_dir: str | None = "tensorboard"
    # When set, drops the default timestamp suffix so sweeps can pin stable
    # paths like `runs/pooling/{pooling}/seed_{seed}`.
    tensorboard_run_name: str | None = None
    # When True, additionally mirrors every JSONL scalar under raw/<event>/<field>.
    tensorboard_raw_mirror: bool = False
    device: str = "cuda"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    seed: int = 42
