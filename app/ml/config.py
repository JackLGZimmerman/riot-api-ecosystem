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
SYNERGY_1VX_TABLE = "game_data_filtered.synergy_1vx"
CHAMPION_ID_NAME_DICT = "game_data.championid_name_map_dict"
ITEM_VALUE_MAP_DICT = "game_data.item_value_map_dict"

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
    use_moe: bool = False
    n_experts: int = 8
    expert_hidden: int = 128
    router_hidden: int = 64
    moe_top_k: int = 2
    router_temperature: float = 1.0
    moe_dropout: float = 0.10
    moe_aux_loss_coef: float = 0.01
    # Steps to use dense routing (k=n_experts) before switching to top_k.
    moe_warmup_steps: int = 81
    # Shazeer-style Gaussian noise scale on routing logits in train mode only.
    # 0 disables; default ~1/sqrt(n_experts) for n_experts=8.
    moe_router_noise: float = 0.354


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
    early_stop_patience: int = 10
    # When False, skip the final reload-best + test eval block (sweeps).
    run_final_test: bool = True
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
