from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config.settings import PROJECT_ROOT

ML_DATA_DIR = PROJECT_ROOT / "app" / "ml" / "data"
CACHE_DIR = ML_DATA_DIR / "cache"

PLAYER_PIVOT_TABLE = "game_data_filtered.ml_game_player_pivot"

POSITIONS: tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


@dataclass(frozen=True)
class DatasetConfig:
    cache_dir: Path = CACHE_DIR
    max_games: int | None = None
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    smoothing_prior_mean: float = 0.5
    smoothing_prior_strength: float = 20.0
    # Shrink under-sampled 1v1/2vx pairs toward a composite of their two sides'
    # solo priors instead of a flat 0.5 (see app/ml docs). Improves interaction
    # ranking (AUC) once the model is regularised.
    interaction_per_side_fallback: bool = True


@dataclass(frozen=True)
class TrainConfig:
    model_path: Path = ML_DATA_DIR / "structured_winrate_model.pt"
    metrics_path: Path = ML_DATA_DIR / "metrics_latest.json"
    # Large batches act as implicit regularization: they slow the first-epoch
    # fit of the residual in-sample interaction leakage, so the model holds the
    # honest base ceiling instead of collapsing. See documentation/README.md.
    batch_size: int = 32768
    max_epochs: int = 40
    patience: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    delta_baseline_mode: str = "logit"
    device: str = "auto"
