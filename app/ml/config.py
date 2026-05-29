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
    model_path: Path = ML_DATA_DIR / "linear_winrate_model.npz"
    metrics_path: Path = ML_DATA_DIR / "metrics_latest.json"
    # L2 penalty on feature weights (not the intercept). The 45 interaction
    # features overfit badly unregularised (train ~0.70 / val ~0.53); l2~0.01
    # recovers val/test to ~0.57. Validated on the full 1.95M-game cache.
    l2: float = 0.01
