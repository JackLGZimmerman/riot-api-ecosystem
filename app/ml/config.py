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
    # Dynamic low-sample weighting (shared with app/classification via
    # app.core.utils.smoothing). The prior strength applied to each
    # identity/interaction is multiplied by sqrt(1 + amplification_threshold/max(n, 1)),
    # so under-sampled pairs shrink harder toward their composite prior while
    # well-sampled pairs keep `smoothing_prior_strength`. 0.0 disables
    # amplification (flat strength). See app/ml docs for the low-sample recovery
    # evaluation behind this value.
    amplification_threshold: float = 50.0
    # "cascade" removes broad-prior shrinkage once an identity/interaction's
    # own support clears `prior_confidence_matchups`; low-support rows still use
    # the configured dynamic Bayesian fallback. Use "additive" for the legacy
    # always-smooth behaviour.
    smoothing_mode: str = "cascade"
    prior_confidence_matchups: float = 50.0
    # Strength used for confidence = n/(n+s) in object features and pooling.
    # Was 150 under flat smoothing (a heavy gate to mute leaky low-support pairs).
    # With amplified smoothing (`amplification_threshold`) those low-support
    # values are trustworthy, so the gate can open up: sweeping s over 5–150
    # peaks on a flat 28–35 plateau (val/test AUC); 30 chosen. Below ~20 tail
    # calibration degrades and s=0 is degenerate (0/0 for zero-support pairs).
    confidence_gate_strength: float = 30.0
    # Shrink under-sampled 1v1/2vx pairs toward a composite of their two sides'
    # solo priors instead of a flat 0.5 (see app/ml docs). Improves interaction
    # ranking (AUC) once the model is regularised.
    interaction_per_side_fallback: bool = True
    # Leave-one-out the train split's own outcome from its solo/1v1/2vx priors
    # before smoothing, so the joint-minus-expected delta stops leaking the label.
    # See documentation/README.md.
    interaction_loo: bool = True


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
