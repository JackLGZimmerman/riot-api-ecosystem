from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config.settings import PROJECT_ROOT
from app.core.utils.common import POSITIONS as POSITIONS
from app.core.utils.smoothing import (
    BUILD_GROUPS as BUILD_GROUPS,
    BUILD_TO_GROUP as BUILD_TO_GROUP,
    build_group_for as build_group_for,
)

ML_DATA_DIR = PROJECT_ROOT / "app" / "ml" / "data"
CACHE_DIR = ML_DATA_DIR / "cache"
DEFAULT_ENCODER_SIDECAR_PATH = (
    ML_DATA_DIR / "semantic_identity_sidecar_compact.npz"
)
DEFAULT_TRAIN_BATCH_CAP = 16384
DEFAULT_PRODUCTION_MODEL_PATH = ML_DATA_DIR / "hgnn_production_model.pt"
DEFAULT_PRODUCTION_METRICS_PATH = ML_DATA_DIR / "metrics_latest.json"

PLAYER_PIVOT_TABLE = "game_data_filtered.ml_game_player_pivot"
SOLO_PRIOR_TABLE = "game_data_filtered.synergy_1vx"
SOLO_PRIOR_DICT = "game_data_filtered.synergy_1vx_dict"


@dataclass(frozen=True)
class DatasetConfig:
    cache_dir: Path = CACHE_DIR
    max_games: int | None = None
    player_pivot_table: str = PLAYER_PIVOT_TABLE
    solo_prior_table: str = SOLO_PRIOR_TABLE
    solo_prior_dict: str = SOLO_PRIOR_DICT
    smoothing_prior_mean: float = 0.5
    smoothing_prior_strength: float = 20.0
    # Dynamic low-sample weighting. The prior strength applied to each
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
    # Strength used for confidence = n/(n+s) in object features.
    confidence_gate_strength: float = 30.0
    # Leave-one-out the train split's own outcome from its solo prior before
    # smoothing, so the joint-minus-expected delta stops leaking the label.
    # See documentation/README.md.
    interaction_loo: bool = True
    # Draft-time caches must not use final item-derived build labels. When
    # false, the cache builder rewrites all prior lookup keys to this constant
    # build label and requires matching no-build aggregate priors to exist.
    use_final_build_labels: bool = True
    draft_unknown_build_label: str = "unknown"
    # Optional frozen three-encoder sidecar artifact. Current caches record its
    # metadata and gather latents from the compact artifact at tensor-build time.
    encoder_sidecar_path: Path | None = DEFAULT_ENCODER_SIDECAR_PATH


@dataclass(frozen=True)
class TrainConfig:
    model_path: Path = DEFAULT_PRODUCTION_MODEL_PATH
    metrics_path: Path = DEFAULT_PRODUCTION_METRICS_PATH
    batch_size: int = DEFAULT_TRAIN_BATCH_CAP
    # Effective training batch cap for the current production architecture.
    # Retune with epoch samples/s whenever parameter count changes.
    # Set to 0 or None to disable for explicit throughput/allocator sweeps.
    train_batch_cap: int | None = DEFAULT_TRAIN_BATCH_CAP
    # Optional per-epoch row cap for candidate screens. Production defaults to
    # full train epochs; sweep runners can set this to compare many candidates
    # before a final full-data promotion run.
    train_epoch_max_games: int | None = None
    max_epochs: int = 40
    patience: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    device: str = "auto"
    # Where to cache raw split tensors before minibatch indexing. "cpu" keeps
    # the large cache off GPU and moves only each indexed minibatch to the model
    # device; "model" is kept for explicit throughput sweeps.
    raw_tensor_cache_device: str = "cpu"
    seed: int = 0
    max_grad_norm: float | None = 1.0
    # Production artifact paths are the load/serve defaults, not routine train
    # outputs. Promotion runs must opt in before overwriting them.
    allow_production_artifact_overwrite: bool = False
