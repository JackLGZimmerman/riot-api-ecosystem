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
DEFAULT_TRAIN_BATCH_CAP = 40960

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
    val_fraction: float = 0.1
    test_fraction: float = 0.1
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
    # Optional frozen three-encoder sidecar artifact. When set during cache
    # build, per-slot static/full-game/temporal latent arrays are materialised
    # into the HGNN cache. Existing caches without these arrays still load.
    encoder_sidecar_path: Path | None = DEFAULT_ENCODER_SIDECAR_PATH


@dataclass(frozen=True)
class TrainConfig:
    model_path: Path = ML_DATA_DIR / "hgnn_production_model.pt"
    metrics_path: Path = ML_DATA_DIR / "metrics_latest.json"
    audit_prediction_cache_path: Path | None = None
    warm_start_model_path: Path | None = None
    # When warm-starting a larger candidate model from production, optionally
    # keep loaded checkpoint parameters fixed and train only newly introduced
    # parameters that were missing from the checkpoint.
    freeze_warm_start_loaded_parameters: bool = False
    batch_size: int = 32768
    # Effective training batch safety cap for the team-swapped HGNN loop.
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
    # Where to cache the raw train/val/test tensors before minibatch indexing.
    # "model" preserves the historical behavior; "cpu" keeps the large raw
    # cache off GPU and moves only each indexed minibatch to the model device.
    raw_tensor_cache_device: str = "model"
    seed: int = 0
    max_grad_norm: float | None = 1.0
    # Supported values are defined in app.ml.train.CHECKPOINT_METRICS.
    # Production promotion tracks the raw held-out accuracy gate directly.
    checkpoint_metric: str = "val_accuracy"
    # Minimum checkpoint-score improvement required to reset early stopping.
    # The raw-accuracy confirmation path keeps exact best-epoch selection.
    checkpoint_min_delta: float = 0.0
    # Throughput sweeps only need the per-epoch validation/timing row. When set,
    # skip the expensive final train/val/test prediction pass.
    skip_final_evaluation: bool = False
    # Experimental, training-only pairwise ranking objective. When >0, each
    # batch samples positive/negative logit pairs and adds a soft AUC surrogate
    # to BCE. This is opt-in research behaviour and does not affect inference.
    auc_ranking_loss_weight: float = 0.0
    auc_ranking_loss_pairs: int = 4096
    # Optional semantic context-bin calibration objective. It matches the mean
    # predicted side win rate to the empirical side win rate inside the shared
    # HGNN context audit bins, giving rare semantic tails direct training signal.
    semantic_context_calibration_loss_weight: float = 0.0
    semantic_context_calibration_min_count: int = 8
    semantic_context_calibration_tail_weight: float = 2.0
    # Reporting keeps the full expanded group audit. Train-time calibration can
    # optionally use a smaller core surface to avoid chasing duplicated/noisy
    # group tails.
    semantic_context_calibration_group_surface: str = "full"
    semantic_context_calibration_bin_weighting: str = "uniform"
    # Calibration target family. "champion_raw" matches each champion-specific audit
    # bin's raw train win rate (high variance: median bin n~500, noise floor ~10.5
    # pp^2, overfits when up-weighted). "*_eb" targets use empirical-Bayes-shrunk
    # train rates. "group_eb" uses deterministic build/role group bins;
    # "context_eb" uses champion/context audit bins; "group_context_eb" combines
    # both families for a residual-calibration style objective.
    semantic_context_calibration_target: str = "champion_raw"
    # "absolute" matches current predictions directly to train EB/raw bin
    # targets. "residual" instead teaches the semantic MoE to reproduce a
    # bounded train-only logit correction relative to the warm-start model,
    # mirroring the post-hoc semantic residual calibrator.
    semantic_context_calibration_objective: str = "absolute"
    semantic_context_calibration_group_residual_shrink_strength: float = 100000.0
    semantic_context_calibration_group_residual_clip: float = 0.02
    semantic_context_calibration_group_residual_scale: float = 1.0
    semantic_context_calibration_context_residual_shrink_strength: float = 50000.0
    semantic_context_calibration_context_residual_clip: float = 0.02
    semantic_context_calibration_context_residual_scale: float = 1.0
    # Residual-only alternative to squared error. The uncertainty-aware Huber
    # variant ignores residual gaps inside the train EB target's logit-scale
    # uncertainty band, then applies a Huber penalty to the excess.
    semantic_context_calibration_residual_loss: str = "mse"
    semantic_context_calibration_uncertainty_band_scale: float = 1.0
    semantic_context_calibration_uncertainty_huber_delta: float = 0.01
    # Optional group-spec holdout for calibration-target audits. "even_odd"
    # trains on one parity of group specs and reports trained vs held-out bins.
    semantic_context_calibration_holdout_mode: str = "none"
    semantic_context_calibration_holdout_fold: int = 0
    # Validation/reporting lens for context-gap checkpoint metrics. This keeps
    # low-support champion slices visible in the raw audit while letting model
    # selection target bins with enough support to make a 2-3pp max meaningful.
    semantic_context_metric_min_count: int = 2048
    # Number of initial epochs that log BCE/context-loss gradient alignment on
    # semantic MoE and semantic group-relationship parameters.
    semantic_context_calibration_gradient_diagnostics_epochs: int = 5
