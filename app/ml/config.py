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

PLAYER_PIVOT_TABLE = "game_data_filtered.ml_game_player_pivot"
SOLO_PRIOR_TABLE = "game_data_filtered.synergy_1vx"
SOLO_PRIOR_DICT = "game_data_filtered.synergy_1vx_dict"
MATCHUP_1V1_DICT = "game_data_filtered.matchup_1v1_dict"
SYNERGY_2VX_DICT = "game_data_filtered.synergy_2vx_dict"


# Backoff levels for nested empirical-Bayes pooling of the interaction priors.
# 1v1 keeps the existing build -> no-build -> champion-pair -> composite floor
# ladder. 2vx uses build -> build-sibling group -> no-build -> neutral floor.
MATCHUP_1V1_NOBUILD_DICT = "game_data_filtered.matchup_1v1_nobuild_dict"
MATCHUP_1V1_CHAMP_DICT = "game_data_filtered.matchup_1v1_champ_dict"
SYNERGY_2VX_BUILD_GROUP_DICT = "game_data_filtered.synergy_2vx_build_group_dict"
SYNERGY_2VX_NOBUILD_DICT = "game_data_filtered.synergy_2vx_nobuild_dict"
SYNERGY_2VX_CHAMP_DICT = "game_data_filtered.synergy_2vx_champ_dict"
# Source tables (finest -> coarsest) for the empirical-Bayes per-level strength
# moments, with the win-rate column each table exposes.
MATCHUP_1V1_LEVEL_TABLES: tuple[tuple[str, str], ...] = (
    ("game_data_filtered.matchup_1v1", "left_win_rate"),
    ("game_data_filtered.matchup_1v1_nobuild", "blue_win_rate"),
    ("game_data_filtered.matchup_1v1_champ", "blue_win_rate"),
)
SYNERGY_2VX_LEVEL_TABLES: tuple[tuple[str, str], ...] = (
    ("game_data_filtered.synergy_2vx", "win_rate"),
    ("game_data_filtered.synergy_2vx_build_group", "win_rate"),
    ("game_data_filtered.synergy_2vx_nobuild", "win_rate"),
)

@dataclass(frozen=True)
class DatasetConfig:
    cache_dir: Path = CACHE_DIR
    max_games: int | None = None
    player_pivot_table: str = PLAYER_PIVOT_TABLE
    solo_prior_table: str = SOLO_PRIOR_TABLE
    solo_prior_dict: str = SOLO_PRIOR_DICT
    matchup_1v1_dict: str = MATCHUP_1V1_DICT
    synergy_2vx_dict: str = SYNERGY_2VX_DICT
    matchup_1v1_nobuild_dict: str = MATCHUP_1V1_NOBUILD_DICT
    matchup_1v1_champ_dict: str = MATCHUP_1V1_CHAMP_DICT
    synergy_2vx_build_group_dict: str = SYNERGY_2VX_BUILD_GROUP_DICT
    synergy_2vx_nobuild_dict: str = SYNERGY_2VX_NOBUILD_DICT
    synergy_2vx_champ_dict: str = SYNERGY_2VX_CHAMP_DICT
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
    # Strength used for confidence = n/(n+s) in object features and pooling.
    confidence_gate_strength: float = 30.0
    # Shrink under-sampled 1v1/2vx pairs toward a composite of their two sides'
    # solo priors instead of a flat 0.5 (see app/ml docs). Improves interaction
    # ranking (AUC) once the model is regularised. This composite is the terminal
    # floor of the nested-pooling ladder below.
    interaction_per_side_fallback: bool = True
    # Nested empirical-Bayes pooling of the 1v1/2vx priors: shrink the
    # build-conditioned cell toward its no-build pair, then its champion-only
    # pair, then the per-side composite floor. Per-level Beta strengths are
    # estimated by method-of-moments from each level's table (see
    # app.core.utils.smoothing.eb_strength). Disable to fall back to the legacy
    # single-level composite smoothing of the build-conditioned cell only.
    interaction_nested_pooling: bool = True
    # Leave-one-out the train split's own outcome from its solo/1v1/2vx priors
    # before smoothing, so the joint-minus-expected delta stops leaking the label.
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
    encoder_sidecar_path: Path | None = None


@dataclass(frozen=True)
class TrainConfig:
    model_path: Path = ML_DATA_DIR / "structured_winrate_model.pt"
    metrics_path: Path = ML_DATA_DIR / "metrics_latest.json"
    batch_size: int = 32768
    max_epochs: int = 40
    patience: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    device: str = "auto"
    seed: int = 0
    max_grad_norm: float | None = 1.0
    # Supported values are defined in app.ml.train.CHECKPOINT_METRICS.
    # The default preserves the production threshold-tuned checkpoint path.
    checkpoint_metric: str = "val_threshold_accuracy"
    # Minimum checkpoint-score improvement required to reset early stopping.
    # For the default threshold-accuracy metric, 5e-4 means tiny validation
    # wiggles no longer keep training alive when we are looking for material
    # movement.
    checkpoint_min_delta: float = 5e-4
    # Experimental, training-only pairwise ranking objective. When >0, each
    # batch samples positive/negative logit pairs and adds a soft AUC surrogate
    # to BCE. This is opt-in research behaviour and does not affect inference.
    auc_ranking_loss_weight: float = 0.0
    auc_ranking_loss_pairs: int = 4096
