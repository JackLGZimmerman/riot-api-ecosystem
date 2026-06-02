"""Target/prior definitions and pipeline config.

The full-game metric catalogue (raw source groups + derived ratios) lives in
`registry.py`; the legacy names re-exported below are views over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.classification.embeddings.registry import (
    ALL_METRICS,
    DERIVED_METRIC_FUNCS,
    FINAL_SNAPSHOT_AVG_METRICS as FINAL_SNAPSHOT_AVG_METRICS,
    LARGEST_AVG_METRICS as LARGEST_AVG_METRICS,
    PER_MINUTE_METRICS as PER_MINUTE_METRICS,
    RATE_LIKE_METRICS as RATE_LIKE_METRICS,
    RATE_METRICS as RATE_METRICS,
)
from app.core.config.settings import PROJECT_ROOT

EMBEDDINGS_CACHE_DIR = (
    PROJECT_ROOT / "app" / "classification" / "data" / "embeddings" / "cache"
)

PARTICIPANT_STATS_TABLE = "game_data_filtered.participant_stats"
ITEM_VALUE_TOTALS_TABLE = "game_data_filtered.participant_item_value_totals"
ML_GAME_SPLIT_TABLE = "game_data_filtered.ml_game_split"
FINAL_PARTICIPANT_STATS_TABLE = "game_data.tl_participant_stats"

# Challenge-derived data is forbidden in classification embeddings. Do not add
# participant_challenges joins or challenge_* columns here; they are post-game
# diagnostics and are not valid identity semantics for draft-time use.


class IdentityType(str, Enum):
    BASELINE = "baseline"
    SIBLING = "sibling"
    CHAMPION_ROLE = "champion_role"
    ROLE_BUILD = "role_build"
    CHAMPION_BUILD = "champion_build"
    BUILD = "build"


LEVEL_KEY: dict[IdentityType, tuple[str, ...]] = {
    IdentityType.BASELINE: ("championid", "teamposition", "build"),
    IdentityType.SIBLING: ("championid", "teamposition", "build"),
    IdentityType.CHAMPION_ROLE: ("championid", "teamposition"),
    IdentityType.ROLE_BUILD: ("teamposition", "build_group"),
    IdentityType.CHAMPION_BUILD: ("championid", "build_group"),
    IdentityType.BUILD: ("build_group",),
}

PRIOR_LEVELS: tuple[IdentityType, ...] = (
    IdentityType.SIBLING,
    IdentityType.CHAMPION_ROLE,
    IdentityType.CHAMPION_BUILD,
    IdentityType.ROLE_BUILD,
    IdentityType.BUILD,
)

# Prior strength caps. Two dicts because the two metric families have different
# effective-N denominators and therefore different natural scales:
#   * rate metrics use `matchups` (~10^1-10^3) as effective N
#   * per-minute metrics use `sum_w_timeplayed` seconds (~10^3-10^5) as effective N
# Per-minute caps are rate caps scaled by the typical sum_w_timeplayed per matchup.
PRIOR_PER_MINUTE_SCALE: float = 200.0

DEFAULT_PRIOR_RATE_STRENGTHS: dict[IdentityType, float] = {
    IdentityType.SIBLING: 20.0,
    IdentityType.CHAMPION_BUILD: 12.0,
    IdentityType.CHAMPION_ROLE: 7.0,
    IdentityType.ROLE_BUILD: 7.0,
    IdentityType.BUILD: 4.0,
}

DEFAULT_PRIOR_PER_MINUTE_STRENGTHS: dict[IdentityType, float] = {
    k: v * PRIOR_PER_MINUTE_SCALE for k, v in DEFAULT_PRIOR_RATE_STRENGTHS.items()
}


def _assert_no_challenge_features(
    feature_set: tuple[str, ...],
    *,
    owner: str,
) -> None:
    leaked = tuple(name for name in feature_set if "challenge" in name.lower())
    if leaked:
        preview = ", ".join(leaked[:5])
        raise AssertionError(f"{owner} cannot use challenge data: {preview}")


def full_game_derived_metric_names() -> tuple[str, ...]:
    """Derived semantic metrics computed from full-game source columns."""
    return tuple(DERIVED_METRIC_FUNCS)


def raw_metric_names() -> tuple[str, ...]:
    """Raw full-game source metrics used by classification profiles."""
    _assert_no_challenge_features(ALL_METRICS, owner="raw_metric_names")
    return ALL_METRICS


def raw_and_derived_metric_names() -> tuple[str, ...]:
    """All preserved full-game profile metrics: raw sources plus derived ratios."""
    feature_set = (*ALL_METRICS, *full_game_derived_metric_names())
    _assert_no_challenge_features(feature_set, owner="raw_and_derived_metric_names")
    return feature_set


@dataclass(frozen=True)
class EmbeddingConfig:
    cache_dir: Path = EMBEDDINGS_CACHE_DIR
    split: str = "train"

    prior_rate_strengths: dict[IdentityType, float] = field(
        default_factory=DEFAULT_PRIOR_RATE_STRENGTHS.copy
    )
    prior_per_minute_strengths: dict[IdentityType, float] = field(
        default_factory=DEFAULT_PRIOR_PER_MINUTE_STRENGTHS.copy
    )
    # Dynamic low-sample weighting (shared with app/ml via
    # app.core.utils.smoothing): each prior level's weight is multiplied by
    # sqrt(1 + amplification_threshold/max(n, 1)), so under-sampled levels are
    # pulled harder toward their prior.
    amplification_threshold: float = 50.0
    # Smoothing strategy:
    #   "additive" — pool every prior level into one weighted mixture (legacy).
    #   "cascade"  — shrink toward only the highest-priority prior level whose
    #                own sample size clears `prior_confidence_matchups`; broader
    #                levels are used only as fallback when no specific level is
    #                confident enough. Prevents broad priors from contaminating
    #                well-sampled specific priors.
    smoothing_mode: str = "cascade"
    prior_confidence_matchups: float = 50.0
    # When cascade selects a single level, give it the same *total* prior weight
    # the additive mixture would have applied across all levels. This isolates
    # the effect of the prior *value* (single vs pooled) from the effect of the
    # shrinkage *magnitude*, which otherwise drops sharply in cascade mode.
    cascade_match_weight: bool = True
    # Roles that must not borrow from cross-role prior levels. CHAMPION_BUILD
    # aggregates a champion across all roles; BUILD aggregates all roles into one
    # group. Both contaminate utility/jungle embeddings with lane-champion stats.
    # Lane roles (TOP/MIDDLE/BOTTOM) are allowed to share because their mechanics
    # are similar enough that cross-champion borrowing is informative.
    isolated_roles: frozenset[str] = frozenset({"UTILITY", "JUNGLE"})
    isolated_role_excluded_levels: tuple[IdentityType, ...] = (
        IdentityType.CHAMPION_BUILD,
        IdentityType.BUILD,
    )
    feature_set: tuple[str, ...] = field(default_factory=raw_and_derived_metric_names)
    matrix_clip_value: float | None = 8.0
