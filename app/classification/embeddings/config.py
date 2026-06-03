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

# Rate metrics use `matchups` as effective N; per-minute use `sum_w_timeplayed`
# (~200x larger), so per-minute caps are the rate caps scaled by this factor.
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
    # Under-sampled levels pulled harder toward their prior (weight *=
    # sqrt(1 + amplification_threshold/max(n, 1))). Shared with app/ml.
    amplification_threshold: float = 50.0
    # "additive" pools all prior levels; "cascade" shrinks toward only the
    # highest-priority level clearing `prior_confidence_matchups`, so broad priors
    # don't contaminate well-sampled specific ones.
    smoothing_mode: str = "cascade"
    prior_confidence_matchups: float = 50.0
    # In cascade, give the selected level the total weight the additive mixture
    # would apply, isolating prior value (single vs pooled) from shrinkage magnitude.
    cascade_match_weight: bool = True
    # UTILITY/JUNGLE must not borrow from cross-role levels (CHAMPION_BUILD, BUILD),
    # which would contaminate them with lane-champion stats; lanes may share.
    isolated_roles: frozenset[str] = frozenset({"UTILITY", "JUNGLE"})
    isolated_role_excluded_levels: tuple[IdentityType, ...] = (
        IdentityType.CHAMPION_BUILD,
        IdentityType.BUILD,
    )
    feature_set: tuple[str, ...] = field(default_factory=raw_and_derived_metric_names)
    matrix_clip_value: float | None = 8.0
    # Opt-in feature blocks, off by default to keep the 147-feature baseline byte-stable.
    include_static_champion: bool = False  # +47 static champion base stats
    include_context_features: bool = False  # +55 team-share / matchup features
