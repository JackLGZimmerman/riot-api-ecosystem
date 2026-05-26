"""Target/prior definitions and pipeline config."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from app.core.config.settings import PROJECT_ROOT

EMBEDDINGS_CACHE_DIR = (
    PROJECT_ROOT / "app" / "classification" / "data" / "embeddings" / "cache"
)
GROUPING_REPORT_PATH = EMBEDDINGS_CACHE_DIR.parent / "groupings_report.html"

SOURCE_TABLE = "game_data_filtered.synergy_1vx_temporal"

# Phases ordered earliest -> latest for temporal sequencing.
PHASES: tuple[str, ...] = ("early_mid", "mid", "mid_late", "late")
PHASE_INDEX: dict[str, int] = {p: i for i, p in enumerate(PHASES)}

BUILD_LABELS: tuple[str, ...] = (
    "attack_damage",
    "ability_power",
    "lethality",
    "on_hit",
    "crit",
    "utility_enchanter",
    "utility_protection",
    "ar_tank",
    "mr_tank",
    "ad_off_tank",
    "ap_off_tank",
)

BUILD_GROUPS: dict[str, tuple[str, ...]] = {
    "ap": ("ability_power", "ap_off_tank"),
    "ad": ("attack_damage", "ad_off_tank"),
    "tank": ("ar_tank", "mr_tank"),
    "utility": ("utility_enchanter", "utility_protection"),
}
SIBLING_BUILD_PAIRS: tuple[tuple[str, str], ...] = (
    ("ability_power", "ap_off_tank"),
    ("attack_damage", "ad_off_tank"),
    ("ar_tank", "mr_tank"),
    ("utility_enchanter", "utility_protection"),
)
SIBLING_BUILD_BY_LABEL: dict[str, str] = {}
for _left, _right in SIBLING_BUILD_PAIRS:
    SIBLING_BUILD_BY_LABEL.update({_left: _right, _right: _left})


def _sql_strings(values: Iterable[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def build_group_sql(column: str = "build", alias: str | None = "build_group") -> str:
    clauses = ", ".join(
        f"{column} IN ({_sql_strings(labels)}), '{group}'"
        for group, labels in BUILD_GROUPS.items()
    )
    expr = f"multiIf({clauses}, {column})"
    return f"{expr} AS {alias}" if alias else expr


def sibling_build_sql(column: str = "build") -> str:
    clauses = ", ".join(
        f"{column} = '{source}', '{target}'"
        for source, target in SIBLING_BUILD_BY_LABEL.items()
    )
    return f"multiIf({clauses}, '')"


class IdentityType(str, Enum):
    BASELINE = "baseline"
    SIBLING = "sibling"
    CHAMPION_ROLE = "champion_role"
    ROLE_BUILD = "role_build"
    CHAMPION_BUILD = "champion_build"
    BUILD = "build"


PRIOR_TABLE: dict[IdentityType, str] = {
    IdentityType.SIBLING: "game_data_filtered.synergy_1vx_temporal_prior_sibling",
    IdentityType.CHAMPION_ROLE: (
        "game_data_filtered.synergy_1vx_temporal_prior_champion_role"
    ),
    IdentityType.ROLE_BUILD: "game_data_filtered.synergy_1vx_temporal_prior_role_build",
    IdentityType.CHAMPION_BUILD: (
        "game_data_filtered.synergy_1vx_temporal_prior_champion_build"
    ),
    IdentityType.BUILD: "game_data_filtered.synergy_1vx_temporal_prior_build",
}


DEFAULT_GROUP_MIN_MATCHUPS: dict[IdentityType, float] = {
    IdentityType.BASELINE: 0.0,
    IdentityType.BUILD: 0.0,
    IdentityType.ROLE_BUILD: 100.0,
    IdentityType.CHAMPION_ROLE: 10.0,
    IdentityType.CHAMPION_BUILD: 100.0,
    IdentityType.SIBLING: 100.0,
}


# Key columns identifying a unique identity instance at each level. Phase is a
# temporal axis and is appended on top of these keys for per-row lookups.
LEVEL_KEY: dict[IdentityType, tuple[str, ...]] = {
    IdentityType.BASELINE: ("championid", "teamposition", "build"),
    IdentityType.SIBLING: ("championid", "teamposition", "build"),
    IdentityType.CHAMPION_ROLE: ("championid", "teamposition"),
    IdentityType.ROLE_BUILD: ("teamposition", "build_group"),
    IdentityType.CHAMPION_BUILD: ("championid", "build_group"),
    IdentityType.BUILD: ("build_group",),
}

# Prior lookup and strength order follows the 9000 -> 9040 table sequence:
# most contextually relevant first, broadest fallback last.
PRIOR_LEVELS: tuple[IdentityType, ...] = (
    IdentityType.SIBLING,
    IdentityType.CHAMPION_ROLE,
    IdentityType.ROLE_BUILD,
    IdentityType.CHAMPION_BUILD,
    IdentityType.BUILD,
)

EMBEDDING_LEVELS: tuple[IdentityType, ...] = (IdentityType.BASELINE,)

# Role+build archetype priors dominate so embeddings cluster across champions
# along the role/build axis. Sibling pulls (champion, role, build) toward its
# sibling build; champion_role/champion_build are kept low so single-champion
# rows don't collapse into within-champion micro-clusters.
DEFAULT_PRIOR_RATE_STRENGTHS: dict[IdentityType, float] = {
    IdentityType.SIBLING: 12.0,
    IdentityType.CHAMPION_ROLE: 4.0,
    IdentityType.CHAMPION_BUILD: 4.0,
    IdentityType.ROLE_BUILD: 30.0,
    IdentityType.BUILD: 12.0,
}

DEFAULT_PRIOR_PER_MINUTE_STRENGTHS: dict[IdentityType, float] = {
    IdentityType.SIBLING: 2_400.0,
    IdentityType.CHAMPION_ROLE: 800.0,
    IdentityType.CHAMPION_BUILD: 800.0,
    IdentityType.ROLE_BUILD: 6_000.0,
    IdentityType.BUILD: 2_400.0,
}

# Raw metrics loaded and smoothed. `DEFAULT_RAW_FEATURE_SET` keeps the curated
# archetype-defining subset, while `DEFAULT_EMBEDDING_FEATURE_SET` adds the
# promoted ratio-derived features. Extra raw columns above the curated set are
# loaded so the experiment harness can mix in derived metrics that need them
# (e.g. damage-type shares, first-blood/tower participation, kda, epic_kills).
#
# Permanently excluded from source: damage-taken type splits, structure-kill
# detail (turret/inhibitor takedowns/slots), jungle CS splits, ward placement
# detail (collapsed into visionscore), triplekills/killingsprees, and
# damagedealtto{buildings,turrets,epicmonsters}.

# Per-game event rates in [0, 1]; smoothed with matchups as effective N.
RATE_METRICS: tuple[str, ...] = (
    "win",
    "firstbloodkill",
    "firstbloodassist",
    "firsttowerkill",
    "firsttowerassist",
)

# Per-game maxima (weighted average, NOT per-minute); also matchups-weighted.
LARGEST_AVG_METRICS: tuple[str, ...] = (
    "largestkillingspree",
    "largestmultikill",
    "largestcriticalstrike",
)

# Per-minute volume metrics; smoothed with sum_w_timeplayed as effective N.
PER_MINUTE_METRICS: tuple[str, ...] = (
    "kills",
    "deaths",
    "assists",
    "doublekills",
    "goldearned",
    "champexperience",
    "totaldamagedealt",
    "totaldamagedealttochampions",
    "physicaldamagedealt",
    "physicaldamagedealttochampions",
    "magicdamagedealt",
    "magicdamagedealttochampions",
    "truedamagedealt",
    "totaldamagetaken",
    "damageselfmitigated",
    "totalheal",
    "totalhealsonteammates",
    "totaldamageshieldedonteammates",
    "timeccingothers",
    "totalminionskilled",
    "neutralminionskilled",
    "baronkills",
    "dragonkills",
    "damagedealttoobjectives",
    "visionscore",
)

RATE_LIKE_METRICS: tuple[str, ...] = (*RATE_METRICS, *LARGEST_AVG_METRICS)
ALL_METRICS: tuple[str, ...] = (*RATE_LIKE_METRICS, *PER_MINUTE_METRICS)


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, denom, out=out, where=denom > 1e-9)
    return out.astype(np.float32)


# Derived metric formulas operate on smoothed per-row arrays. Each callable
# takes a dict mapping `smoothed_<metric>` -> ndarray and returns one ndarray.
DERIVED_METRIC_FUNCS: dict[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]] = {
    "kda_ratio": lambda d: _safe_divide(
        d["smoothed_kills"] + d["smoothed_assists"],
        np.maximum(d["smoothed_deaths"], 1e-6),
    ),
    "assist_to_kill_ratio": lambda d: _safe_divide(
        d["smoothed_assists"], np.maximum(d["smoothed_kills"], 1e-6)
    ),
    "net_kills": lambda d: (d["smoothed_kills"] - d["smoothed_deaths"]).astype(
        np.float32
    ),
    "kill_share_proxy": lambda d: _safe_divide(
        d["smoothed_kills"], d["smoothed_kills"] + d["smoothed_assists"]
    ),
    "assist_share_proxy": lambda d: _safe_divide(
        d["smoothed_assists"], d["smoothed_kills"] + d["smoothed_assists"]
    ),
    "death_pressure_ratio": lambda d: _safe_divide(
        d["smoothed_deaths"], d["smoothed_kills"] + d["smoothed_assists"]
    ),
    "combat_events": lambda d: (
        d["smoothed_kills"] + d["smoothed_deaths"] + d["smoothed_assists"]
    ).astype(np.float32),
    "positive_combat_events": lambda d: (
        d["smoothed_kills"] + d["smoothed_assists"]
    ).astype(np.float32),
    "multikill_rate_per_kill": lambda d: _safe_divide(
        d["smoothed_doublekills"], d["smoothed_kills"]
    ),
    "first_blood_participation": lambda d: (
        d["smoothed_firstbloodkill"] + d["smoothed_firstbloodassist"]
    ).astype(np.float32),
    "first_blood_kill_share": lambda d: _safe_divide(
        d["smoothed_firstbloodkill"],
        d["smoothed_firstbloodkill"] + d["smoothed_firstbloodassist"],
    ),
    "first_blood_assist_share": lambda d: _safe_divide(
        d["smoothed_firstbloodassist"],
        d["smoothed_firstbloodkill"] + d["smoothed_firstbloodassist"],
    ),
    "first_tower_participation": lambda d: (
        d["smoothed_firsttowerkill"] + d["smoothed_firsttowerassist"]
    ).astype(np.float32),
    "first_tower_kill_share": lambda d: _safe_divide(
        d["smoothed_firsttowerkill"],
        d["smoothed_firsttowerkill"] + d["smoothed_firsttowerassist"],
    ),
    "first_tower_assist_share": lambda d: _safe_divide(
        d["smoothed_firsttowerassist"],
        d["smoothed_firsttowerkill"] + d["smoothed_firsttowerassist"],
    ),
    "physical_damage_share": lambda d: _safe_divide(
        d["smoothed_physicaldamagedealt"], d["smoothed_totaldamagedealt"]
    ),
    "magic_damage_share": lambda d: _safe_divide(
        d["smoothed_magicdamagedealt"], d["smoothed_totaldamagedealt"]
    ),
    "true_damage_share": lambda d: _safe_divide(
        d["smoothed_truedamagedealt"], d["smoothed_totaldamagedealt"]
    ),
    "physical_champion_damage_share": lambda d: _safe_divide(
        d["smoothed_physicaldamagedealttochampions"],
        d["smoothed_totaldamagedealttochampions"],
    ),
    "magic_champion_damage_share": lambda d: _safe_divide(
        d["smoothed_magicdamagedealttochampions"],
        d["smoothed_totaldamagedealttochampions"],
    ),
    "physical_champion_damage_focus": lambda d: _safe_divide(
        d["smoothed_physicaldamagedealttochampions"],
        d["smoothed_physicaldamagedealt"],
    ),
    "magic_champion_damage_focus": lambda d: _safe_divide(
        d["smoothed_magicdamagedealttochampions"],
        d["smoothed_magicdamagedealt"],
    ),
    "champion_damage_focus": lambda d: _safe_divide(
        d["smoothed_totaldamagedealttochampions"], d["smoothed_totaldamagedealt"]
    ),
    "non_champion_damage_share": lambda d: _safe_divide(
        np.maximum(
            d["smoothed_totaldamagedealt"]
            - d["smoothed_totaldamagedealttochampions"],
            0.0,
        ),
        d["smoothed_totaldamagedealt"],
    ),
    "physical_vs_magic_champion_damage": lambda d: _safe_divide(
        d["smoothed_physicaldamagedealttochampions"],
        np.maximum(d["smoothed_magicdamagedealttochampions"], 1e-6),
    ),
    "damage_mitigated_ratio": lambda d: _safe_divide(
        d["smoothed_damageselfmitigated"], d["smoothed_totaldamagetaken"]
    ),
    "effective_damage_load": lambda d: (
        d["smoothed_totaldamagetaken"] + d["smoothed_damageselfmitigated"]
    ).astype(np.float32),
    "damage_dealt_taken_ratio": lambda d: _safe_divide(
        d["smoothed_totaldamagedealttochampions"], d["smoothed_totaldamagetaken"]
    ),
    "net_champion_damage_trade": lambda d: (
        d["smoothed_totaldamagedealttochampions"] - d["smoothed_totaldamagetaken"]
    ).astype(np.float32),
    "self_or_non_teammate_heal": lambda d: np.maximum(
        d["smoothed_totalheal"] - d["smoothed_totalhealsonteammates"], 0.0
    ).astype(np.float32),
    "teammate_heal_share": lambda d: _safe_divide(
        d["smoothed_totalhealsonteammates"],
        np.maximum(d["smoothed_totalheal"], 1e-6),
    ),
    "ally_protection": lambda d: (
        d["smoothed_totalhealsonteammates"]
        + d["smoothed_totaldamageshieldedonteammates"]
    ).astype(np.float32),
    "shield_to_teammate_heal_ratio": lambda d: _safe_divide(
        d["smoothed_totaldamageshieldedonteammates"],
        np.maximum(d["smoothed_totalhealsonteammates"], 1e-6),
    ),
    "protection_to_damage_taken": lambda d: _safe_divide(
        d["smoothed_totalhealsonteammates"]
        + d["smoothed_totaldamageshieldedonteammates"],
        d["smoothed_totaldamagetaken"],
    ),
    "protection_to_champion_damage": lambda d: _safe_divide(
        d["smoothed_totalhealsonteammates"]
        + d["smoothed_totaldamageshieldedonteammates"],
        d["smoothed_totaldamagedealttochampions"],
    ),
    "champion_damage_per_gold": lambda d: _safe_divide(
        d["smoothed_totaldamagedealttochampions"], d["smoothed_goldearned"]
    ),
    "damage_taken_per_gold": lambda d: _safe_divide(
        d["smoothed_totaldamagetaken"], d["smoothed_goldearned"]
    ),
    "gold_per_xp": lambda d: _safe_divide(
        d["smoothed_goldearned"], d["smoothed_champexperience"]
    ),
    "xp_per_gold": lambda d: _safe_divide(
        d["smoothed_champexperience"], d["smoothed_goldearned"]
    ),
    "gold_per_takedown": lambda d: _safe_divide(
        d["smoothed_goldearned"],
        d["smoothed_kills"] + d["smoothed_assists"],
    ),
    "objective_damage_per_gold": lambda d: _safe_divide(
        d["smoothed_damagedealttoobjectives"], d["smoothed_goldearned"]
    ),
    "xp_per_takedown": lambda d: _safe_divide(
        d["smoothed_champexperience"],
        d["smoothed_kills"] + d["smoothed_assists"],
    ),
    "total_farm": lambda d: (
        d["smoothed_totalminionskilled"] + d["smoothed_neutralminionskilled"]
    ).astype(np.float32),
    "neutral_farm_share": lambda d: _safe_divide(
        d["smoothed_neutralminionskilled"],
        d["smoothed_totalminionskilled"] + d["smoothed_neutralminionskilled"],
    ),
    "lane_farm_share": lambda d: _safe_divide(
        d["smoothed_totalminionskilled"],
        d["smoothed_totalminionskilled"] + d["smoothed_neutralminionskilled"],
    ),
    "gold_per_farm": lambda d: _safe_divide(
        d["smoothed_goldearned"],
        d["smoothed_totalminionskilled"] + d["smoothed_neutralminionskilled"],
    ),
    "xp_per_farm": lambda d: _safe_divide(
        d["smoothed_champexperience"],
        d["smoothed_totalminionskilled"] + d["smoothed_neutralminionskilled"],
    ),
    "epic_kills": lambda d: (
        d["smoothed_baronkills"] + d["smoothed_dragonkills"]
    ).astype(np.float32),
    "baron_to_dragon_ratio": lambda d: _safe_divide(
        d["smoothed_baronkills"], np.maximum(d["smoothed_dragonkills"], 1e-6)
    ),
    "objective_damage_share": lambda d: _safe_divide(
        d["smoothed_damagedealttoobjectives"], d["smoothed_totaldamagedealt"]
    ),
    "objective_vs_champion_damage": lambda d: _safe_divide(
        d["smoothed_damagedealttoobjectives"],
        d["smoothed_totaldamagedealttochampions"],
    ),
    "vision_per_gold": lambda d: _safe_divide(
        d["smoothed_visionscore"], d["smoothed_goldearned"]
    ),
    "vision_per_death": lambda d: _safe_divide(
        d["smoothed_visionscore"], np.maximum(d["smoothed_deaths"], 1e-6)
    ),
    "cc_to_takedowns": lambda d: _safe_divide(
        d["smoothed_timeccingothers"],
        d["smoothed_kills"] + d["smoothed_assists"],
    ),
    "cc_to_champion_damage": lambda d: _safe_divide(
        d["smoothed_timeccingothers"], d["smoothed_totaldamagedealttochampions"]
    ),
    "cc_taken_pressure_ratio": lambda d: _safe_divide(
        d["smoothed_timeccingothers"], d["smoothed_totaldamagetaken"]
    ),
}


# Raw feature set: the curated metric subset (excluding the extended columns
# added to support derived metrics). Mirrors the pre-experiment block.
DEFAULT_RAW_FEATURE_SET: tuple[str, ...] = (
    "win",
    "firstbloodkill",
    "firsttowerkill",
    "largestkillingspree",
    "largestmultikill",
    "largestcriticalstrike",
    "kills",
    "deaths",
    "assists",
    "goldearned",
    "totaldamagedealttochampions",
    "physicaldamagedealttochampions",
    "magicdamagedealttochampions",
    "totaldamagetaken",
    "damageselfmitigated",
    "totalhealsonteammates",
    "totaldamageshieldedonteammates",
    "timeccingothers",
    "totalminionskilled",
    "neutralminionskilled",
    "visionscore",
)

# Ratio-derived features selected by the 2026-05-26 deep sweep. They sharpen
# broad AD/AP/tank/jungle/support axes without increasing anti-pair leakage.
DEFAULT_DERIVED_RATIO_FEATURE_SET: tuple[str, ...] = (
    "physical_damage_share",
    "magic_damage_share",
    "damage_mitigated_ratio",
    "champion_damage_per_gold",
    "damage_taken_per_gold",
    "neutral_farm_share",
    "champion_damage_focus",
)

DEFAULT_EMBEDDING_FEATURE_SET: tuple[str, ...] = (
    *DEFAULT_RAW_FEATURE_SET,
    *DEFAULT_DERIVED_RATIO_FEATURE_SET,
)


@dataclass(frozen=True)
class EmbeddingConfig:
    cache_dir: Path = EMBEDDINGS_CACHE_DIR
    split: str = "train"

    # Bayesian shrinkage strengths by prior level. Rate strengths are in
    # pseudo-matchups; per-minute strengths in pseudo-sum_w_timeplayed seconds.
    prior_rate_strengths: dict[IdentityType, float] = field(
        default_factory=DEFAULT_PRIOR_RATE_STRENGTHS.copy
    )
    prior_per_minute_strengths: dict[IdentityType, float] = field(
        default_factory=DEFAULT_PRIOR_PER_MINUTE_STRENGTHS.copy
    )

    # Sample-size scale for smooth prior amplification: every row's prior
    # weights are multiplied by `1 + threshold / matchups`, so amplification
    # decays continuously from extreme at low matchups to ~1x at high matchups.
    # 50 was selected by the sweep in `experiments.py`: it still reins in rare
    # rows, but keeps enough observed signal to improve semantic grouping and
    # avoid over-pulling specialised identities into broad prior families.
    extreme_low_sample_threshold: float = 50.0

    # Threshold for forming similarity groups.
    similarity_threshold: float = 0.82
    group_min_matchups: dict[IdentityType, float] = field(
        default_factory=DEFAULT_GROUP_MIN_MATCHUPS.copy
    )
    # The HTML report is intentionally single-threshold. Compare thresholds in
    # the experiment harness, then render only the selected production lens.
    report_thresholds: tuple[float, ...] = ()
    report_path: Path = GROUPING_REPORT_PATH

    # PCA truncation drops trailing eigen-axes whose cumulative variance is
    # below `1 - projection_keep_variance`. With the curated 21-metric
    # snapshot block this collapses 84 dims to ~8 high-signal axes — keeping
    # only the archetype-defining variance and dropping the long tail that
    # was spreading semantically-similar identities across noise directions.
    projection_keep_variance: float = 0.91

    # Metrics that form the per-phase feature block. Entries are either
    # raw metric names (column exists in ALL_METRICS) or derived metric names
    # registered in DERIVED_METRIC_FUNCS.
    feature_set: tuple[str, ...] = DEFAULT_EMBEDDING_FEATURE_SET
