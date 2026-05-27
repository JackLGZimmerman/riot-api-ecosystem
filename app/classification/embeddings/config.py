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
SPECIALIST_REPORT_PATH = EMBEDDINGS_CACHE_DIR.parent / "specialist_report.html"

SOURCE_TABLE = "game_data_filtered.synergy_1vx_temporal"

PHASES: tuple[str, ...] = ("early_mid", "mid", "mid_late", "late")
PHASE_INDEX: dict[str, int] = {p: i for i, p in enumerate(PHASES)}

BUILD_GROUPS: dict[str, tuple[str, ...]] = {
    "ap": ("ability_power", "ap_off_tank"),
    "ad": ("attack_damage", "ad_off_tank"),
    "tank": ("ar_tank", "mr_tank"),
    "utility": ("utility_enchanter", "utility_protection"),
}
SIBLING_BUILD_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (labels[0], labels[1]) for labels in BUILD_GROUPS.values()
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
    IdentityType.CHAMPION_ROLE: "game_data_filtered.synergy_1vx_temporal_prior_champion_role",
    IdentityType.ROLE_BUILD: "game_data_filtered.synergy_1vx_temporal_prior_role_build",
    IdentityType.CHAMPION_BUILD: "game_data_filtered.synergy_1vx_temporal_prior_champion_build",
    IdentityType.BUILD: "game_data_filtered.synergy_1vx_temporal_prior_build",
}


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
    IdentityType.ROLE_BUILD,
    IdentityType.CHAMPION_BUILD,
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


# ---------------------------------------------------------------------------
# Metrics catalogues. Two stages:
#   1. Source: every raw column loaded from 6010 / 9000-9040 (ALL_METRICS).
#   2. Derived: pre-divided ratios computed from smoothed source columns
#      (DERIVED_METRIC_FUNCS).
# Both are kept full so future specialists can pull from them.
# ---------------------------------------------------------------------------

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
    "triplekills",
    "killingsprees",
    "goldearned",
    "champexperience",
    "totaldamagedealt",
    "totaldamagedealttochampions",
    "physicaldamagedealt",
    "physicaldamagedealttochampions",
    "magicdamagedealt",
    "magicdamagedealttochampions",
    "truedamagedealt",
    "truedamagedealttochampions",
    "damagedealttobuildings",
    "damagedealttoturrets",
    "damagedealttoobjectives",
    "damagedealttoepicmonsters",
    "totaldamagetaken",
    "physicaldamagetaken",
    "magicdamagetaken",
    "truedamagetaken",
    "damageselfmitigated",
    "totalheal",
    "totalhealsonteammates",
    "totaldamageshieldedonteammates",
    "timeccingothers",
    "totaltimeccdealt",
    "totalminionskilled",
    "neutralminionskilled",
    "totalallyjungleminionskilled",
    "totalenemyjungleminionskilled",
    "baronkills",
    "dragonkills",
    "inhibitorkills",
    "inhibitortakedowns",
    "inhibitorslost",
    "turretkills",
    "turrettakedowns",
    "turretslost",
    "visionscore",
    "wardsplaced",
    "wardskilled",
    "detectorwardsplaced",
    "visionwardsboughtingame",
)

RATE_LIKE_METRICS: tuple[str, ...] = (*RATE_METRICS, *LARGEST_AVG_METRICS)
ALL_METRICS: tuple[str, ...] = (*RATE_LIKE_METRICS, *PER_MINUTE_METRICS)


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, denom, out=out, where=denom > 1e-9)
    return out.astype(np.float32)


# Full derived-metric catalogue. Each callable takes a metric-name ->
# ndarray dict and returns one ndarray. The catalogue is preserved as a research
# menu — future specialists pick from here without recomputing source columns.
DERIVED_METRIC_FUNCS: dict[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]] = {
    # Durability
    "all_sources_taken": lambda d: (
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"]
    ),
    "all_sources_taken_to_death_ratio": lambda d: _safe_divide(
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
        d["deaths"],
    ),
    "self_or_non_teammate_heal": lambda d: np.maximum(
        d["totalheal"] - d["totalhealsonteammates"], 0.0
    ).astype(np.float32),
    "self_heal_to_taken_ratio": lambda d: _safe_divide(
        np.maximum(d["totalheal"] - d["totalhealsonteammates"], 0.0),
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
    ),
    "magic_damage_to_total_taken_ratio": lambda d: _safe_divide(
        d["magicdamagetaken"],
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
    ),
    "physical_damage_to_total_taken_ratio": lambda d: _safe_divide(
        d["physicaldamagetaken"],
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
    ),
    "self_mitigation_to_total_taken_ratio": lambda d: _safe_divide(
        d["damageselfmitigated"],
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
    ),
    "all_sources_taken_to_gold_ratio": lambda d: _safe_divide(
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
        d["goldearned"],
    ),
    # Sustained Damage
    "physical_damage_ratio": lambda d: _safe_divide(
        d["physicaldamagedealttochampions"],
        d["totaldamagedealttochampions"],
    ),
    # Damage applied per unit of "risk": taken damage compounded by death rate.
    # Catches safe long-range applicators (Xerath, Ziggs) that combine high
    # output, low taken, and low deaths into one axis.
    "damage_to_taken_to_death_ratio": lambda d: _safe_divide(
        _safe_divide(
            d["totaldamagedealttochampions"],
            d["totaldamagedealt"],
        ),
        d["deaths"],
    ),
    "damage_dealt_to_gold_ratio": lambda d: _safe_divide(
        d["totaldamagedealttochampions"],
        d["goldearned"],
    ),
    # Burst Damage
    "burst_kills_to_assists_ratio": lambda d: _safe_divide(
        d["kills"],
        d["assists"],
    ),
    "burst_kills_to_assists_to_gold_ratio": lambda d: _safe_divide(
        _safe_divide(
            d["kills"],
            d["assists"],
        ),
        d["goldearned"],
    ),
    # Vision
    "visionscore_to_wards_placed_killed_ratio": lambda d: _safe_divide(
        d["visionscore"],
        d["detectorwardsplaced"] + d["wardsplaced"] + d["wardskilled"],
    ),
    "visionscore_to_gold_ratio": lambda d: _safe_divide(
        d["visionscore"],
        d["goldearned"],
    ),
    "wards_killed_to_placed_ratio": lambda d: _safe_divide(
        d["wardskilled"],
        d["detectorwardsplaced"] + d["wardsplaced"],
    ),
    # Farming
    "jungle_to_lane_minions_ratio": lambda d: _safe_divide(
        d["totalallyjungleminionskilled"] + d["totalenemyjungleminionskilled"],
        d["totalminionskilled"],
    ),
    "total_farm": lambda d: (
        d["totalallyjungleminionskilled"]
        + d["totalenemyjungleminionskilled"]
        + d["totalminionskilled"]
    ),
    "enemy_to_ally_jungle_ratio": lambda d: _safe_divide(
        d["totalenemyjungleminionskilled"],
        d["totalallyjungleminionskilled"],
    ),
    "total_farm_to_gold_ratio": lambda d: _safe_divide(
        d["totalallyjungleminionskilled"]
        + d["totalenemyjungleminionskilled"]
        + d["totalminionskilled"],
        d["goldearned"],
    ),
    "total_farm_to_deaths_ratio": lambda d: _safe_divide(
        d["totalallyjungleminionskilled"]
        + d["totalenemyjungleminionskilled"]
        + d["totalminionskilled"],
        d["deaths"],
    ),
    # Structures
    "structure_pressure": lambda d: (
        d["turretkills"]
        + d["turrettakedowns"]
        + d["inhibitorkills"]
        + d["inhibitortakedowns"]
    ),
    "structure_loss": lambda d: d["turretslost"] + d["inhibitorslost"],
    "structure_damage_focus": lambda d: (
        d["damagedealttobuildings"] + d["damagedealttoturrets"]
    ),
    "structure_conversion": lambda d: _safe_divide(
        d["turretkills"]
        + d["turrettakedowns"]
        + d["inhibitorkills"]
        + d["inhibitortakedowns"],
        np.maximum(d["damagedealttobuildings"] + d["damagedealttoturrets"], 1.0),
    ),
    "structure_net_control": lambda d: (
        d["turretkills"]
        + d["turrettakedowns"]
        + d["inhibitorkills"]
        + d["inhibitortakedowns"]
        - d["turretslost"]
        - d["inhibitorslost"]
    ),
    # Epic Objectives
    "objective_pressure": lambda d: d["baronkills"] + d["dragonkills"],
    "objective_farm_control": lambda d: d["neutralminionskilled"],
    "objective_damage_focus": lambda d: d["damagedealttoobjectives"],
    "objective_conversion": lambda d: _safe_divide(
        d["baronkills"] + d["dragonkills"],
        np.maximum(d["damagedealttoobjectives"], 1.0),
    ),
    "objective_total_control": lambda d: (
        d["baronkills"] + d["dragonkills"] + d["neutralminionskilled"]
    ),
}


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
    extreme_low_sample_threshold: float = 50.0
    similarity_threshold: float = 0.82
    specialist_report_path: Path = SPECIALIST_REPORT_PATH
    projection_keep_variance: float = 0.91
    feature_set: tuple[str, ...] = ()


SPECIALIST_CACHE_DIR = EMBEDDINGS_CACHE_DIR / "specialists"


@dataclass(frozen=True)
class SpecialistSpec:
    """Narrow single-question embedding.

    Features are selected for independent PCA directions relevant to the
    specialist. Groups with median pairwise cosine below `min_median_sim` or
    size below `min_group_size` are dropped.
    """

    name: str
    feature_set: tuple[str, ...]
    similarity_threshold: float
    projection_keep_variance: float
    min_median_sim: float = 0.95
    min_group_size: int = 3


# Active specialist registry. Previously registered names
# (kept for reference): temporal, damage_profile, burst, team_utility,
# crowd_control, vision, engage, objective, farming.
SPECIALISTS: tuple[SpecialistSpec, ...] = (
    SpecialistSpec(
        name="durability",
        feature_set=(
            "all_sources_taken_to_death_ratio",
            "all_sources_taken_to_gold_ratio",
            "self_heal_to_taken_ratio",
            "self_mitigation_to_total_taken_ratio",
            "magic_damage_to_total_taken_ratio",
            "physical_damage_to_total_taken_ratio",
        ),
        similarity_threshold=0.72,
        projection_keep_variance=0.90,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="sustained_damage",
        feature_set=(
            "physical_damage_ratio",
            "damage_to_taken_to_death_ratio",
            "damage_dealt_to_gold_ratio",
        ),
        similarity_threshold=0.70,
        projection_keep_variance=0.85,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="burst_damage",
        feature_set=(
            "largestcriticalstrike",
            "burst_kills_to_assists_ratio",
            "burst_kills_to_assists_to_gold_ratio",
        ),
        similarity_threshold=0.60,
        projection_keep_variance=0.85,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="vision",
        feature_set=(
            "visionscore_to_gold_ratio",
            "visionscore_to_wards_placed_killed_ratio",
            "wards_killed_to_placed_ratio",
        ),
        similarity_threshold=0.68,
        projection_keep_variance=0.85,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="farming",
        feature_set=(
            "totalminionskilled",
            "neutralminionskilled",
            "total_farm_to_gold_ratio",
            "total_farm_to_deaths_ratio",
        ),
        similarity_threshold=0.88,
        projection_keep_variance=0.95,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="epic_objectives",
        feature_set=(
            "objective_pressure",
            "objective_farm_control",
            "objective_damage_focus",
            "objective_conversion",
            "objective_total_control",
        ),
        similarity_threshold=0.70,
        projection_keep_variance=0.85,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="structure",
        feature_set=(
            "structure_pressure",
            "structure_loss",
            "structure_damage_focus",
            "structure_conversion",
            "structure_net_control",
        ),
        similarity_threshold=0.70,
        projection_keep_variance=0.85,
        min_median_sim=0.85,
    ),
)
