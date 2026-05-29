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

# Final per-participant timeline snapshots (weighted average, NOT per-minute).
FINAL_SNAPSHOT_AVG_METRICS: tuple[str, ...] = (
    "healthmax",
    "lifesteal",
    "movementspeed",
    "omnivamp",
    "physicalvamp",
    "spellvamp",
    "armor",
    "magicresist",
    "abilitypower",
    "attackdamage",
    "attackspeed",
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

RATE_LIKE_METRICS: tuple[str, ...] = (
    *RATE_METRICS,
    *LARGEST_AVG_METRICS,
    *FINAL_SNAPSHOT_AVG_METRICS,
)
ALL_METRICS: tuple[str, ...] = (*RATE_LIKE_METRICS, *PER_MINUTE_METRICS)


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, denom, out=out, where=denom > 1e-9)
    return out.astype(np.float32)


DERIVED_METRIC_FUNCS: dict[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]] = {
    # Early pressure
    "first_blood_participation": lambda d: (
        d["firstbloodkill"] + d["firstbloodassist"]
    ).astype(np.float32),
    "first_tower_participation": lambda d: (
        d["firsttowerkill"] + d["firsttowerassist"]
    ).astype(np.float32),
    "early_snowball_participation": lambda d: (
        d["firstbloodkill"]
        + d["firstbloodassist"]
        + d["firsttowerkill"]
        + d["firsttowerassist"]
    ).astype(np.float32),
    # Durability
    "durability_total": lambda d: (
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"]
    ),
    "durability_total_to_deaths_ratio": lambda d: _safe_divide(
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
        d["deaths"],
    ),
    "self_heal": lambda d: np.maximum(
        d["totalheal"] - d["totalhealsonteammates"], 0.0
    ).astype(np.float32),
    "self_heal_to_durability_total_ratio": lambda d: _safe_divide(
        np.maximum(d["totalheal"] - d["totalhealsonteammates"], 0.0),
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
    ),
    # Durability
    "vamp_sustain": lambda d: (
        d["lifesteal"] + d["omnivamp"] + d["spellvamp"] + d["physicalvamp"]
    ).astype(np.float32),
    "healthmax_to_goldearned_ratio": lambda d: _safe_divide(
        d["healthmax"],
        d["goldearned"],
    ),
    "durability_total_to_healthmax_ratio": lambda d: _safe_divide(
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
        d["healthmax"],
    ),
    "magicdamagetaken_to_durability_total_ratio": lambda d: _safe_divide(
        d["magicdamagetaken"],
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
    ),
    "physicaldamagetaken_to_durability_total_ratio": lambda d: _safe_divide(
        d["physicaldamagetaken"],
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
    ),
    "damageselfmitigated_to_durability_total_ratio": lambda d: _safe_divide(
        d["damageselfmitigated"],
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
    ),
    "damageselfmitigated_to_goldearned_ratio": lambda d: _safe_divide(
        d["damageselfmitigated"],
        d["goldearned"],
    ),
    "durability_total_to_goldearned_ratio": lambda d: _safe_divide(
        d["damageselfmitigated"]
        + (d["totalheal"] - d["totalhealsonteammates"])
        + d["totaldamagetaken"],
        d["goldearned"],
    ),
    "damage_taken_to_goldearned_ratio": lambda d: _safe_divide(
        d["totaldamagetaken"],
        d["goldearned"],
    ),
    "totaldamagetaken_to_deaths_ratio": lambda d: _safe_divide(
        d["totaldamagetaken"],
        d["deaths"],
    ),
    "self_heal_to_goldearned_ratio": lambda d: _safe_divide(
        np.maximum(d["totalheal"] - d["totalhealsonteammates"], 0.0),
        d["goldearned"],
    ),
    "self_heal_to_deaths_ratio": lambda d: _safe_divide(
        np.maximum(d["totalheal"] - d["totalhealsonteammates"], 0.0),
        d["deaths"],
    ),
    "totalheal_to_goldearned_ratio": lambda d: _safe_divide(
        d["totalheal"],
        d["goldearned"],
    ),
    # Resistances
    "armor_to_goldearned_ratio": lambda d: _safe_divide(
        d["armor"],
        d["goldearned"],
    ),
    "magicresist_to_goldearned_ratio": lambda d: _safe_divide(
        d["magicresist"],
        d["goldearned"],
    ),
    # Sustained Damage
    "physicaldamagedealttochampions_share": lambda d: _safe_divide(
        d["physicaldamagedealttochampions"],
        d["totaldamagedealttochampions"],
    ),
    "magicdamagedealttochampions_share": lambda d: _safe_divide(
        d["magicdamagedealttochampions"],
        d["totaldamagedealttochampions"],
    ),
    "truedamagedealttochampions_share": lambda d: _safe_divide(
        d["truedamagedealttochampions"],
        d["totaldamagedealttochampions"],
    ),
    "champion_damage_to_total_damage_ratio": lambda d: _safe_divide(
        d["totaldamagedealttochampions"],
        d["totaldamagedealt"],
    ),
    "champion_damage_share_to_deaths_ratio": lambda d: _safe_divide(
        _safe_divide(
            d["totaldamagedealttochampions"],
            d["totaldamagedealt"],
        ),
        d["deaths"],
    ),
    "totaldamagedealttochampions_to_goldearned_ratio": lambda d: _safe_divide(
        d["totaldamagedealttochampions"],
        d["goldearned"],
    ),
    "totaldamagedealttochampions_to_deaths_ratio": lambda d: _safe_divide(
        d["totaldamagedealttochampions"],
        d["deaths"],
    ),
    "physicaldamagedealt_share": lambda d: _safe_divide(
        d["physicaldamagedealt"],
        d["totaldamagedealt"],
    ),
    "magicdamagedealt_share": lambda d: _safe_divide(
        d["magicdamagedealt"],
        d["totaldamagedealt"],
    ),
    "truedamagedealt_share": lambda d: _safe_divide(
        d["truedamagedealt"],
        d["totaldamagedealt"],
    ),
    # Ability Power
    "abilitypower_to_goldearned_ratio": lambda d: _safe_divide(
        d["abilitypower"],
        d["goldearned"],
    ),
    # Attack Damage
    "attackdamage_to_goldearned_ratio": lambda d: _safe_divide(
        d["attackdamage"],
        d["goldearned"],
    ),
    # Burst Damage
    "takedowns": lambda d: (d["kills"] + d["assists"]).astype(np.float32),
    "kills_to_deaths_ratio": lambda d: _safe_divide(
        d["kills"],
        d["deaths"],
    ),
    "assists_to_deaths_ratio": lambda d: _safe_divide(
        d["assists"],
        d["deaths"],
    ),
    "takedowns_to_deaths_ratio": lambda d: _safe_divide(
        d["kills"] + d["assists"],
        d["deaths"],
    ),
    "kills_to_assists_ratio": lambda d: _safe_divide(
        d["kills"],
        d["assists"],
    ),
    "kills_to_assists_ratio_to_goldearned_ratio": lambda d: _safe_divide(
        _safe_divide(
            d["kills"],
            d["assists"],
        ),
        d["goldearned"],
    ),
    # Vision
    "visionscore_to_ward_actions_ratio": lambda d: _safe_divide(
        d["visionscore"],
        d["detectorwardsplaced"] + d["wardsplaced"] + d["wardskilled"],
    ),
    "visionscore_to_goldearned_ratio": lambda d: _safe_divide(
        d["visionscore"],
        d["goldearned"],
    ),
    "wardskilled_to_wardsplaced_ratio": lambda d: _safe_divide(
        d["wardskilled"],
        d["detectorwardsplaced"] + d["wardsplaced"],
    ),
    # Farming
    "jungle_minions": lambda d: (
        d["totalallyjungleminionskilled"] + d["totalenemyjungleminionskilled"]
    ).astype(np.float32),
    "jungle_minion_share": lambda d: _safe_divide(
        d["totalallyjungleminionskilled"] + d["totalenemyjungleminionskilled"],
        d["totalallyjungleminionskilled"]
        + d["totalenemyjungleminionskilled"]
        + d["totalminionskilled"],
    ),
    "jungle_minions_to_lane_minions_ratio": lambda d: _safe_divide(
        d["totalallyjungleminionskilled"] + d["totalenemyjungleminionskilled"],
        d["totalminionskilled"],
    ),
    "total_farm": lambda d: (
        d["totalallyjungleminionskilled"]
        + d["totalenemyjungleminionskilled"]
        + d["totalminionskilled"]
    ),
    "enemy_to_ally_jungle_minions_ratio": lambda d: _safe_divide(
        d["totalenemyjungleminionskilled"],
        d["totalallyjungleminionskilled"],
    ),
    "enemy_jungle_minion_share": lambda d: _safe_divide(
        d["totalenemyjungleminionskilled"],
        d["totalallyjungleminionskilled"] + d["totalenemyjungleminionskilled"],
    ),
    "total_farm_to_goldearned_ratio": lambda d: _safe_divide(
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
    "champexperience_to_goldearned_ratio": lambda d: _safe_divide(
        d["champexperience"],
        d["goldearned"],
    ),
    # Structures
    "structure_takedowns": lambda d: (
        d["turretkills"]
        + d["turrettakedowns"]
        + d["inhibitorkills"]
        + d["inhibitortakedowns"]
    ),
    "structure_losses": lambda d: d["turretslost"] + d["inhibitorslost"],
    "structure_damage": lambda d: (
        d["damagedealttobuildings"] + d["damagedealttoturrets"]
    ),
    "structure_takedowns_to_structure_damage_ratio": lambda d: _safe_divide(
        d["turretkills"]
        + d["turrettakedowns"]
        + d["inhibitorkills"]
        + d["inhibitortakedowns"],
        np.maximum(d["damagedealttobuildings"] + d["damagedealttoturrets"], 1.0),
    ),
    "structure_damage_to_goldearned_ratio": lambda d: _safe_divide(
        d["damagedealttobuildings"] + d["damagedealttoturrets"],
        d["goldearned"],
    ),
    "structure_damage_to_deaths_ratio": lambda d: _safe_divide(
        d["damagedealttobuildings"] + d["damagedealttoturrets"],
        d["deaths"],
    ),
    "structure_takedowns_to_goldearned_ratio": lambda d: _safe_divide(
        d["turretkills"]
        + d["turrettakedowns"]
        + d["inhibitorkills"]
        + d["inhibitortakedowns"],
        d["goldearned"],
    ),
    "structure_takedowns_to_deaths_ratio": lambda d: _safe_divide(
        d["turretkills"]
        + d["turrettakedowns"]
        + d["inhibitorkills"]
        + d["inhibitortakedowns"],
        d["deaths"],
    ),
    "structure_takedowns_to_losses_ratio": lambda d: _safe_divide(
        d["turretkills"]
        + d["turrettakedowns"]
        + d["inhibitorkills"]
        + d["inhibitortakedowns"],
        np.maximum(d["turretslost"] + d["inhibitorslost"], 1.0),
    ),
    "structure_net_control": lambda d: (
        d["turretkills"]
        + d["turrettakedowns"]
        + d["inhibitorkills"]
        + d["inhibitortakedowns"]
        - d["turretslost"]
        - d["inhibitorslost"]
    ),
    # Crowd Control
    "cc_effectiveness_ratio": lambda d: _safe_divide(
        d["timeccingothers"],
        np.maximum(d["totaltimeccdealt"], 1.0),
    ),
    "cc_to_assists_ratio": lambda d: _safe_divide(
        d["timeccingothers"],
        d["assists"],
    ),
    # Epic Objectives
    "epic_kills": lambda d: d["baronkills"] + d["dragonkills"],
    "objective_neutral_minions": lambda d: d["neutralminionskilled"],
    "objective_damage": lambda d: d["damagedealttoobjectives"],
    "epic_kills_to_damagedealttoobjectives_ratio": lambda d: _safe_divide(
        d["baronkills"] + d["dragonkills"],
        np.maximum(d["damagedealttoobjectives"], 1.0),
    ),
    "objective_damage_to_goldearned_ratio": lambda d: _safe_divide(
        d["damagedealttoobjectives"],
        d["goldearned"],
    ),
    "objective_damage_to_total_damage_ratio": lambda d: _safe_divide(
        d["damagedealttoobjectives"],
        d["totaldamagedealt"],
    ),
    "epic_monster_damage_to_objective_damage_ratio": lambda d: _safe_divide(
        d["damagedealttoepicmonsters"],
        d["damagedealttoobjectives"],
    ),
    "epic_kills_to_goldearned_ratio": lambda d: _safe_divide(
        d["baronkills"] + d["dragonkills"],
        d["goldearned"],
    ),
    "damagedealttoobjectives_per_epic_kill_per_gold": lambda d: _safe_divide(
        _safe_divide(
            d["damagedealttoobjectives"],
            d["baronkills"] + d["dragonkills"],
        ),
        d["goldearned"],
    ),
    # Enchanter
    "ally_support": lambda d: (
        d["totalhealsonteammates"] + d["totaldamageshieldedonteammates"]
    ).astype(np.float32),
    "totalhealsonteammates_to_goldearned_ratio": lambda d: _safe_divide(
        d["totalhealsonteammates"],
        d["goldearned"],
    ),
    "totaldamageshieldedonteammates_to_goldearned_ratio": lambda d: _safe_divide(
        d["totaldamageshieldedonteammates"],
        d["goldearned"],
    ),
    "ally_support_to_goldearned_ratio": lambda d: _safe_divide(
        d["totalhealsonteammates"] + d["totaldamageshieldedonteammates"],
        d["goldearned"],
    ),
    "ally_support_to_assists_ratio": lambda d: _safe_divide(
        d["totalhealsonteammates"] + d["totaldamageshieldedonteammates"],
        d["assists"],
    ),
    "totalhealsonteammates_to_deaths_ratio": lambda d: _safe_divide(
        d["totalhealsonteammates"],
        d["deaths"],
    ),
    "totaldamageshieldedonteammates_to_deaths_ratio": lambda d: _safe_divide(
        d["totaldamageshieldedonteammates"],
        d["deaths"],
    ),
    "ally_support_to_deaths_ratio": lambda d: _safe_divide(
        d["totalhealsonteammates"] + d["totaldamageshieldedonteammates"],
        d["deaths"],
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
    similarity_threshold: float = 0.82
    specialist_report_path: Path = SPECIALIST_REPORT_PATH
    projection_keep_variance: float = 0.91
    feature_set: tuple[str, ...] = ()


SPECIALIST_CACHE_DIR = EMBEDDINGS_CACHE_DIR / "specialists"
SINGULAR_METRIC_CACHE_DIR = EMBEDDINGS_CACHE_DIR / "singular_metrics"


@dataclass(frozen=True)
class SpecialistSpec:
    """Narrow single-question embedding.

    Features are selected for independent PCA directions relevant to the
    specialist. Tuning uses semantic thresholds/features, not size floors;
    small coherent groups are valid specialist reads.
    """

    name: str
    feature_set: tuple[str, ...]
    similarity_threshold: float
    projection_keep_variance: float
    min_median_sim: float = 0.95


@dataclass(frozen=True)
class SingularMetricSpec:
    """One-dimensional phase-relative ordering.

    These are intentionally not clustered. They produce continuous rank-like
    features for metrics whose meaning is mainly "higher/lower than peers"
    rather than "member of this semantic group".
    """

    name: str
    feature: str
    higher_is_more: bool = True
    description: str = ""


SPECIALISTS: tuple[SpecialistSpec, ...] = (
    SpecialistSpec(
        name="early_agency",
        feature_set=(
            "first_blood_participation",
            "first_tower_participation",
            "early_snowball_participation",
            "kills_to_deaths_ratio",
            "assists_to_deaths_ratio",
        ),
        similarity_threshold=0.78,
        projection_keep_variance=0.70,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="durability",
        feature_set=(
            "durability_total_to_deaths_ratio",
            "durability_total_to_goldearned_ratio",
            "healthmax_to_goldearned_ratio",
            "self_heal_to_durability_total_ratio",
            "damageselfmitigated_to_durability_total_ratio",
            "magicdamagetaken_to_durability_total_ratio",
            "physicaldamagetaken_to_durability_total_ratio",
            "vamp_sustain",
            "durability_total_to_healthmax_ratio",
        ),
        similarity_threshold=0.80,
        projection_keep_variance=0.60,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="self_sustain",
        feature_set=(
            "self_heal_to_durability_total_ratio",
            "self_heal_to_goldearned_ratio",
            "self_heal_to_deaths_ratio",
            "totalheal_to_goldearned_ratio",
            "vamp_sustain",
        ),
        similarity_threshold=0.72,
        projection_keep_variance=0.90,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="damage_profile",
        feature_set=(
            "physicaldamagedealttochampions_share",
            "magicdamagedealttochampions_share",
            "truedamagedealttochampions_share",
            "physicaldamagedealt_share",
            "magicdamagedealt_share",
            "truedamagedealt_share",
        ),
        similarity_threshold=0.78,
        projection_keep_variance=0.70,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="sustained_damage",
        feature_set=(
            "totaldamagedealttochampions",
            "totaldamagedealttochampions_to_goldearned_ratio",
            "totaldamagedealttochampions_to_deaths_ratio",
            "champion_damage_to_total_damage_ratio",
        ),
        similarity_threshold=0.55,
        projection_keep_variance=0.80,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="damage_efficiency",
        feature_set=(
            "champion_damage_to_total_damage_ratio",
            "totaldamagedealttochampions_to_goldearned_ratio",
            "totaldamagedealttochampions_to_deaths_ratio",
            "kills_to_deaths_ratio",
            "takedowns_to_deaths_ratio",
        ),
        similarity_threshold=0.78,
        projection_keep_variance=0.70,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="burst_skirmish",
        feature_set=(
            "kills_to_assists_ratio",
            "kills_to_deaths_ratio",
            "takedowns_to_deaths_ratio",
            "largestkillingspree",
            "largestmultikill",
            "damageselfmitigated_to_goldearned_ratio",
            "attackspeed",
        ),
        similarity_threshold=0.78,
        projection_keep_variance=0.70,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="takedown_shape",
        feature_set=(
            "takedowns",
            "kills_to_assists_ratio",
            "kills_to_deaths_ratio",
            "assists_to_deaths_ratio",
            "takedowns_to_deaths_ratio",
            "killingsprees",
        ),
        similarity_threshold=0.74,
        projection_keep_variance=0.92,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="vision",
        feature_set=(
            "visionscore_to_goldearned_ratio",
            "visionscore_to_ward_actions_ratio",
            "wardskilled_to_wardsplaced_ratio",
        ),
        similarity_threshold=0.35,
        projection_keep_variance=0.85,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="utility_pickmaking",
        feature_set=(
            "assists_to_deaths_ratio",
            "cc_to_assists_ratio",
            "timeccingothers",
            "visionscore_to_goldearned_ratio",
            "ally_support_to_assists_ratio",
        ),
        similarity_threshold=0.72,
        projection_keep_variance=0.90,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="farming",
        feature_set=(
            "totalminionskilled",
            "neutralminionskilled",
            "total_farm_to_goldearned_ratio",
            "total_farm_to_deaths_ratio",
            # Would like invade specific markers
        ),
        similarity_threshold=0.60,
        projection_keep_variance=0.95,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="economy_scaling",
        feature_set=(
            "goldearned",
            "champexperience",
            "champexperience_to_goldearned_ratio",
            "total_farm_to_goldearned_ratio",
            "totaldamagedealttochampions_to_goldearned_ratio",
            "durability_total_to_goldearned_ratio",
        ),
        similarity_threshold=0.78,
        projection_keep_variance=0.70,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="jungle_control",
        feature_set=(
            "neutralminionskilled",
            "jungle_minions",
            "jungle_minion_share",
            "enemy_jungle_minion_share",
            "enemy_to_ally_jungle_minions_ratio",
            "total_farm_to_goldearned_ratio",
        ),
        similarity_threshold=0.76,
        projection_keep_variance=0.92,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="epic_objectives",
        feature_set=(
            "epic_kills",
            "objective_neutral_minions",
            "objective_damage_to_goldearned_ratio",
            "objective_damage_to_total_damage_ratio",
            "epic_monster_damage_to_objective_damage_ratio",
            "damagedealttoobjectives_per_epic_kill_per_gold",
        ),
        similarity_threshold=0.15,
        projection_keep_variance=0.95,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="structure",
        feature_set=(
            "structure_takedowns",
            "structure_losses",
            "structure_damage",
            "structure_takedowns_to_structure_damage_ratio",
            "structure_net_control",
            "structure_damage_to_goldearned_ratio",
            "structure_damage_to_deaths_ratio",
            "structure_takedowns_to_losses_ratio",
        ),
        similarity_threshold=0.74,
        projection_keep_variance=0.82,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="siege_pressure",
        feature_set=(
            "first_tower_participation",
            "structure_damage",
            "structure_takedowns",
            "structure_takedowns_to_structure_damage_ratio",
            "structure_damage_to_goldearned_ratio",
            "structure_takedowns_to_goldearned_ratio",
        ),
        similarity_threshold=0.72,
        projection_keep_variance=0.90,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="map_control",
        feature_set=(
            "visionscore_to_goldearned_ratio",
            "structure_net_control",
            "structure_takedowns_to_goldearned_ratio",
            "objective_damage_to_goldearned_ratio",
            "epic_kills_to_goldearned_ratio",
            "enemy_jungle_minion_share",
        ),
        similarity_threshold=0.72,
        projection_keep_variance=0.90,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="crowd_control",
        feature_set=(
            "timeccingothers",
            "totaltimeccdealt",
            "cc_effectiveness_ratio",
            "cc_to_assists_ratio",
        ),
        similarity_threshold=0.68,
        projection_keep_variance=0.85,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="enchanters",
        feature_set=(
            "totalhealsonteammates_to_goldearned_ratio",
            "totaldamageshieldedonteammates_to_goldearned_ratio",
            "ally_support_to_goldearned_ratio",
        ),
        similarity_threshold=0.70,
        projection_keep_variance=0.82,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="resistances",
        feature_set=(
            "armor",
            "magicresist",
        ),
        similarity_threshold=0.60,
        projection_keep_variance=0.90,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="defensive_statline",
        feature_set=(
            "healthmax",
            "armor",
            "magicresist",
            "damage_taken_to_goldearned_ratio",
            "totaldamagetaken_to_deaths_ratio",
            "durability_total_to_healthmax_ratio",
        ),
        similarity_threshold=0.74,
        projection_keep_variance=0.75,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="ability_power",
        feature_set=(
            "abilitypower",
            "abilitypower_to_goldearned_ratio",
            "magicdamagedealttochampions_share",
            "magicdamagedealt_share",
        ),
        similarity_threshold=0.70,
        projection_keep_variance=0.90,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="attack_damage",
        feature_set=(
            "attackdamage",
            "attackdamage_to_goldearned_ratio",
            "physicaldamagedealttochampions_share",
            "physicaldamagedealt_share",
        ),
        similarity_threshold=0.65,
        projection_keep_variance=0.85,
        min_median_sim=0.85,
    ),
    SpecialistSpec(
        name="on_hit_carry",
        feature_set=(
            "attackdamage",
            "attackspeed",
            "largestcriticalstrike",
            "physicaldamagedealttochampions_share",
            "physicaldamagedealt_share",
        ),
        similarity_threshold=0.70,
        projection_keep_variance=0.85,
        min_median_sim=0.85,
    ),
)


SINGULAR_METRICS: tuple[SingularMetricSpec, ...] = (
    SingularMetricSpec(
        name="movement_speed",
        feature="movementspeed",
        description="Phase-relative mobility ordering.",
    ),
    SingularMetricSpec(
        name="attack_speed",
        feature="attackspeed",
        description="Phase-relative basic-attack cadence ordering.",
    ),
    SingularMetricSpec(
        name="critical_strike_ceiling",
        feature="largestcriticalstrike",
        description="Highest observed critical strike magnitude.",
    ),
    SingularMetricSpec(
        name="spree_ceiling",
        feature="largestkillingspree",
        description="Phase-relative snowball ceiling from killing sprees.",
    ),
    SingularMetricSpec(
        name="multikill_ceiling",
        feature="largestmultikill",
        description="Phase-relative multi-kill ceiling.",
    ),
    SingularMetricSpec(
        name="low_death_rate",
        feature="deaths",
        higher_is_more=False,
        description="Lower death rate ordered as the stronger signal.",
    ),
)
