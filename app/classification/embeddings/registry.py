"""Declarative metric registry for classification embeddings.

Single source of truth for the full-game metric catalogue. `config.py` exposes
the legacy names (`ALL_METRICS`, `RATE_METRICS`, ..., `DERIVED_METRIC_FUNCS`) as
views over this registry. See METRIC_CATALOGUE_PLAN.md.

Two stages:
  1. Raw source columns, grouped by how ClickHouse aggregates them and which
     evidence weight their priors use.
  2. Derived ratios computed from smoothed source columns at matrix-build time;
     these are not individually smoothed (evidence STATIC_NONE).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Source(str, Enum):
    """How a raw column is produced (or DERIVED for computed ratios)."""

    RATE = "rate"  # sum(x)/count() over participant_stats
    LARGEST_AVG = "largest_avg"  # per-game maxima, matchups-averaged (rate agg)
    FINAL_SNAPSHOT = "final_snapshot"  # final tl_participant_stats snapshot
    PER_MINUTE = "per_minute"  # 60*sum(x)/sum(timeplayed)
    DERIVED = "derived"  # computed from smoothed source columns


class Evidence(str, Enum):
    """Which weight column the hierarchical prior uses for a metric."""

    MATCHUPS = "matchups"
    SUM_W_TIMEPLAYED = "sum_w_timeplayed"
    STATIC_NONE = "static_none"  # derived metrics: not individually smoothed


# Raw source groups. Order is the catalogue contract: ALL_METRICS is
# (rate, largest_avg, final_snapshot, per_minute) and downstream feature
# ordering depends on it staying stable.

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

# Final participant stat snapshots (weighted average, NOT per-minute).
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

# (names, source, evidence) tuples define the raw layer in catalogue order.
_RAW_GROUPS: tuple[tuple[tuple[str, ...], Source, Evidence], ...] = (
    (RATE_METRICS, Source.RATE, Evidence.MATCHUPS),
    (LARGEST_AVG_METRICS, Source.LARGEST_AVG, Evidence.MATCHUPS),
    (FINAL_SNAPSHOT_AVG_METRICS, Source.FINAL_SNAPSHOT, Evidence.MATCHUPS),
    (PER_MINUTE_METRICS, Source.PER_MINUTE, Evidence.SUM_W_TIMEPLAYED),
)


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


# MetricSpec layer.


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: Source
    evidence_kind: Evidence
    dependencies: tuple[str, ...]  # source columns the calculation reads
    calculation: Callable[[Mapping[str, np.ndarray]], np.ndarray] | None


class _DependencyRecorder(Mapping[str, np.ndarray]):
    """Records which raw metrics a derived calculation reads.

    Returns a unit array for any valid raw metric so the calculation's
    arithmetic runs without touching real data. Rejecting unknown names means
    introspection also validates that dependencies are real raw metrics.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __getitem__(self, metric: str) -> np.ndarray:
        if metric not in ALL_METRICS:
            raise KeyError(metric)
        if metric not in self.seen:
            self.seen.append(metric)
        return np.ones(1, dtype=np.float32)

    def __iter__(self) -> Iterator[str]:
        return iter(ALL_METRICS)

    def __len__(self) -> int:
        return len(ALL_METRICS)


def _derived_dependencies(
    fn: Callable[[Mapping[str, np.ndarray]], np.ndarray],
) -> tuple[str, ...]:
    recorder = _DependencyRecorder()
    fn(recorder)
    return tuple(recorder.seen)


def _build_full_game_specs() -> tuple[MetricSpec, ...]:
    specs: list[MetricSpec] = []
    for names, source, evidence in _RAW_GROUPS:
        for name in names:
            specs.append(MetricSpec(name, source, evidence, (), None))
    for name, fn in DERIVED_METRIC_FUNCS.items():
        specs.append(
            MetricSpec(
                name,
                Source.DERIVED,
                Evidence.STATIC_NONE,
                _derived_dependencies(fn),
                fn,
            )
        )
    return tuple(specs)


FULL_GAME_SPECS: tuple[MetricSpec, ...] = _build_full_game_specs()
RAW_SPECS: tuple[MetricSpec, ...] = tuple(
    spec for spec in FULL_GAME_SPECS if spec.source is not Source.DERIVED
)
DERIVED_SPECS: tuple[MetricSpec, ...] = tuple(
    spec for spec in FULL_GAME_SPECS if spec.source is Source.DERIVED
)

# Evidence used when smoothing each raw metric's hierarchical prior.
EVIDENCE_BY_RAW_METRIC: dict[str, Evidence] = {
    spec.name: spec.evidence_kind for spec in RAW_SPECS
}


def catalogue_hash() -> str:
    """Stable hash over the full-game catalogue (name, source, evidence, order).

    Stored in the baseline `_raw` cache so adding or re-typing a metric
    invalidates stale NPZs. Derived calculation bodies are not hashed; the
    matrix golden test guards those.
    """
    parts = [
        f"{spec.name}|{spec.source.value}|{spec.evidence_kind.value}"
        for spec in FULL_GAME_SPECS
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
