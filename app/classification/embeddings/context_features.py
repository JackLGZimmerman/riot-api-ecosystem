"""Participant-grain context features (Phase 3): team-share + role matchup.

These cannot be recovered from identity rows: they need the other four teammates
(team share) and the same-role opponent (matchup). They are computed at
participant-match grain in ClickHouse, averaged to (championid, teamposition,
build) identity rows, then smoothed as MATCHUPS-evidence rate-like metrics and
appended to the full-game matrix. See METRIC_CATALOGUE_PLAN.md.
"""

from __future__ import annotations

import numpy as np

from app.classification.embeddings.config import (
    ITEM_VALUE_TOTALS_TABLE,
    ML_GAME_SPLIT_TABLE,
    PARTICIPANT_STATS_TABLE,
)
from app.core.utils.common import sql_literal

# Raw participant columns the expressions below read.
_BASE_COLUMNS: tuple[str, ...] = (
    "kills",
    "deaths",
    "assists",
    "goldearned",
    "champexperience",
    "totalminionskilled",
    "totalallyjungleminionskilled",
    "totalenemyjungleminionskilled",
    "totaldamagedealttochampions",
    "totaldamagedealt",
    "totaldamagetaken",
    "damageselfmitigated",
    "totalheal",
    "totalhealsonteammates",
    "totaldamageshieldedonteammates",
    "visionscore",
    "detectorwardsplaced",
    "wardsplaced",
    "wardskilled",
    "damagedealttoobjectives",
    "damagedealttobuildings",
    "damagedealttoturrets",
    "baronkills",
    "dragonkills",
    "turretkills",
    "turrettakedowns",
    "inhibitorkills",
    "inhibitortakedowns",
)

# Participant-grain quantity for each curated metric, as a ClickHouse expression
# over the columns selected in the context CTE. Composites match the full-game
# derived definitions (takedowns, total_farm, durability_total, ...).
_EXPR: dict[str, str] = {
    "kills": "kills",
    "deaths": "deaths",
    "assists": "assists",
    "takedowns": "(kills + assists)",
    "gold": "goldearned",
    "xp": "champexperience",
    "total_farm": (
        "(totalminionskilled + totalallyjungleminionskilled"
        " + totalenemyjungleminionskilled)"
    ),
    "lane_farm": "totalminionskilled",
    "jungle_farm": "(totalallyjungleminionskilled + totalenemyjungleminionskilled)",
    "champion_damage": "totaldamagedealttochampions",
    "total_damage": "totaldamagedealt",
    "damage_taken": "totaldamagetaken",
    "self_mitigated": "damageselfmitigated",
    "durability_total": (
        "(toFloat64(damageselfmitigated)"
        " + (toFloat64(totalheal) - toFloat64(totalhealsonteammates))"
        " + toFloat64(totaldamagetaken))"
    ),
    "ally_support": "(totalhealsonteammates + totaldamageshieldedonteammates)",
    "vision_score": "visionscore",
    "ward_actions": "(detectorwardsplaced + wardsplaced + wardskilled)",
    "objective_damage": "damagedealttoobjectives",
    "structure_damage": "(damagedealttobuildings + damagedealttoturrets)",
    "epic_kills": "(baronkills + dragonkills)",
    "structure_takedowns": (
        "(turretkills + turrettakedowns + inhibitorkills + inhibitortakedowns)"
    ),
}

# Curated recipes (METRIC_CATALOGUE_PLAN.md), not cross-products.
TEAM_SHARE_METRICS: tuple[str, ...] = (
    "kills",
    "deaths",
    "assists",
    "takedowns",
    "gold",
    "xp",
    "total_farm",
    "lane_farm",
    "jungle_farm",
    "champion_damage",
    "total_damage",
    "damage_taken",
    "self_mitigated",
    "durability_total",
    "ally_support",
    "vision_score",
    "ward_actions",
    "objective_damage",
    "structure_damage",
    "epic_kills",
    "structure_takedowns",
)
MATCHUP_RAW_METRICS: tuple[str, ...] = (
    "kills",
    "deaths",
    "takedowns",
    "gold",
    "xp",
    "total_farm",
    "champion_damage",
    "damage_taken",
    "vision_score",
    "objective_damage",
    "structure_damage",
)
MATCHUP_SHARE_METRICS: tuple[str, ...] = ("gold", "xp", "total_farm", "champion_damage")
# Herfindahl concentration of a metric across the identity's five teammates:
# HHI = sum_i (x_i / team_total)^2. High = one player carries that metric.
CONCENTRATION_METRICS: tuple[str, ...] = MATCHUP_SHARE_METRICS

TEAM_SHARE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{metric}_team_share" for metric in TEAM_SHARE_METRICS
)
CONCENTRATION_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{metric}_team_concentration" for metric in CONCENTRATION_METRICS
)
# The team query emits shares then concentrations, in this order.
TEAM_FEATURE_NAMES: tuple[str, ...] = (
    *TEAM_SHARE_FEATURE_NAMES,
    *CONCENTRATION_FEATURE_NAMES,
)


def _matchup_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for metric in MATCHUP_RAW_METRICS:
        names.append(f"{metric}_vs_role_opponent_diff")
        names.append(f"{metric}_vs_role_opponent_advantage")
    for metric in MATCHUP_SHARE_METRICS:
        names.append(f"{metric}_share_vs_role_opponent_diff")
        names.append(f"{metric}_share_vs_role_opponent_advantage")
    return tuple(names)


MATCHUP_FEATURE_NAMES: tuple[str, ...] = _matchup_feature_names()
CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    *TEAM_FEATURE_NAMES,
    *MATCHUP_FEATURE_NAMES,
)

_SETTINGS = """
SETTINGS
    max_threads = 4,
    max_bytes_before_external_group_by = 2000000000,
    max_bytes_before_external_sort = 2000000000,
    join_algorithm = 'grace_hash'
"""


def _context_cte(split_sql: str, num_chunks: int, chunk_index: int) -> str:
    base_cols = ",\n        ".join(f"ps.{col} AS {col}" for col in _BASE_COLUMNS)
    # Shard by match hash so every participant of a match lands in one shard,
    # keeping team-sum and role-pair aggregations correct within a shard.
    shard = (
        f"\n        AND cityHash64(ps.matchid) % {num_chunks} = {chunk_index}"
        if num_chunks > 1
        else ""
    )
    return f"""
participant_context AS (
    SELECT
        ps.matchid AS matchid,
        ps.teamid AS teamid,
        assumeNotNull(ps.championid) AS championid,
        toString(ps.teamposition) AS teamposition,
        toString(ivt.highest_value_label) AS build,
        {base_cols}
    FROM {PARTICIPANT_STATS_TABLE} AS ps
    INNER JOIN {ML_GAME_SPLIT_TABLE} AS s
        ON ps.matchid = s.matchid
    INNER JOIN {ITEM_VALUE_TOTALS_TABLE} AS ivt
        ON ps.matchid = ivt.matchid AND ps.participantid = ivt.participantid
    WHERE
        s.split = {split_sql}
        AND isNotNull(ps.championid)
        AND toString(ps.teamposition) != 'UNKNOWN'{shard}
)
"""


def team_share_query(
    split: str, num_chunks: int = 1, chunk_index: int = 0
) -> tuple[str, tuple[str, ...]]:
    """Per-identity SUM of participant_x / team_sum(x) plus a `cnt` column.

    Team totals come from a GROUP BY (matchid, teamid) joined back to the
    participant rows. Returns sums + count so shards combine into an average.
    """
    split_sql = sql_literal(split)
    totals: list[str] = []
    aggs: list[str] = []
    for metric in TEAM_SHARE_METRICS:
        expr = f"toFloat64({_EXPR[metric]})"
        totals.append(f"sum({expr}) AS tt_{metric}")
        aggs.append(
            f"toFloat64(sum({expr} / greatest(tt.tt_{metric}, 1)))"
            f" AS {metric}_team_share"
        )
    # Sum of squares per team -> Herfindahl HHI = sum(x^2) / sum(x)^2.
    for metric in CONCENTRATION_METRICS:
        expr = f"toFloat64({_EXPR[metric]})"
        totals.append(f"sum({expr} * {expr}) AS ss_{metric}")
        aggs.append(
            f"toFloat64(sum(tt.ss_{metric}"
            f" / greatest(tt.tt_{metric} * tt.tt_{metric}, 1)))"
            f" AS {metric}_team_concentration"
        )
    query = f"""
WITH
{_context_cte(split_sql, num_chunks, chunk_index)},
team_tot AS (
    SELECT
        matchid,
        teamid,
        {",\n        ".join(totals)}
    FROM participant_context
    GROUP BY matchid, teamid
)
SELECT
    pc.championid AS championid,
    pc.teamposition AS teamposition,
    pc.build AS build,
    count() AS cnt,
    {",\n    ".join(aggs)}
FROM participant_context AS pc
INNER JOIN team_tot AS tt ON pc.matchid = tt.matchid AND pc.teamid = tt.teamid
GROUP BY championid, teamposition, build
{_SETTINGS}
"""
    return query, TEAM_FEATURE_NAMES


def matchup_query(
    split: str, num_chunks: int = 1, chunk_index: int = 0
) -> tuple[str, tuple[str, ...]]:
    """Per-identity SUM of (self - same-role opponent) diff/advantage plus `cnt`.

    Each (matchid, teamposition) has exactly two players (one per team), so the
    opponent value is `pair_sum - self` and `self - opponent = 2*self - pair_sum`.
    This replaces an 11M x 11M self-join with a GROUP BY (matchid, teamposition)
    joined back, and `HAVING count() = 2` drops unpaired roles (remakes etc.).
    """
    split_sql = sql_literal(split)
    team_totals = [
        f"sum(toFloat64({_EXPR[metric]})) AS tt_{metric}"
        for metric in MATCHUP_SHARE_METRICS
    ]
    per_part: list[str] = [
        f"toFloat64({_EXPR[metric]}) AS p_{metric}" for metric in MATCHUP_RAW_METRICS
    ]
    for metric in MATCHUP_SHARE_METRICS:
        per_part.append(
            f"toFloat64({_EXPR[metric]}) / greatest(tt.tt_{metric}, 1) AS s_{metric}"
        )
    pair_sums = [f"sum(p_{metric}) AS pp_{metric}" for metric in MATCHUP_RAW_METRICS]
    pair_sums += [f"sum(s_{metric}) AS ps_{metric}" for metric in MATCHUP_SHARE_METRICS]

    aggs: list[str] = []
    for metric in MATCHUP_RAW_METRICS:
        self_v, pair_v = f"pp.p_{metric}", f"pr.pp_{metric}"
        diff = f"2 * {self_v} - {pair_v}"
        aggs.append(
            f"toFloat64(sum({diff})) AS {metric}_vs_role_opponent_diff"
        )
        aggs.append(
            f"toFloat64(sum(({diff})"
            f" / greatest(abs({self_v}) + abs({pair_v} - {self_v}), 1)))"
            f" AS {metric}_vs_role_opponent_advantage"
        )
    for metric in MATCHUP_SHARE_METRICS:
        self_v, pair_v = f"pp.s_{metric}", f"pr.ps_{metric}"
        diff = f"2 * {self_v} - {pair_v}"
        aggs.append(
            f"toFloat64(sum({diff})) AS {metric}_share_vs_role_opponent_diff"
        )
        aggs.append(
            f"toFloat64(sum(({diff})"
            f" / greatest(abs({self_v}) + abs({pair_v} - {self_v}), 1)))"
            f" AS {metric}_share_vs_role_opponent_advantage"
        )

    query = f"""
WITH
{_context_cte(split_sql, num_chunks, chunk_index)},
team_tot AS (
    SELECT
        matchid,
        teamid,
        {",\n        ".join(team_totals)}
    FROM participant_context
    GROUP BY matchid, teamid
),
per_part AS (
    SELECT
        pc.matchid AS matchid,
        pc.teamposition AS teamposition,
        pc.championid AS championid,
        pc.build AS build,
        {",\n        ".join(per_part)}
    FROM participant_context AS pc
    INNER JOIN team_tot AS tt ON pc.matchid = tt.matchid AND pc.teamid = tt.teamid
),
pair AS (
    SELECT
        matchid,
        teamposition,
        {",\n        ".join(pair_sums)}
    FROM per_part
    GROUP BY matchid, teamposition
    HAVING count() = 2
)
SELECT
    pp.championid AS championid,
    pp.teamposition AS teamposition,
    pp.build AS build,
    count() AS cnt,
    {",\n    ".join(aggs)}
FROM per_part AS pp
INNER JOIN pair AS pr
    ON pp.matchid = pr.matchid AND pp.teamposition = pr.teamposition
GROUP BY championid, teamposition, build
{_SETTINGS}
"""
    return query, MATCHUP_FEATURE_NAMES


# --- pure-python mirrors of the SQL formulas, for unit tests ---


def team_share(participant: np.ndarray, team_total: np.ndarray) -> np.ndarray:
    return participant / np.maximum(team_total, 1.0)


def matchup_diff(participant: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    return participant - opponent


def matchup_advantage(participant: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    return (participant - opponent) / np.maximum(
        np.abs(participant) + np.abs(opponent), 1.0
    )


def concentration(shares: np.ndarray) -> np.ndarray:
    """Herfindahl concentration sum(share_i^2) along the last axis."""
    return np.sum(np.square(shares), axis=-1)
