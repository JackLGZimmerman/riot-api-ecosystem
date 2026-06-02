"""Historic relationship-detail vectors for 1v1 matchups and 2vX synergies."""

from __future__ import annotations

import logging
from itertools import combinations, product
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from app.classification.embeddings.config import (
    ITEM_VALUE_TOTALS_TABLE,
    ML_GAME_SPLIT_TABLE,
    PARTICIPANT_STATS_TABLE,
    RELATIONSHIP_DETAIL_CACHE_DIR,
)
from app.core.logging.logger import setup_logging_config
from app.core.utils.common import (
    POSITIONS,
    apply_median_mad,
    median_mad_standardise,
    sql_literal,
)
from app.core.utils.smoothing import build_group_for
from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)

_RELATIONSHIP_DETAIL_RAW_SCHEMA = "stats_only_v2"
_RELATIONSHIP_DETAIL_FEATURES: tuple[str, ...] = (
    "gold_diff_mean",
    "gold_adv_2k_net_rate",
    "gold_adv_5k_net_rate",
    "cs_diff_mean",
    "cs_adv_50_net_rate",
    "xp_diff_mean",
    "champion_damage_diff_mean",
    "damage_adv_20k_net_rate",
    "damage_taken_diff_mean",
    "objective_damage_diff_mean",
    "structure_damage_diff_mean",
    "vision_score_diff_mean",
    "ally_support_diff_mean",
    "cc_time_diff_mean",
    "damage_mitigated_diff_mean",
    "self_heal_diff_mean",
)
_RELATIONSHIP_DETAIL_DIM = len(_RELATIONSHIP_DETAIL_FEATURES)


@dataclass(frozen=True)
class RelationshipDetailResult:
    kind: str
    path: Path
    exact_rows: int
    dim: int


def _participant_context_cte(where_extra: str = "") -> str:
    return f"""
filtered_players AS (
    SELECT
        ps.matchid AS matchid,
        ps.participantid AS participantid,
        ps.teamid AS teamid,
        assumeNotNull(ps.championid) AS championid,
        toString(ps.teamposition) AS teamposition,
        toString(ivt.highest_value_label) AS build,
        ps.goldearned AS gold,
        ps.champexperience AS xp,
        ps.totalminionskilled + ps.neutralminionskilled AS cs,
        ps.totaldamagedealttochampions AS champion_damage,
        ps.totaldamagetaken AS damage_taken,
        ps.damagedealttoobjectives AS objective_damage,
        ps.damagedealttobuildings + ps.damagedealttoturrets AS structure_damage,
        ps.visionscore AS vision_score,
        ps.totalhealsonteammates + ps.totaldamageshieldedonteammates AS ally_support,
        ps.timeccingothers AS cc_time,
        ps.damageselfmitigated AS damage_mitigated,
        greatest(ps.totalheal - ps.totalhealsonteammates, 0) AS self_heal
    FROM {PARTICIPANT_STATS_TABLE} AS ps
    INNER JOIN {ML_GAME_SPLIT_TABLE} AS s
        ON ps.matchid = s.matchid
    INNER JOIN {ITEM_VALUE_TOTALS_TABLE} AS ivt
        ON ps.matchid = ivt.matchid AND ps.participantid = ivt.participantid
    WHERE
        s.split = 'train'
        AND isNotNull(ps.championid)
        AND toString(ps.teamposition) != 'UNKNOWN'
        {where_extra}
),
players AS (
    SELECT
        fp.matchid AS matchid,
        fp.participantid AS participantid,
        fp.teamid AS teamid,
        fp.championid AS championid,
        fp.teamposition AS teamposition,
        fp.build AS build,
        fp.gold AS gold,
        fp.xp AS xp,
        fp.cs AS cs,
        fp.champion_damage AS champion_damage,
        fp.damage_taken AS damage_taken,
        fp.objective_damage AS objective_damage,
        fp.structure_damage AS structure_damage,
        fp.vision_score AS vision_score,
        fp.ally_support AS ally_support,
        fp.cc_time AS cc_time,
        fp.damage_mitigated AS damage_mitigated,
        fp.self_heal AS self_heal
    FROM filtered_players AS fp
)
"""


def _m1v1_query(
    blue_position: str | None = None,
    red_position: str | None = None,
) -> str:
    where_extra = ""
    if blue_position is not None and red_position is not None:
        where_extra = (
            "AND ("
            f"(ps.teamid = 100 AND toString(ps.teamposition) = {sql_literal(blue_position)})"
            " OR "
            f"(ps.teamid = 200 AND toString(ps.teamposition) = {sql_literal(red_position)})"
            ")"
        )
    return f"""
WITH
{_participant_context_cte(where_extra)}
SELECT
    tupleElement(left_key, 1) AS left_championid,
    tupleElement(left_key, 2) AS left_teamposition,
    tupleElement(left_key, 3) AS left_build,
    tupleElement(right_key, 1) AS right_championid,
    tupleElement(right_key, 2) AS right_teamposition,
    tupleElement(right_key, 3) AS right_build,
    count() AS matchups,
    avg(signed_gold) AS gold_diff_mean,
    avg(signed_gold >= 2000) - avg(signed_gold <= -2000) AS gold_adv_2k_net_rate,
    avg(signed_gold >= 5000) - avg(signed_gold <= -5000) AS gold_adv_5k_net_rate,
    avg(signed_cs) AS cs_diff_mean,
    avg(signed_cs >= 50) - avg(signed_cs <= -50) AS cs_adv_50_net_rate,
    avg(signed_xp) AS xp_diff_mean,
    avg(signed_champion_damage) AS champion_damage_diff_mean,
    avg(signed_champion_damage >= 20000) - avg(signed_champion_damage <= -20000) AS damage_adv_20k_net_rate,
    avg(signed_damage_taken) AS damage_taken_diff_mean,
    avg(signed_objective_damage) AS objective_damage_diff_mean,
    avg(signed_structure_damage) AS structure_damage_diff_mean,
    avg(signed_vision_score) AS vision_score_diff_mean,
    avg(signed_ally_support) AS ally_support_diff_mean,
    avg(signed_cc_time) AS cc_time_diff_mean,
    avg(signed_damage_mitigated) AS damage_mitigated_diff_mean,
    avg(signed_self_heal) AS self_heal_diff_mean
FROM (
    SELECT
        (b.championid, b.teamposition, b.build) AS b_key,
        (r.championid, r.teamposition, r.build) AS r_key,
        if(b_key <= r_key, b_key, r_key) AS left_key,
        if(b_key <= r_key, r_key, b_key) AS right_key,
        if(b_key <= r_key, 1.0, -1.0) AS sign,
        sign * (toFloat64(b.gold) - toFloat64(r.gold)) AS signed_gold,
        sign * (toFloat64(b.cs) - toFloat64(r.cs)) AS signed_cs,
        sign * (toFloat64(b.xp) - toFloat64(r.xp)) AS signed_xp,
        sign * (toFloat64(b.champion_damage) - toFloat64(r.champion_damage)) AS signed_champion_damage,
        sign * (toFloat64(b.damage_taken) - toFloat64(r.damage_taken)) AS signed_damage_taken,
        sign * (toFloat64(b.objective_damage) - toFloat64(r.objective_damage)) AS signed_objective_damage,
        sign * (toFloat64(b.structure_damage) - toFloat64(r.structure_damage)) AS signed_structure_damage,
        sign * (toFloat64(b.vision_score) - toFloat64(r.vision_score)) AS signed_vision_score,
        sign * (toFloat64(b.ally_support) - toFloat64(r.ally_support)) AS signed_ally_support,
        sign * (toFloat64(b.cc_time) - toFloat64(r.cc_time)) AS signed_cc_time,
        sign * (toFloat64(b.damage_mitigated) - toFloat64(r.damage_mitigated)) AS signed_damage_mitigated,
        sign * (toFloat64(b.self_heal) - toFloat64(r.self_heal)) AS signed_self_heal
    FROM players AS b
    INNER JOIN players AS r
        ON b.matchid = r.matchid
    WHERE b.teamid = 100 AND r.teamid = 200
)
GROUP BY left_key, right_key
SETTINGS
    max_threads = 1,
    max_bytes_before_external_group_by = 500000000,
    max_bytes_before_external_sort = 500000000,
    join_algorithm = 'full_sorting_merge'
"""


def _s2vx_query(
    position_a: str | None = None,
    position_b: str | None = None,
) -> str:
    where_extra = ""
    pair_filter = "a.participantid < b.participantid"
    if position_a is not None and position_b is not None:
        where_extra = (
            "AND toString(ps.teamposition) IN "
            f"({sql_literal(position_a)}, {sql_literal(position_b)})"
        )
        pair_filter = (
            f"a.teamposition = {sql_literal(position_a)} "
            f"AND b.teamposition = {sql_literal(position_b)}"
        )
    return f"""
WITH
{_participant_context_cte(where_extra)}
SELECT
    tupleElement(left_key, 1) AS championid_1,
    tupleElement(left_key, 2) AS teamposition_1,
    tupleElement(left_key, 3) AS build_1,
    tupleElement(right_key, 1) AS championid_2,
    tupleElement(right_key, 2) AS teamposition_2,
    tupleElement(right_key, 3) AS build_2,
    count() AS matchups,
    avg(pair_gold) AS gold_diff_mean,
    avg(pair_gold >= 24000) AS gold_adv_2k_net_rate,
    avg(pair_gold >= 30000) AS gold_adv_5k_net_rate,
    avg(pair_cs) AS cs_diff_mean,
    avg(pair_cs >= 450) AS cs_adv_50_net_rate,
    avg(pair_xp) AS xp_diff_mean,
    avg(pair_champion_damage) AS champion_damage_diff_mean,
    avg(pair_champion_damage >= 60000) AS damage_adv_20k_net_rate,
    avg(pair_damage_taken) AS damage_taken_diff_mean,
    avg(pair_objective_damage) AS objective_damage_diff_mean,
    avg(pair_structure_damage) AS structure_damage_diff_mean,
    avg(pair_vision_score) AS vision_score_diff_mean,
    avg(pair_ally_support) AS ally_support_diff_mean,
    avg(pair_cc_time) AS cc_time_diff_mean,
    avg(pair_damage_mitigated) AS damage_mitigated_diff_mean,
    avg(pair_self_heal) AS self_heal_diff_mean
FROM (
    SELECT
        (a.championid, a.teamposition, a.build) AS a_key,
        (b.championid, b.teamposition, b.build) AS b_key,
        if(a_key <= b_key, a_key, b_key) AS left_key,
        if(a_key <= b_key, b_key, a_key) AS right_key,
        0.5 * (toFloat64(a.gold) + toFloat64(b.gold)) AS pair_gold,
        0.5 * (toFloat64(a.cs) + toFloat64(b.cs)) AS pair_cs,
        0.5 * (toFloat64(a.xp) + toFloat64(b.xp)) AS pair_xp,
        0.5 * (toFloat64(a.champion_damage) + toFloat64(b.champion_damage)) AS pair_champion_damage,
        0.5 * (toFloat64(a.damage_taken) + toFloat64(b.damage_taken)) AS pair_damage_taken,
        0.5 * (toFloat64(a.objective_damage) + toFloat64(b.objective_damage)) AS pair_objective_damage,
        0.5 * (toFloat64(a.structure_damage) + toFloat64(b.structure_damage)) AS pair_structure_damage,
        0.5 * (toFloat64(a.vision_score) + toFloat64(b.vision_score)) AS pair_vision_score,
        0.5 * (toFloat64(a.ally_support) + toFloat64(b.ally_support)) AS pair_ally_support,
        0.5 * (toFloat64(a.cc_time) + toFloat64(b.cc_time)) AS pair_cc_time,
        0.5 * (toFloat64(a.damage_mitigated) + toFloat64(b.damage_mitigated)) AS pair_damage_mitigated,
        0.5 * (toFloat64(a.self_heal) + toFloat64(b.self_heal)) AS pair_self_heal
    FROM players AS a
    INNER JOIN players AS b
        ON a.matchid = b.matchid AND a.teamid = b.teamid
    WHERE {pair_filter}
)
GROUP BY left_key, right_key
SETTINGS
    max_threads = 1,
    max_bytes_before_external_group_by = 500000000,
    max_bytes_before_external_sort = 500000000,
    join_algorithm = 'full_sorting_merge'
"""


def _level_key(kind: str, level: str, key: tuple, values: np.ndarray) -> tuple[tuple, np.ndarray]:
    left = key[:3]
    right = key[3:]
    if level == "exact":
        return key, values
    if level == "build_group":
        left_r = (left[0], left[1], build_group_for(str(left[2])))
        right_r = (right[0], right[1], build_group_for(str(right[2])))
    elif level == "nobuild":
        left_r = (left[0], left[1])
        right_r = (right[0], right[1])
    elif level == "champion":
        left_r = (left[0],)
        right_r = (right[0],)
    else:
        raise ValueError(f"Unknown relationship detail level: {level}")

    if left_r <= right_r:
        return (*left_r, *right_r), values
    signed = -values if kind == "m1v1" else values
    return (*right_r, *left_r), signed


def _weighted_levels(
    kind: str,
    exact_keys: list[tuple],
    exact_values: np.ndarray,
    matchups: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    levels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for level in ("exact", "build_group", "nobuild", "champion"):
        grouped: dict[tuple, list[tuple[np.ndarray, float]]] = defaultdict(list)
        for key, values, count in zip(exact_keys, exact_values, matchups, strict=True):
            reduced, oriented = _level_key(kind, level, key, values)
            grouped[reduced].append((oriented, float(count)))

        keys = sorted(grouped)
        raw = np.zeros((len(keys), exact_values.shape[1]), dtype=np.float64)
        counts = np.zeros(len(keys), dtype=np.float32)
        for i, key in enumerate(keys):
            total = sum(count for _, count in grouped[key])
            counts[i] = total
            if total > 0.0:
                raw[i] = sum(values * count for values, count in grouped[key]) / total
        levels[level] = (
            np.array(keys, dtype=object),
            raw.astype(np.float32),
            counts,
        )
    return levels


def _write_artifact(kind: str, rows: Iterable[tuple], path: Path) -> RelationshipDetailResult:
    rows = list(rows)
    key_width = 6
    exact_keys = [tuple(row[:key_width]) for row in rows]
    matchups = np.asarray([row[key_width] for row in rows], dtype=np.float32)
    raw_values = np.asarray(
        [
            row[key_width + 1 : key_width + 1 + len(_RELATIONSHIP_DETAIL_FEATURES)]
            for row in rows
        ],
        dtype=np.float32,
    )
    levels = _weighted_levels(kind, exact_keys, raw_values, matchups)
    exact_standardised, med, mad = median_mad_standardise(levels["exact"][1])

    payload: dict[str, np.ndarray] = {
        "feature_names": np.array(_RELATIONSHIP_DETAIL_FEATURES, dtype=object),
        "median": med,
        "mad": mad,
        "dim": np.array(_RELATIONSHIP_DETAIL_DIM, dtype=np.int32),
    }
    for level, (keys, raw, counts) in levels.items():
        payload[f"{level}_keys"] = keys
        payload[f"{level}_raw_values"] = raw.astype(np.float32)
        payload[f"{level}_values"] = (
            exact_standardised if level == "exact" else apply_median_mad(raw, med, mad)
        )
        payload[f"{level}_matchups"] = counts.astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return RelationshipDetailResult(
        kind=kind,
        path=path,
        exact_rows=len(rows),
        dim=_RELATIONSHIP_DETAIL_DIM,
    )


def _raw_cache_path(output_dir: Path, kind: str, label: str) -> Path:
    return output_dir / "_raw" / f"{kind}_{_RELATIONSHIP_DETAIL_RAW_SCHEMA}_{label}.npz"


def _save_rows(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, rows=np.asarray(rows, dtype=object))
    tmp.replace(path)


def _load_rows(path: Path) -> list[tuple] | None:
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=True)
    return [tuple(row) for row in payload["rows"].tolist()]


def _fetch_rows_cached(
    *,
    kind: str,
    label: str,
    query: str,
    output_dir: Path,
) -> list[tuple]:
    path = _raw_cache_path(output_dir, kind, label)
    cached = _load_rows(path)
    if cached is not None:
        logger.info("Loaded cached %s relationship chunk %s: rows=%d", kind, label, len(cached))
        return cached
    rows = list(get_client().query(query).result_rows)
    _save_rows(path, rows)
    logger.info("Fetched %s relationship chunk %s: rows=%d", kind, label, len(rows))
    return rows


def _m1v1_rows(output_dir: Path) -> list[tuple]:
    rows: list[tuple] = []
    for blue_position, red_position in product(POSITIONS, POSITIONS):
        label = f"{blue_position}_{red_position}"
        rows.extend(
            _fetch_rows_cached(
                kind="m1v1",
                label=label,
                query=_m1v1_query(blue_position, red_position),
                output_dir=output_dir,
            )
        )
    return rows


def _s2vx_rows(output_dir: Path) -> list[tuple]:
    rows: list[tuple] = []
    for position_a, position_b in combinations(POSITIONS, 2):
        label = f"{position_a}_{position_b}"
        rows.extend(
            _fetch_rows_cached(
                kind="s2vx",
                label=label,
                query=_s2vx_query(position_a, position_b),
                output_dir=output_dir,
            )
        )
    return rows


def write_relationship_detail_embeddings(
    *,
    output_dir: Path = RELATIONSHIP_DETAIL_CACHE_DIR,
) -> list[RelationshipDetailResult]:
    results = [
        _write_artifact("m1v1", _m1v1_rows(output_dir), output_dir / "m1v1.npz"),
        _write_artifact("s2vx", _s2vx_rows(output_dir), output_dir / "s2vx.npz"),
    ]
    for result in results:
        logger.info(
            "Wrote %s relationship detail embeddings: path=%s rows=%d dim=%d",
            result.kind,
            result.path,
            result.exact_rows,
            result.dim,
        )
    return results


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.INFO)
    write_relationship_detail_embeddings()


if __name__ == "__main__":
    main()
