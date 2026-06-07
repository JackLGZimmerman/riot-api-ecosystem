"""Production loadout and patch-only temporal feature extraction for HGNN."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from app.ml.config import DatasetConfig
from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)

LOADOUT_FEATURE_NAMES: tuple[str, ...] = (
    "spell_pair_edge_nobuild",
    "spell_pair_coverage_nobuild",
    "broad_rune_edge_nobuild",
    "broad_rune_coverage_nobuild",
    "full_rune_page_edge_nobuild",
    "full_rune_page_coverage_nobuild",
    "secondary_rune_pair_edge_nobuild",
    "secondary_rune_pair_coverage_nobuild",
    "stat_shard_edge_nobuild",
    "stat_shard_coverage_nobuild",
)
PATCH_FEATURE_NAMES: tuple[str, ...] = (
    "patch_blue_rate_logit_delta",
    "patch_blue_rate_coverage",
)

LOADOUT_FEATURE_DIM = len(LOADOUT_FEATURE_NAMES)
PATCH_FEATURE_DIM = len(PATCH_FEATURE_NAMES)
LOADOUT_SIGNED_FEATURE_INDICES: tuple[int, ...] = (0, 2, 4, 6, 8)
PATCH_SIGNED_FEATURE_INDICES: tuple[int, ...] = (0,)

EPS = 1.0e-6
SUPPORT_STRENGTH = 30.0
MIN_LOADOUT_N = 100
MIN_DEEP_N = 100
MIN_PATCH_N = 50
TMP_TABLE_PREFIX = "game_data_filtered.hgnn_prod_loadout_patch"

SPLIT_DB_NAMES = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


def _sql_str(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _ch_logit(expr: str) -> str:
    clipped = f"least(greatest(toFloat64({expr}), {EPS}), {1.0 - EPS})"
    return f"log({clipped} / (1.0 - {clipped}))"


def _tmp_suffix() -> str:
    return f"{os.getpid()}_{int(time.time() * 1000)}"


def _aggregate_table_names(suffix: str) -> dict[str, str]:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in suffix)
    return {
        "solo": f"{TMP_TABLE_PREFIX}_solo_tmp_{safe}",
        "patch_blue": f"{TMP_TABLE_PREFIX}_patch_blue_tmp_{safe}",
        "spell": f"{TMP_TABLE_PREFIX}_spell_tmp_{safe}",
        "broad_rune": f"{TMP_TABLE_PREFIX}_broad_rune_tmp_{safe}",
        "full_rune": f"{TMP_TABLE_PREFIX}_full_rune_tmp_{safe}",
        "secondary_rune": f"{TMP_TABLE_PREFIX}_secondary_rune_tmp_{safe}",
        "stat_shard": f"{TMP_TABLE_PREFIX}_stat_shard_tmp_{safe}",
    }


def _drop_aggregate_tables(client: Any, tables: dict[str, str]) -> None:
    for table in tables.values():
        client.command(f"DROP TABLE IF EXISTS {table}")


def _prepare_aggregate_tables(client: Any, suffix: str) -> dict[str, str]:
    tables = _aggregate_table_names(suffix)
    _drop_aggregate_tables(client, tables)
    settings = {"max_query_size": 200_000_000}
    client.command(
        f"""
CREATE TABLE {tables["solo"]}
ENGINE = Memory
AS
SELECT
  toInt32(ifNull(ps.championid, 0)) AS championid,
  toString(ps.teamposition) AS teamposition,
  count() AS matchups,
  avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp
  ON ps.matchid = sp.matchid AND sp.split = 'train'
GROUP BY championid, teamposition
""",
        settings=settings,
    )
    client.command(
        f"""
CREATE TABLE {tables["patch_blue"]}
ENGINE = Memory
AS
SELECT
  gi.season AS season,
  gi.patch AS patch,
  count() AS matchups,
  avg(p.blue_win) AS blue_win_rate
FROM game_data_filtered.ml_game_player_pivot AS p
INNER JOIN game_data.info AS gi
  ON p.matchid = gi.matchid
WHERE p.split = 'train'
GROUP BY season, patch
""",
        settings=settings,
    )
    client.command(
        f"""
CREATE TABLE {tables["spell"]}
ENGINE = Memory
AS
SELECT
  toInt32(ifNull(ps.championid, 0)) AS championid,
  toString(ps.teamposition) AS teamposition,
  least(ps.summoner1id, ps.summoner2id) AS spell_a,
  greatest(ps.summoner1id, ps.summoner2id) AS spell_b,
  count() AS matchups,
  avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp
  ON ps.matchid = sp.matchid AND sp.split = 'train'
GROUP BY championid, teamposition, spell_a, spell_b
""",
        settings=settings,
    )
    client.command(
        f"""
CREATE TABLE {tables["broad_rune"]}
ENGINE = Memory
AS
SELECT
  toInt32(ifNull(ps.championid, 0)) AS championid,
  toString(ps.teamposition) AS teamposition,
  pki.primary_perk_1 AS primary_perk_1,
  pki.primary_style AS primary_style,
  pki.sub_style AS sub_style,
  count() AS matchups,
  avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp
  ON ps.matchid = sp.matchid AND sp.split = 'train'
INNER JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
GROUP BY championid, teamposition, primary_perk_1, primary_style, sub_style
""",
        settings=settings,
    )
    client.command(
        f"""
CREATE TABLE {tables["full_rune"]}
ENGINE = Memory
AS
SELECT
  toInt32(ifNull(ps.championid, 0)) AS championid,
  toString(ps.teamposition) AS teamposition,
  pki.primary_perk_1 AS primary_perk_1,
  pki.primary_perk_2 AS primary_perk_2,
  pki.primary_perk_3 AS primary_perk_3,
  pki.primary_perk_4 AS primary_perk_4,
  pki.sub_perk_1 AS sub_perk_1,
  pki.sub_perk_2 AS sub_perk_2,
  pki.stat_offense AS stat_offense,
  pki.stat_flex AS stat_flex,
  pki.stat_defense AS stat_defense,
  count() AS matchups,
  avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp
  ON ps.matchid = sp.matchid AND sp.split = 'train'
INNER JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
GROUP BY
  championid, teamposition, primary_perk_1, primary_perk_2,
  primary_perk_3, primary_perk_4, sub_perk_1, sub_perk_2,
  stat_offense, stat_flex, stat_defense
""",
        settings=settings,
    )
    client.command(
        f"""
CREATE TABLE {tables["secondary_rune"]}
ENGINE = Memory
AS
SELECT
  toInt32(ifNull(ps.championid, 0)) AS championid,
  toString(ps.teamposition) AS teamposition,
  pki.primary_perk_1 AS primary_perk_1,
  pki.sub_perk_1 AS sub_perk_1,
  pki.sub_perk_2 AS sub_perk_2,
  count() AS matchups,
  avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp
  ON ps.matchid = sp.matchid AND sp.split = 'train'
INNER JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
GROUP BY championid, teamposition, primary_perk_1, sub_perk_1, sub_perk_2
""",
        settings=settings,
    )
    client.command(
        f"""
CREATE TABLE {tables["stat_shard"]}
ENGINE = Memory
AS
SELECT
  toInt32(ifNull(ps.championid, 0)) AS championid,
  toString(ps.teamposition) AS teamposition,
  pki.stat_offense AS stat_offense,
  pki.stat_flex AS stat_flex,
  pki.stat_defense AS stat_defense,
  count() AS matchups,
  avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp
  ON ps.matchid = sp.matchid AND sp.split = 'train'
INNER JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
GROUP BY championid, teamposition, stat_offense, stat_flex, stat_defense
""",
        settings=settings,
    )
    return tables


def _selected_sql(cfg: DatasetConfig, split: str, limit: int) -> str:
    return f"""
SELECT matchid, blue_win
FROM {cfg.player_pivot_table}
WHERE split = {_sql_str(split)}
ORDER BY matchid
LIMIT {int(limit)}
"""


def _delta_sql(rate_name: str, n_name: str, base_name: str, min_n: int) -> str:
    return (
        f"if({n_name} >= {int(min_n)}, "
        f"({_ch_logit(rate_name)} - {_ch_logit(base_name)}) "
        f"* {n_name} / ({n_name} + {SUPPORT_STRENGTH}), 0.0)"
    )


def _feature_query(
    cfg: DatasetConfig,
    split: str,
    limit: int,
    tables: dict[str, str],
) -> str:
    selected = _selected_sql(cfg, split, limit)
    is_train = "1" if split == "train" else "0"
    base_n = f"greatest(toFloat64(base_n_raw) - {is_train}, 0.0)"
    base_wr = (
        f"if({base_n} > 0.0, "
        f"(toFloat64(base_wr_raw) * toFloat64(base_n_raw) - ({is_train} * win)) "
        f"/ greatest(toFloat64(base_n_raw) - {is_train}, 1.0), 0.5)"
    )

    def adj_n(raw: str) -> str:
        return f"greatest(toFloat64({raw}) - {is_train}, 0.0)"

    def adj_wr(raw_wr: str, raw_n: str, outcome: str = "win") -> str:
        n_expr = adj_n(raw_n)
        return (
            f"if({n_expr} > 0.0, "
            f"(toFloat64({raw_wr}) * toFloat64({raw_n}) - ({is_train} * {outcome})) "
            f"/ greatest(toFloat64({raw_n}) - {is_train}, 1.0), 0.5)"
        )

    patch_blue_delta = (
        f"if(patch_blue_n_adj >= {MIN_PATCH_N}, "
        f"({_ch_logit('patch_blue_wr_adj')} - {_ch_logit('0.5')}) "
        f"* patch_blue_n_adj / (patch_blue_n_adj + {SUPPORT_STRENGTH}), 0.0)"
    )
    spell_delta = _delta_sql("spell_wr_adj", "spell_n_adj", "base_wr_adj", MIN_LOADOUT_N)
    broad_delta = _delta_sql("broad_wr_adj", "broad_n_adj", "base_wr_adj", MIN_DEEP_N)
    full_delta = _delta_sql("full_wr_adj", "full_n_adj", "broad_or_base_wr", MIN_DEEP_N)
    secondary_delta = _delta_sql(
        "secondary_wr_adj", "secondary_n_adj", "broad_or_base_wr", MIN_DEEP_N
    )
    shard_delta = _delta_sql("shard_wr_adj", "shard_n_adj", "base_wr_adj", MIN_DEEP_N)

    return f"""
WITH
selected AS (
  {selected}
),
pki_selected AS (
  SELECT
    matchid,
    teamid,
    puuid,
    stat_defense,
    stat_flex,
    stat_offense,
    primary_style,
    sub_style,
    primary_perk_1,
    primary_perk_2,
    primary_perk_3,
    primary_perk_4,
    sub_perk_1,
    sub_perk_2
  FROM game_data.participant_perk_ids
  WHERE matchid IN (SELECT matchid FROM selected)
)
SELECT
  matchid,
  any(blue_win) AS blue_win,
  any(patch_blue_delta) AS {PATCH_FEATURE_NAMES[0]},
  toFloat64(any(patch_blue_ok)) AS {PATCH_FEATURE_NAMES[1]},
  if(countIf(spell_ok AND teamid = 100) > 0, avgIf(spell_delta, spell_ok AND teamid = 100), 0.0)
    - if(countIf(spell_ok AND teamid = 200) > 0, avgIf(spell_delta, spell_ok AND teamid = 200), 0.0)
    AS {LOADOUT_FEATURE_NAMES[0]},
  toFloat64(countIf(spell_ok)) / 10.0 AS {LOADOUT_FEATURE_NAMES[1]},
  if(countIf(broad_ok AND teamid = 100) > 0, avgIf(broad_delta, broad_ok AND teamid = 100), 0.0)
    - if(countIf(broad_ok AND teamid = 200) > 0, avgIf(broad_delta, broad_ok AND teamid = 200), 0.0)
    AS {LOADOUT_FEATURE_NAMES[2]},
  toFloat64(countIf(broad_ok)) / 10.0 AS {LOADOUT_FEATURE_NAMES[3]},
  if(countIf(full_ok AND teamid = 100) > 0, avgIf(full_delta, full_ok AND teamid = 100), 0.0)
    - if(countIf(full_ok AND teamid = 200) > 0, avgIf(full_delta, full_ok AND teamid = 200), 0.0)
    AS {LOADOUT_FEATURE_NAMES[4]},
  toFloat64(countIf(full_ok)) / 10.0 AS {LOADOUT_FEATURE_NAMES[5]},
  if(countIf(secondary_ok AND teamid = 100) > 0, avgIf(secondary_delta, secondary_ok AND teamid = 100), 0.0)
    - if(countIf(secondary_ok AND teamid = 200) > 0, avgIf(secondary_delta, secondary_ok AND teamid = 200), 0.0)
    AS {LOADOUT_FEATURE_NAMES[6]},
  toFloat64(countIf(secondary_ok)) / 10.0 AS {LOADOUT_FEATURE_NAMES[7]},
  if(countIf(shard_ok AND teamid = 100) > 0, avgIf(shard_delta, shard_ok AND teamid = 100), 0.0)
    - if(countIf(shard_ok AND teamid = 200) > 0, avgIf(shard_delta, shard_ok AND teamid = 200), 0.0)
    AS {LOADOUT_FEATURE_NAMES[8]},
  toFloat64(countIf(shard_ok)) / 10.0 AS {LOADOUT_FEATURE_NAMES[9]}
FROM (
  SELECT
    *,
    patch_blue_n_adj >= {MIN_PATCH_N} AS patch_blue_ok,
    {patch_blue_delta} AS patch_blue_delta,
    spell_n_adj >= {MIN_LOADOUT_N} AS spell_ok,
    {spell_delta} AS spell_delta,
    broad_n_adj >= {MIN_DEEP_N} AS broad_ok,
    {broad_delta} AS broad_delta,
    if(broad_n_adj >= {MIN_DEEP_N}, broad_wr_adj, base_wr_adj) AS broad_or_base_wr,
    full_n_adj >= {MIN_DEEP_N} AS full_ok,
    {full_delta} AS full_delta,
    secondary_n_adj >= {MIN_DEEP_N} AS secondary_ok,
    {secondary_delta} AS secondary_delta,
    shard_n_adj >= {MIN_DEEP_N} AS shard_ok,
    {shard_delta} AS shard_delta
  FROM (
    SELECT
      *,
      {base_n} AS base_n_adj,
      {base_wr} AS base_wr_adj,
      {adj_n('patch_blue_n_raw')} AS patch_blue_n_adj,
      {adj_wr('patch_blue_wr_raw', 'patch_blue_n_raw', 'blue_win')} AS patch_blue_wr_adj,
      {adj_n('spell_n_raw')} AS spell_n_adj,
      {adj_wr('spell_wr_raw', 'spell_n_raw')} AS spell_wr_adj,
      {adj_n('broad_n_raw')} AS broad_n_adj,
      {adj_wr('broad_wr_raw', 'broad_n_raw')} AS broad_wr_adj,
      {adj_n('full_n_raw')} AS full_n_adj,
      {adj_wr('full_wr_raw', 'full_n_raw')} AS full_wr_adj,
      {adj_n('secondary_n_raw')} AS secondary_n_adj,
      {adj_wr('secondary_wr_raw', 'secondary_n_raw')} AS secondary_wr_adj,
      {adj_n('shard_n_raw')} AS shard_n_adj,
      {adj_wr('shard_wr_raw', 'shard_n_raw')} AS shard_wr_adj
    FROM (
      SELECT
        s.matchid AS matchid,
        s.blue_win AS blue_win,
        ps.teamid AS teamid,
        toFloat64(ps.win) AS win,
        ifNull(base.matchups, 0) AS base_n_raw,
        ifNull(base.win_rate, 0.5) AS base_wr_raw,
        ifNull(patch_blue.matchups, 0) AS patch_blue_n_raw,
        ifNull(patch_blue.blue_win_rate, 0.5) AS patch_blue_wr_raw,
        ifNull(spell.matchups, 0) AS spell_n_raw,
        ifNull(spell.win_rate, 0.5) AS spell_wr_raw,
        ifNull(broad.matchups, 0) AS broad_n_raw,
        ifNull(broad.win_rate, 0.5) AS broad_wr_raw,
        ifNull(full.matchups, 0) AS full_n_raw,
        ifNull(full.win_rate, 0.5) AS full_wr_raw,
        ifNull(secondary.matchups, 0) AS secondary_n_raw,
        ifNull(secondary.win_rate, 0.5) AS secondary_wr_raw,
        ifNull(shard.matchups, 0) AS shard_n_raw,
        ifNull(shard.win_rate, 0.5) AS shard_wr_raw
      FROM game_data_filtered.participant_stats AS ps
      INNER JOIN selected AS s
        ON ps.matchid = s.matchid
      INNER JOIN game_data.info AS gi
        ON ps.matchid = gi.matchid
      ANY LEFT JOIN pki_selected AS pki
        ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
      LEFT JOIN {tables["solo"]} AS base
        ON base.championid = toInt32(ifNull(ps.championid, 0))
       AND base.teamposition = toString(ps.teamposition)
      LEFT JOIN {tables["patch_blue"]} AS patch_blue
        ON patch_blue.season = gi.season
       AND patch_blue.patch = gi.patch
      LEFT JOIN {tables["spell"]} AS spell
        ON spell.championid = toInt32(ifNull(ps.championid, 0))
       AND spell.teamposition = toString(ps.teamposition)
       AND spell.spell_a = least(ps.summoner1id, ps.summoner2id)
       AND spell.spell_b = greatest(ps.summoner1id, ps.summoner2id)
      LEFT JOIN {tables["broad_rune"]} AS broad
        ON broad.championid = toInt32(ifNull(ps.championid, 0))
       AND broad.teamposition = toString(ps.teamposition)
       AND broad.primary_perk_1 = pki.primary_perk_1
       AND broad.primary_style = pki.primary_style
       AND broad.sub_style = pki.sub_style
      LEFT JOIN {tables["full_rune"]} AS full
        ON full.championid = toInt32(ifNull(ps.championid, 0))
       AND full.teamposition = toString(ps.teamposition)
       AND full.primary_perk_1 = pki.primary_perk_1
       AND full.primary_perk_2 = pki.primary_perk_2
       AND full.primary_perk_3 = pki.primary_perk_3
       AND full.primary_perk_4 = pki.primary_perk_4
       AND full.sub_perk_1 = pki.sub_perk_1
       AND full.sub_perk_2 = pki.sub_perk_2
       AND full.stat_offense = pki.stat_offense
       AND full.stat_flex = pki.stat_flex
       AND full.stat_defense = pki.stat_defense
      LEFT JOIN {tables["secondary_rune"]} AS secondary
        ON secondary.championid = toInt32(ifNull(ps.championid, 0))
       AND secondary.teamposition = toString(ps.teamposition)
       AND secondary.primary_perk_1 = pki.primary_perk_1
       AND secondary.sub_perk_1 = pki.sub_perk_1
       AND secondary.sub_perk_2 = pki.sub_perk_2
      LEFT JOIN {tables["stat_shard"]} AS shard
        ON shard.championid = toInt32(ifNull(ps.championid, 0))
       AND shard.teamposition = toString(ps.teamposition)
       AND shard.stat_offense = pki.stat_offense
       AND shard.stat_flex = pki.stat_flex
       AND shard.stat_defense = pki.stat_defense
      WHERE ps.matchid IN (SELECT matchid FROM selected)
    )
  )
)
GROUP BY matchid
ORDER BY matchid
"""


def _load_features(
    client: Any,
    cfg: DatasetConfig,
    split: str,
    limit: int,
    tables: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if limit <= 0:
        return (
            np.zeros((0, PATCH_FEATURE_DIM), dtype=np.float32),
            np.zeros((0, LOADOUT_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
        )
    query = _feature_query(cfg, split, limit, tables)
    result = client.query(query, settings={"max_query_size": 200_000_000})
    if not result.result_rows:
        return (
            np.zeros((0, PATCH_FEATURE_DIM), dtype=np.float32),
            np.zeros((0, LOADOUT_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
        )
    columns = {name: idx for idx, name in enumerate(result.column_names)}
    patch = np.asarray(
        [
            [float(row[columns[name]]) for name in PATCH_FEATURE_NAMES]
            for row in result.result_rows
        ],
        dtype=np.float32,
    )
    loadout = np.asarray(
        [
            [float(row[columns[name]]) for name in LOADOUT_FEATURE_NAMES]
            for row in result.result_rows
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [float(row[columns["blue_win"]]) for row in result.result_rows],
        dtype=np.float64,
    )
    return patch, loadout, labels


def write_loadout_patch_feature_arrays(
    *,
    cfg: DatasetConfig,
    arrays: dict[str, np.ndarray],
    split_counts: dict[str, int],
    split_order: tuple[str, ...],
) -> None:
    """Fill production loadout and patch feature memmaps.

    Aggregates are train-only. Train rows subtract the candidate game's own
    outcome from every matching aggregate before computing empirical-Bayes logit
    deltas. The rune table joins through puuid only to align the selected rune
    row to the participant; no player identity is projected into the features.
    """
    required = {"patch_features", "loadout_features", "blue_win"}
    missing = sorted(name for name in required if name not in arrays)
    if missing:
        raise ValueError("feature arrays are missing: " + ", ".join(missing))
    client = get_client()
    tables = _prepare_aggregate_tables(client, _tmp_suffix())
    try:
        offset = 0
        for split_name in split_order:
            db_split = SPLIT_DB_NAMES[split_name]
            count = int(split_counts[split_name])
            patch, loadout, labels = _load_features(
                client,
                cfg,
                db_split,
                count,
                tables,
            )
            if patch.shape[0] != count or loadout.shape[0] != count:
                raise RuntimeError(
                    f"{split_name} feature rows do not match cache rows: "
                    f"patch={patch.shape[0]} loadout={loadout.shape[0]} expected={count}"
                )
            cached_labels = arrays["blue_win"][offset : offset + count]
            if not np.array_equal(labels.astype(np.uint8), cached_labels.astype(np.uint8)):
                raise RuntimeError(
                    f"{split_name} loadout/patch labels do not align with cache labels"
                )
            arrays["patch_features"][offset : offset + count] = patch
            arrays["loadout_features"][offset : offset + count] = loadout
            offset += count
            logger.info("Wrote split %s loadout/patch features: %d games", split_name, count)
    finally:
        _drop_aggregate_tables(client, tables)
    for name in ("patch_features", "loadout_features"):
        flush = getattr(arrays[name], "flush", None)
        if flush is not None:
            flush()


def feature_metadata() -> dict[str, object]:
    return {
        "loadout_feature_names": list(LOADOUT_FEATURE_NAMES),
        "patch_feature_names": list(PATCH_FEATURE_NAMES),
        "loadout_signed_feature_indices": list(LOADOUT_SIGNED_FEATURE_INDICES),
        "patch_signed_feature_indices": list(PATCH_SIGNED_FEATURE_INDICES),
        "patch_temporal_scope": "season_patch_blue_side_only",
        "uses_player_identity_features": False,
        "puuid_usage": "join_predicate_only_for_rune_alignment",
    }


__all__ = [
    "LOADOUT_FEATURE_DIM",
    "LOADOUT_FEATURE_NAMES",
    "LOADOUT_SIGNED_FEATURE_INDICES",
    "PATCH_FEATURE_DIM",
    "PATCH_FEATURE_NAMES",
    "PATCH_SIGNED_FEATURE_INDICES",
    "feature_metadata",
    "write_loadout_patch_feature_arrays",
]
