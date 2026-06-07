"""Second-pass allowed-signal audit for HGNN central-band misses.

The first-pass audit checks broad champion/build, spell, rune, and patch priors.
This pass keeps the same non-identity boundary but asks whether more detailed
rune pages and build-profile shapes explain additional misses.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from database.clickhouse.client import get_client
from app.ml.experiments.hgnn_central_band_allowed_review import (
    _in_sql,
    _mean,
    _query_map,
    _sql_str,
)

BUILD_LABELS = (
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

MIN_DEEP_N = 100
MIN_STRONG_N = 250
TEAM_EDGE_THRESHOLD = 0.012
STRONG_SLOT_DELTA = 0.04


def _build_sort_sql() -> str:
    labels = "[" + ",".join(_sql_str(label) for label in BUILD_LABELS) + "]"
    scores = "[" + ",".join(BUILD_LABELS) + "]"
    return f"arrayReverseSort(x -> x.2, arrayZip({labels}, {scores}))"


def _margin_bucket(highest: float | None, second: float | None) -> str:
    if highest is None or second is None or highest <= 0:
        return "unknown"
    ratio = (highest - second) / highest
    if ratio < 0.20:
        return "low"
    if ratio < 0.45:
        return "medium"
    return "high"


def _slot_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (int(row["championid"]), str(row["teamposition"]), str(row["build"]))


def _team(row: dict[str, Any]) -> str:
    return "blue" if int(row["teamid"]) == 100 else "red"


def _load_champions() -> dict[int, str]:
    out: dict[int, str] = {}
    with Path("database/clickhouse/support/championid_name_map.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["_key"])] = str(row["name"])
    return out


def _champ_name(champions: dict[int, str], championid: int) -> str:
    return champions.get(int(championid), str(championid))


def _load_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        batch_number = int(data["batch_number"])
        for row in data["batch"]:
            copied = dict(row)
            copied["source_batch_number"] = batch_number
            rows.append(copied)
    return rows


def _load_allowed(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["all_games"]:
            out[str(row["matchid"])] = row
    return out


def _load_participants(client, matchids: list[str]) -> list[dict[str, Any]]:
    ids_sql = "(" + ",".join(_sql_str(matchid) for matchid in matchids) + ")"
    sorted_expr = _build_sort_sql()
    query = f"""
SELECT
  ps.matchid AS matchid,
  ps.participantid AS participantid,
  ps.teamid AS teamid,
  toInt32(ifNull(ps.championid, 0)) AS championid,
  toString(ps.teamposition) AS teamposition,
  ps.win AS win,
  least(ps.summoner1id, ps.summoner2id) AS spell_a,
  greatest(ps.summoner1id, ps.summoner2id) AS spell_b,
  coalesce(toString(piv.highest_value_label), '') AS build,
  piv.highest_value AS highest_value,
  {sorted_expr}[2].1 AS second_label,
  {sorted_expr}[2].2 AS second_value,
  gi.season AS season,
  gi.patch AS patch,
  pki.primary_style AS primary_style,
  pki.sub_style AS sub_style,
  pki.primary_perk_1 AS primary_perk_1,
  pki.primary_perk_2 AS primary_perk_2,
  pki.primary_perk_3 AS primary_perk_3,
  pki.primary_perk_4 AS primary_perk_4,
  pki.sub_perk_1 AS sub_perk_1,
  pki.sub_perk_2 AS sub_perk_2,
  pki.stat_offense AS stat_offense,
  pki.stat_flex AS stat_flex,
  pki.stat_defense AS stat_defense
FROM game_data_filtered.participant_stats AS ps
LEFT JOIN game_data_filtered.participant_item_value_totals AS piv
  ON ps.matchid = piv.matchid AND ps.participantid = piv.participantid
LEFT JOIN game_data.info AS gi
  ON ps.matchid = gi.matchid
LEFT JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
WHERE ps.matchid IN {ids_sql}
ORDER BY ps.matchid, ps.participantid
"""
    res = client.query(query, settings={"max_query_size": 100_000_000})
    rows = [dict(zip(res.column_names, row, strict=True)) for row in res.result_rows]
    for row in rows:
        row["margin_bucket"] = _margin_bucket(row["highest_value"], row["second_value"])
    return rows


def _collect_keys(games: dict[str, list[dict[str, Any]]]) -> dict[str, set[tuple[Any, ...]]]:
    keys: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for slots in games.values():
        for row in slots:
            solo = _slot_key(row)
            keys["solo"].add(solo)
            keys["broad_rune"].add(
                (
                    *solo,
                    int(row["primary_perk_1"] or 0),
                    int(row["primary_style"] or 0),
                    int(row["sub_style"] or 0),
                )
            )
            keys["full_rune"].add(
                (
                    *solo,
                    int(row["primary_perk_1"] or 0),
                    int(row["primary_perk_2"] or 0),
                    int(row["primary_perk_3"] or 0),
                    int(row["primary_perk_4"] or 0),
                    int(row["sub_perk_1"] or 0),
                    int(row["sub_perk_2"] or 0),
                    int(row["stat_offense"] or 0),
                    int(row["stat_flex"] or 0),
                    int(row["stat_defense"] or 0),
                )
            )
            keys["secondary_rune"].add(
                (
                    *solo,
                    int(row["primary_perk_1"] or 0),
                    int(row["sub_perk_1"] or 0),
                    int(row["sub_perk_2"] or 0),
                )
            )
            keys["stat_shard"].add(
                (
                    *solo,
                    int(row["stat_offense"] or 0),
                    int(row["stat_flex"] or 0),
                    int(row["stat_defense"] or 0),
                )
            )
            keys["secondary_build"].add((*solo, str(row["second_label"])))
            keys["build_margin"].add((*solo, str(row["margin_bucket"])))
    return keys


def _load_maps(client, keys: dict[str, set[tuple[Any, ...]]]) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
    sorted_expr = _build_sort_sql()
    maps: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    maps["solo"] = _query_map(
        client,
        f"""
SELECT championid, toString(teamposition) AS teamposition, toString(build) AS build, matchups, win_rate
FROM game_data_filtered.synergy_1vx
WHERE split = 'train'
  AND (championid, toString(teamposition), toString(build)) IN {_in_sql(keys["solo"])}
""",
        key_cols=["championid", "teamposition", "build"],
        value_cols=["matchups", "win_rate"],
    )
    maps["broad_rune"] = _query_map(
        client,
        f"""
SELECT toInt32(ifNull(ps.championid, 0)) AS championid,
       toString(ps.teamposition) AS teamposition,
       coalesce(toString(piv.highest_value_label), '') AS build,
       pki.primary_perk_1 AS primary_perk_1,
       pki.primary_style AS primary_style,
       pki.sub_style AS sub_style,
       count() AS matchups,
       avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
LEFT JOIN game_data_filtered.participant_item_value_totals AS piv
  ON ps.matchid = piv.matchid AND ps.participantid = piv.participantid
INNER JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
WHERE (toInt32(ifNull(ps.championid, 0)), toString(ps.teamposition),
       coalesce(toString(piv.highest_value_label), ''),
       pki.primary_perk_1, pki.primary_style, pki.sub_style) IN {_in_sql(keys["broad_rune"])}
GROUP BY championid, teamposition, build, primary_perk_1, primary_style, sub_style
""",
        key_cols=[
            "championid",
            "teamposition",
            "build",
            "primary_perk_1",
            "primary_style",
            "sub_style",
        ],
        value_cols=["matchups", "win_rate"],
    )
    maps["full_rune"] = _query_map(
        client,
        f"""
SELECT toInt32(ifNull(ps.championid, 0)) AS championid,
       toString(ps.teamposition) AS teamposition,
       coalesce(toString(piv.highest_value_label), '') AS build,
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
INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
LEFT JOIN game_data_filtered.participant_item_value_totals AS piv
  ON ps.matchid = piv.matchid AND ps.participantid = piv.participantid
INNER JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
WHERE (toInt32(ifNull(ps.championid, 0)), toString(ps.teamposition),
       coalesce(toString(piv.highest_value_label), ''),
       pki.primary_perk_1, pki.primary_perk_2, pki.primary_perk_3,
       pki.primary_perk_4, pki.sub_perk_1, pki.sub_perk_2,
       pki.stat_offense, pki.stat_flex, pki.stat_defense) IN {_in_sql(keys["full_rune"])}
GROUP BY championid, teamposition, build, primary_perk_1, primary_perk_2,
         primary_perk_3, primary_perk_4, sub_perk_1, sub_perk_2,
         stat_offense, stat_flex, stat_defense
""",
        key_cols=[
            "championid",
            "teamposition",
            "build",
            "primary_perk_1",
            "primary_perk_2",
            "primary_perk_3",
            "primary_perk_4",
            "sub_perk_1",
            "sub_perk_2",
            "stat_offense",
            "stat_flex",
            "stat_defense",
        ],
        value_cols=["matchups", "win_rate"],
    )
    maps["secondary_rune"] = _query_map(
        client,
        f"""
SELECT toInt32(ifNull(ps.championid, 0)) AS championid,
       toString(ps.teamposition) AS teamposition,
       coalesce(toString(piv.highest_value_label), '') AS build,
       pki.primary_perk_1 AS primary_perk_1,
       pki.sub_perk_1 AS sub_perk_1,
       pki.sub_perk_2 AS sub_perk_2,
       count() AS matchups,
       avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
LEFT JOIN game_data_filtered.participant_item_value_totals AS piv
  ON ps.matchid = piv.matchid AND ps.participantid = piv.participantid
INNER JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
WHERE (toInt32(ifNull(ps.championid, 0)), toString(ps.teamposition),
       coalesce(toString(piv.highest_value_label), ''),
       pki.primary_perk_1, pki.sub_perk_1, pki.sub_perk_2) IN {_in_sql(keys["secondary_rune"])}
GROUP BY championid, teamposition, build, primary_perk_1, sub_perk_1, sub_perk_2
""",
        key_cols=[
            "championid",
            "teamposition",
            "build",
            "primary_perk_1",
            "sub_perk_1",
            "sub_perk_2",
        ],
        value_cols=["matchups", "win_rate"],
    )
    maps["stat_shard"] = _query_map(
        client,
        f"""
SELECT toInt32(ifNull(ps.championid, 0)) AS championid,
       toString(ps.teamposition) AS teamposition,
       coalesce(toString(piv.highest_value_label), '') AS build,
       pki.stat_offense AS stat_offense,
       pki.stat_flex AS stat_flex,
       pki.stat_defense AS stat_defense,
       count() AS matchups,
       avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
LEFT JOIN game_data_filtered.participant_item_value_totals AS piv
  ON ps.matchid = piv.matchid AND ps.participantid = piv.participantid
INNER JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
WHERE (toInt32(ifNull(ps.championid, 0)), toString(ps.teamposition),
       coalesce(toString(piv.highest_value_label), ''),
       pki.stat_offense, pki.stat_flex, pki.stat_defense) IN {_in_sql(keys["stat_shard"])}
GROUP BY championid, teamposition, build, stat_offense, stat_flex, stat_defense
""",
        key_cols=[
            "championid",
            "teamposition",
            "build",
            "stat_offense",
            "stat_flex",
            "stat_defense",
        ],
        value_cols=["matchups", "win_rate"],
    )
    maps["secondary_build"] = _query_map(
        client,
        f"""
SELECT championid, teamposition, build, second_label, count() AS matchups, avg(win) AS win_rate
FROM (
  SELECT toInt32(ifNull(ps.championid, 0)) AS championid,
         toString(ps.teamposition) AS teamposition,
         coalesce(toString(piv.highest_value_label), '') AS build,
         {sorted_expr}[2].1 AS second_label,
         ps.win AS win
  FROM game_data_filtered.participant_stats AS ps
  INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
  INNER JOIN game_data_filtered.participant_item_value_totals AS piv
    ON ps.matchid = piv.matchid AND ps.participantid = piv.participantid
)
WHERE (championid, teamposition, build, second_label) IN {_in_sql(keys["secondary_build"])}
GROUP BY championid, teamposition, build, second_label
""",
        key_cols=["championid", "teamposition", "build", "second_label"],
        value_cols=["matchups", "win_rate"],
    )
    maps["build_margin"] = _query_map(
        client,
        f"""
SELECT championid, teamposition, build, margin_bucket, count() AS matchups, avg(win) AS win_rate
FROM (
  SELECT championid, teamposition, build, win,
         multiIf(highest_value <= 0, 'unknown',
                 (highest_value - second_value) / highest_value < 0.20, 'low',
                 (highest_value - second_value) / highest_value < 0.45, 'medium',
                 'high') AS margin_bucket
  FROM (
    SELECT toInt32(ifNull(ps.championid, 0)) AS championid,
           toString(ps.teamposition) AS teamposition,
           coalesce(toString(piv.highest_value_label), '') AS build,
           piv.highest_value AS highest_value,
           {sorted_expr}[2].2 AS second_value,
           ps.win AS win
    FROM game_data_filtered.participant_stats AS ps
    INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
    INNER JOIN game_data_filtered.participant_item_value_totals AS piv
      ON ps.matchid = piv.matchid AND ps.participantid = piv.participantid
  )
)
WHERE (championid, teamposition, build, margin_bucket) IN {_in_sql(keys["build_margin"])}
GROUP BY championid, teamposition, build, margin_bucket
""",
        key_cols=["championid", "teamposition", "build", "margin_bucket"],
        value_cols=["matchups", "win_rate"],
    )
    return maps


def _prior(maps: dict[str, dict[tuple[Any, ...], dict[str, Any]]], name: str, key: tuple[Any, ...]) -> tuple[float | None, int]:
    row = maps[name].get(key)
    if not row:
        return None, 0
    return float(row["win_rate"]), int(row["matchups"])


def _base_prior(maps: dict[str, dict[tuple[Any, ...], dict[str, Any]]], row: dict[str, Any]) -> tuple[float, int]:
    wr, n = _prior(maps, "solo", _slot_key(row))
    return (0.5 if wr is None else wr), n


def _detail(
    *,
    row: dict[str, Any],
    signal: str,
    wr: float,
    n: int,
    baseline_wr: float,
    baseline_n: int,
    extra: dict[str, Any],
    champions: dict[int, str],
) -> dict[str, Any]:
    return {
        "signal": signal,
        "team": _team(row),
        "champion": _champ_name(champions, int(row["championid"])),
        "role": str(row["teamposition"]),
        "build": str(row["build"]),
        "wr": wr,
        "n": n,
        "baseline_wr": baseline_wr,
        "baseline_n": baseline_n,
        "delta": wr - baseline_wr,
        **extra,
    }


def _slot_details(row: dict[str, Any], maps: dict[str, dict[tuple[Any, ...], dict[str, Any]]], champions: dict[int, str]) -> list[dict[str, Any]]:
    solo = _slot_key(row)
    base_wr, base_n = _base_prior(maps, row)
    broad_key = (
        *solo,
        int(row["primary_perk_1"] or 0),
        int(row["primary_style"] or 0),
        int(row["sub_style"] or 0),
    )
    broad_wr, broad_n = _prior(maps, "broad_rune", broad_key)
    broad_for_compare = broad_wr if broad_wr is not None and broad_n >= MIN_DEEP_N else base_wr
    broad_n_for_compare = broad_n if broad_wr is not None and broad_n >= MIN_DEEP_N else base_n

    details: list[dict[str, Any]] = []
    checks = [
        (
            "full_rune_page_beyond_keystone_tree",
            "full_rune",
            (
                *solo,
                int(row["primary_perk_1"] or 0),
                int(row["primary_perk_2"] or 0),
                int(row["primary_perk_3"] or 0),
                int(row["primary_perk_4"] or 0),
                int(row["sub_perk_1"] or 0),
                int(row["sub_perk_2"] or 0),
                int(row["stat_offense"] or 0),
                int(row["stat_flex"] or 0),
                int(row["stat_defense"] or 0),
            ),
            broad_for_compare,
            broad_n_for_compare,
            {
                "primary_perk_1": int(row["primary_perk_1"] or 0),
                "primary_perk_2": int(row["primary_perk_2"] or 0),
                "primary_perk_3": int(row["primary_perk_3"] or 0),
                "primary_perk_4": int(row["primary_perk_4"] or 0),
                "sub_perk_1": int(row["sub_perk_1"] or 0),
                "sub_perk_2": int(row["sub_perk_2"] or 0),
                "stat_offense": int(row["stat_offense"] or 0),
                "stat_flex": int(row["stat_flex"] or 0),
                "stat_defense": int(row["stat_defense"] or 0),
            },
        ),
        (
            "secondary_rune_pair_beyond_keystone_tree",
            "secondary_rune",
            (
                *solo,
                int(row["primary_perk_1"] or 0),
                int(row["sub_perk_1"] or 0),
                int(row["sub_perk_2"] or 0),
            ),
            broad_for_compare,
            broad_n_for_compare,
            {
                "primary_perk_1": int(row["primary_perk_1"] or 0),
                "sub_perk_1": int(row["sub_perk_1"] or 0),
                "sub_perk_2": int(row["sub_perk_2"] or 0),
            },
        ),
        (
            "stat_shard_profile",
            "stat_shard",
            (
                *solo,
                int(row["stat_offense"] or 0),
                int(row["stat_flex"] or 0),
                int(row["stat_defense"] or 0),
            ),
            base_wr,
            base_n,
            {
                "stat_offense": int(row["stat_offense"] or 0),
                "stat_flex": int(row["stat_flex"] or 0),
                "stat_defense": int(row["stat_defense"] or 0),
            },
        ),
        (
            "secondary_build_profile",
            "secondary_build",
            (*solo, str(row["second_label"])),
            base_wr,
            base_n,
            {
                "second_label": str(row["second_label"]),
                "highest_value": row["highest_value"],
                "second_value": row["second_value"],
            },
        ),
        (
            "build_margin_profile",
            "build_margin",
            (*solo, str(row["margin_bucket"])),
            base_wr,
            base_n,
            {
                "margin_bucket": str(row["margin_bucket"]),
                "highest_value": row["highest_value"],
                "second_value": row["second_value"],
            },
        ),
    ]
    for signal, map_name, key, baseline_wr, baseline_n, extra in checks:
        wr, n = _prior(maps, map_name, key)
        if wr is None or n < MIN_DEEP_N:
            continue
        details.append(
            _detail(
                row=row,
                signal=signal,
                wr=wr,
                n=n,
                baseline_wr=baseline_wr,
                baseline_n=baseline_n,
                extra=extra,
                champions=champions,
            )
        )
    return details


def _team_edge(details: list[dict[str, Any]], actual: str, signal: str) -> float | None:
    blue = _mean([d["delta"] for d in details if d["signal"] == signal and d["team"] == "blue"])
    red = _mean([d["delta"] for d in details if d["signal"] == signal and d["team"] == "red"])
    if blue is None or red is None:
        return None
    return blue - red if actual == "blue" else red - blue


def analyze(
    *,
    candidate_paths: list[Path],
    allowed_analysis_paths: list[Path],
    output: Path,
) -> dict[str, Any]:
    candidates = _load_candidates(candidate_paths)
    allowed = _load_allowed(allowed_analysis_paths)
    matchids = [str(row["matchid"]) for row in candidates]
    client = get_client()
    champions = _load_champions()
    participant_rows = _load_participants(client, matchids)
    if len(participant_rows) != len(matchids) * 10:
        raise RuntimeError(f"expected {len(matchids) * 10} participant rows, got {len(participant_rows)}")

    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in participant_rows:
        games[str(row["matchid"])].append(row)
    for rows in games.values():
        rows.sort(key=lambda r: int(r["participantid"]))

    keys = _collect_keys(games)
    maps = _load_maps(client, keys)
    prior_lookup_counts = {name: len(values) for name, values in maps.items()}

    candidate_by_match = {str(row["matchid"]): row for row in candidates}
    signal_counter: Counter[str] = Counter()
    new_only_counter: Counter[str] = Counter()
    games_with_any = 0
    games_with_any_new_only = 0
    all_games: list[dict[str, Any]] = []

    for matchid, slots in games.items():
        candidate = candidate_by_match[matchid]
        actual = str(candidate["actual_winner"])
        details = [detail for row in slots for detail in _slot_details(row, maps, champions)]
        signals: list[str] = []
        signal_edges: dict[str, float | None] = {}
        for signal in (
            "full_rune_page_beyond_keystone_tree",
            "secondary_rune_pair_beyond_keystone_tree",
            "stat_shard_profile",
            "secondary_build_profile",
            "build_margin_profile",
        ):
            edge = _team_edge(details, actual, signal)
            signal_edges[signal] = edge
            actual_details = [
                d
                for d in details
                if d["signal"] == signal and d["team"] == actual
            ]
            strong_slot = any(
                d["delta"] >= STRONG_SLOT_DELTA and d["n"] >= MIN_STRONG_N
                for d in actual_details
            )
            if (edge is not None and edge >= TEAM_EDGE_THRESHOLD) or strong_slot:
                signals.append(signal)
                signal_counter[signal] += 1

        if signals:
            games_with_any += 1
        prior_unaccounted = allowed.get(matchid, {}).get("unaccounted_influences", [])
        new_only = bool(signals and not prior_unaccounted)
        if new_only:
            games_with_any_new_only += 1
            for signal in signals:
                new_only_counter[signal] += 1

        all_games.append(
            {
                "matchid": matchid,
                "source_batch_number": candidate["source_batch_number"],
                "batch_position": candidate["batch_position"],
                "split": candidate["split"],
                "pred_blue_win": candidate["pred_blue_win"],
                "predicted_side_0_5": candidate["predicted_side"],
                "actual_winner": actual,
                "prior_unaccounted_influences": prior_unaccounted,
                "deep_unaccounted_influences": signals,
                "new_only_vs_first_pass": new_only,
                "signal_edges": signal_edges,
                "top_actual_details": {
                    signal: sorted(
                        [
                            d
                            for d in details
                            if d["signal"] == signal and d["team"] == actual
                        ],
                        key=lambda d: (d["delta"], d["n"]),
                        reverse=True,
                    )[:3]
                    for signal in (
                        "full_rune_page_beyond_keystone_tree",
                        "secondary_rune_pair_beyond_keystone_tree",
                        "stat_shard_profile",
                        "secondary_build_profile",
                        "build_margin_profile",
                    )
                },
            }
        )

    def score(row: dict[str, Any]) -> float:
        return (
            len(row["deep_unaccounted_influences"]) * 10.0
            + sum(max(edge or 0.0, 0.0) for edge in row["signal_edges"].values())
            + (5.0 if row["new_only_vs_first_pass"] else 0.0)
        )

    output_data = {
        "source": {
            "candidate_paths": [str(path) for path in candidate_paths],
            "allowed_analysis_paths": [str(path) for path in allowed_analysis_paths],
            "identity_policy": (
                "No player identity fields are selected, emitted, aggregated, or used as evidence. "
                "PUUID appears only inside ClickHouse joins for rune alignment and is not materialized."
            ),
        },
        "prior_lookup_counts": prior_lookup_counts,
        "summary": {
            "games": len(all_games),
            "deep_unaccounted_games": games_with_any,
            "deep_new_only_games_vs_first_pass": games_with_any_new_only,
            "deep_unaccounted_counts": dict(signal_counter),
            "deep_new_only_counts_vs_first_pass": dict(new_only_counter),
        },
        "representative_deep_games": sorted(all_games, key=score, reverse=True)[:30],
        "all_games": all_games,
    }
    output.write_text(json.dumps(output_data, indent=2, sort_keys=True), encoding="utf-8")
    return output_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--allowed-analysis", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze(
        candidate_paths=args.candidate,
        allowed_analysis_paths=args.allowed_analysis,
        output=args.output,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
