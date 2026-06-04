from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from database.clickhouse.client import get_client

MODEL_THRESHOLD = 0.516
MIN_REL_N = 100
MIN_FULL_N = 50
MIN_LOADOUT_N = 100

SPELL_NAMES = {
    1: "Cleanse",
    3: "Exhaust",
    4: "Flash",
    6: "Ghost",
    7: "Heal",
    11: "Smite",
    12: "Teleport",
    13: "Clarity",
    14: "Ignite",
    21: "Barrier",
    32: "Mark",
}


def _sql_str(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _sql_val(value: Any) -> str:
    if isinstance(value, str):
        return _sql_str(value)
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return str(int(value))
    return str(value)


def _tuple_sql(values: tuple[Any, ...]) -> str:
    return "(" + ",".join(_sql_val(value) for value in values) + ")"


def _in_sql(items: set[tuple[Any, ...]]) -> str:
    if not items:
        return "(NULL)"
    return "(" + ",".join(_tuple_sql(item) for item in sorted(items)) + ")"


def _query_map(
    client,
    query: str,
    *,
    key_cols: list[str],
    value_cols: list[str],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    res = client.query(query, settings={"max_query_size": 100_000_000})
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in res.result_rows:
        data = dict(zip(res.column_names, row, strict=True))
        out[tuple(data[col] for col in key_cols)] = {
            col: data[col] for col in value_cols
        }
    return out


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _canon2(left: tuple[Any, ...], right: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return (left, right) if left <= right else (right, left)


def _load_champions(path: Path) -> dict[int, str]:
    champs: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            champs[int(row["_key"])] = str(row["name"])
    return champs


def _slot_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (int(row["championid"]), str(row["teamposition"]), str(row["build"]))


def _key_name(key: tuple[int, str, str], champs: dict[int, str]) -> str:
    champion, role, build = key
    return f"{champs.get(int(champion), str(champion))} {role} `{build}`"


def _spell_pair_name(spell_a: int, spell_b: int) -> str:
    left = SPELL_NAMES.get(int(spell_a), str(spell_a))
    right = SPELL_NAMES.get(int(spell_b), str(spell_b))
    return f"{left}+{right}"


def _matchup_blue_wr(
    blue_key: tuple[int, str, str],
    red_key: tuple[int, str, str],
    full_matchup: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, float | int] | None:
    left, right = _canon2(blue_key, red_key)
    row = full_matchup.get((*left, *right))
    if row is None:
        return None
    wr = float(row["left_win_rate"])
    if left != blue_key:
        wr = 1.0 - wr
    return {"blue_wr": wr, "matchups": int(row["matchups"])}


def _synergy_wr(
    left_key: tuple[Any, ...],
    right_key: tuple[Any, ...],
    synergy_map: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, float | int] | None:
    left, right = _canon2(left_key, right_key)
    row = synergy_map.get((*left, *right))
    if row is None:
        return None
    return {"wr": float(row["win_rate"]), "matchups": int(row["matchups"])}


def _load_batch_participants(client, matchids: list[str]) -> list[dict[str, Any]]:
    ids_sql = "(" + ",".join(_sql_str(matchid) for matchid in matchids) + ")"
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
  gi.season AS season,
  gi.patch AS patch,
  gi.subversion AS subversion,
  gi.gameversion AS gameversion,
  gi.gamestarttimestamp AS gamestarttimestamp,
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
    res = client.query(query)
    return [dict(zip(res.column_names, row, strict=True)) for row in res.result_rows]


def _build_key_sets(games: dict[str, list[dict[str, Any]]]) -> dict[str, set[tuple[Any, ...]]]:
    keys: dict[str, set[tuple[Any, ...]]] = {
        "solo": set(),
        "full_matchup": set(),
        "nb_matchup": set(),
        "champ_matchup": set(),
        "full_synergy": set(),
        "nb_synergy": set(),
        "champ_synergy": set(),
        "spell_full": set(),
        "spell_nb": set(),
        "rune_full": set(),
        "rune_nb": set(),
        "patch_solo": set(),
    }
    for slots in games.values():
        for row in slots:
            solo_key = _slot_key(row)
            rune_key = (
                int(row["primary_perk_1"] or 0),
                int(row["primary_style"] or 0),
                int(row["sub_style"] or 0),
            )
            keys["solo"].add(solo_key)
            keys["spell_full"].add((*solo_key, int(row["spell_a"]), int(row["spell_b"])))
            keys["spell_nb"].add((solo_key[0], solo_key[1], int(row["spell_a"]), int(row["spell_b"])))
            keys["rune_full"].add((*solo_key, *rune_key))
            keys["rune_nb"].add((solo_key[0], solo_key[1], *rune_key))
            keys["patch_solo"].add(
                (
                    int(row["season"]),
                    int(row["patch"]),
                    solo_key[0],
                    solo_key[1],
                    solo_key[2],
                )
            )

        blue_keys = [_slot_key(row) for row in slots[:5]]
        red_keys = [_slot_key(row) for row in slots[5:]]
        for blue_key in blue_keys:
            for red_key in red_keys:
                left, right = _canon2(blue_key, red_key)
                keys["full_matchup"].add((*left, *right))
                keys["nb_matchup"].add((blue_key[0], blue_key[1], red_key[0], red_key[1]))
                keys["champ_matchup"].add((blue_key[0], red_key[0]))
        for team_keys in (blue_keys, red_keys):
            for i in range(5):
                for j in range(i + 1, 5):
                    left, right = _canon2(team_keys[i], team_keys[j])
                    keys["full_synergy"].add((*left, *right))
                    nb_left, nb_right = _canon2(
                        (team_keys[i][0], team_keys[i][1]),
                        (team_keys[j][0], team_keys[j][1]),
                    )
                    keys["nb_synergy"].add((*nb_left, *nb_right))
                    champ_left, champ_right = sorted((team_keys[i][0], team_keys[j][0]))
                    keys["champ_synergy"].add((champ_left, champ_right))
    return keys


def _load_prior_maps(client, keys: dict[str, set[tuple[Any, ...]]]) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
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
    maps["full_matchup"] = _query_map(
        client,
        f"""
SELECT left_championid, toString(left_teamposition) AS left_teamposition, toString(left_build) AS left_build,
       right_championid, toString(right_teamposition) AS right_teamposition, toString(right_build) AS right_build,
       matchups, left_win_rate
FROM game_data_filtered.matchup_1v1
WHERE split = 'train'
  AND (left_championid, toString(left_teamposition), toString(left_build),
       right_championid, toString(right_teamposition), toString(right_build)) IN {_in_sql(keys["full_matchup"])}
""",
        key_cols=[
            "left_championid",
            "left_teamposition",
            "left_build",
            "right_championid",
            "right_teamposition",
            "right_build",
        ],
        value_cols=["matchups", "left_win_rate"],
    )
    maps["nb_matchup"] = _query_map(
        client,
        f"""
SELECT blue_championid, toString(blue_teamposition) AS blue_teamposition,
       red_championid, toString(red_teamposition) AS red_teamposition,
       matchups, blue_win_rate
FROM game_data_filtered.matchup_1v1_nobuild
WHERE split = 'train'
  AND (blue_championid, toString(blue_teamposition),
       red_championid, toString(red_teamposition)) IN {_in_sql(keys["nb_matchup"])}
""",
        key_cols=["blue_championid", "blue_teamposition", "red_championid", "red_teamposition"],
        value_cols=["matchups", "blue_win_rate"],
    )
    maps["champ_matchup"] = _query_map(
        client,
        f"""
SELECT blue_championid, red_championid, matchups, blue_win_rate
FROM game_data_filtered.matchup_1v1_champ
WHERE split = 'train'
  AND (blue_championid, red_championid) IN {_in_sql(keys["champ_matchup"])}
""",
        key_cols=["blue_championid", "red_championid"],
        value_cols=["matchups", "blue_win_rate"],
    )
    maps["full_synergy"] = _query_map(
        client,
        f"""
SELECT championid_1, toString(teamposition_1) AS teamposition_1, toString(build_1) AS build_1,
       championid_2, toString(teamposition_2) AS teamposition_2, toString(build_2) AS build_2,
       matchups, win_rate
FROM game_data_filtered.synergy_2vx
WHERE split = 'train'
  AND (championid_1, toString(teamposition_1), toString(build_1),
       championid_2, toString(teamposition_2), toString(build_2)) IN {_in_sql(keys["full_synergy"])}
""",
        key_cols=[
            "championid_1",
            "teamposition_1",
            "build_1",
            "championid_2",
            "teamposition_2",
            "build_2",
        ],
        value_cols=["matchups", "win_rate"],
    )
    maps["nb_synergy"] = _query_map(
        client,
        f"""
SELECT championid_1, toString(teamposition_1) AS teamposition_1,
       championid_2, toString(teamposition_2) AS teamposition_2,
       matchups, win_rate
FROM game_data_filtered.synergy_2vx_nobuild
WHERE split = 'train'
  AND (championid_1, toString(teamposition_1),
       championid_2, toString(teamposition_2)) IN {_in_sql(keys["nb_synergy"])}
""",
        key_cols=["championid_1", "teamposition_1", "championid_2", "teamposition_2"],
        value_cols=["matchups", "win_rate"],
    )
    maps["champ_synergy"] = _query_map(
        client,
        f"""
SELECT championid_1, championid_2, matchups, win_rate
FROM game_data_filtered.synergy_2vx_champ
WHERE split = 'train'
  AND (championid_1, championid_2) IN {_in_sql(keys["champ_synergy"])}
""",
        key_cols=["championid_1", "championid_2"],
        value_cols=["matchups", "win_rate"],
    )
    maps["spell_full"] = _query_map(
        client,
        f"""
SELECT toInt32(ifNull(ps.championid, 0)) AS championid, toString(ps.teamposition) AS teamposition,
       coalesce(toString(piv.highest_value_label), '') AS build,
       least(ps.summoner1id, ps.summoner2id) AS spell_a,
       greatest(ps.summoner1id, ps.summoner2id) AS spell_b,
       count() AS matchups, avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
LEFT JOIN game_data_filtered.participant_item_value_totals AS piv
  ON ps.matchid = piv.matchid AND ps.participantid = piv.participantid
WHERE (toInt32(ifNull(ps.championid, 0)), toString(ps.teamposition),
       coalesce(toString(piv.highest_value_label), ''),
       least(ps.summoner1id, ps.summoner2id),
       greatest(ps.summoner1id, ps.summoner2id)) IN {_in_sql(keys["spell_full"])}
GROUP BY championid, teamposition, build, spell_a, spell_b
""",
        key_cols=["championid", "teamposition", "build", "spell_a", "spell_b"],
        value_cols=["matchups", "win_rate"],
    )
    maps["spell_nb"] = _query_map(
        client,
        f"""
SELECT toInt32(ifNull(ps.championid, 0)) AS championid, toString(ps.teamposition) AS teamposition,
       least(ps.summoner1id, ps.summoner2id) AS spell_a,
       greatest(ps.summoner1id, ps.summoner2id) AS spell_b,
       count() AS matchups, avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
WHERE (toInt32(ifNull(ps.championid, 0)), toString(ps.teamposition),
       least(ps.summoner1id, ps.summoner2id),
       greatest(ps.summoner1id, ps.summoner2id)) IN {_in_sql(keys["spell_nb"])}
GROUP BY championid, teamposition, spell_a, spell_b
""",
        key_cols=["championid", "teamposition", "spell_a", "spell_b"],
        value_cols=["matchups", "win_rate"],
    )
    maps["rune_full"] = _query_map(
        client,
        f"""
SELECT toInt32(ifNull(ps.championid, 0)) AS championid, toString(ps.teamposition) AS teamposition,
       coalesce(toString(piv.highest_value_label), '') AS build,
       pki.primary_perk_1 AS primary_perk_1, pki.primary_style AS primary_style, pki.sub_style AS sub_style,
       count() AS matchups, avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
LEFT JOIN game_data_filtered.participant_item_value_totals AS piv
  ON ps.matchid = piv.matchid AND ps.participantid = piv.participantid
INNER JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
WHERE (toInt32(ifNull(ps.championid, 0)), toString(ps.teamposition),
       coalesce(toString(piv.highest_value_label), ''),
       pki.primary_perk_1, pki.primary_style, pki.sub_style) IN {_in_sql(keys["rune_full"])}
GROUP BY championid, teamposition, build, primary_perk_1, primary_style, sub_style
""",
        key_cols=["championid", "teamposition", "build", "primary_perk_1", "primary_style", "sub_style"],
        value_cols=["matchups", "win_rate"],
    )
    maps["rune_nb"] = _query_map(
        client,
        f"""
SELECT toInt32(ifNull(ps.championid, 0)) AS championid, toString(ps.teamposition) AS teamposition,
       pki.primary_perk_1 AS primary_perk_1, pki.primary_style AS primary_style, pki.sub_style AS sub_style,
       count() AS matchups, avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
INNER JOIN game_data.participant_perk_ids AS pki
  ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
WHERE (toInt32(ifNull(ps.championid, 0)), toString(ps.teamposition),
       pki.primary_perk_1, pki.primary_style, pki.sub_style) IN {_in_sql(keys["rune_nb"])}
GROUP BY championid, teamposition, primary_perk_1, primary_style, sub_style
""",
        key_cols=["championid", "teamposition", "primary_perk_1", "primary_style", "sub_style"],
        value_cols=["matchups", "win_rate"],
    )
    maps["patch_solo"] = _query_map(
        client,
        f"""
SELECT gi.season AS season, gi.patch AS patch,
       toInt32(ifNull(ps.championid, 0)) AS championid, toString(ps.teamposition) AS teamposition,
       coalesce(toString(piv.highest_value_label), '') AS build,
       count() AS matchups, avg(ps.win) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp ON ps.matchid = sp.matchid AND sp.split = 'train'
INNER JOIN game_data.info AS gi ON ps.matchid = gi.matchid
LEFT JOIN game_data_filtered.participant_item_value_totals AS piv
  ON ps.matchid = piv.matchid AND ps.participantid = piv.participantid
WHERE (gi.season, gi.patch, toInt32(ifNull(ps.championid, 0)), toString(ps.teamposition),
       coalesce(toString(piv.highest_value_label), '')) IN {_in_sql(keys["patch_solo"])}
GROUP BY season, patch, championid, teamposition, build
""",
        key_cols=["season", "patch", "championid", "teamposition", "build"],
        value_cols=["matchups", "win_rate"],
    )
    return maps


def _analyze_game(
    cand: dict[str, Any],
    slots: list[dict[str, Any]],
    maps: dict[str, dict[tuple[Any, ...], dict[str, Any]]],
    champs: dict[int, str],
) -> dict[str, Any]:
    actual = str(cand["actual_winner"])
    predicted_threshold = "blue" if float(cand["pred_blue_win"]) >= MODEL_THRESHOLD else "red"

    def solo_prior(key: tuple[int, str, str]) -> tuple[float, int]:
        row = maps["solo"].get(key)
        if row is None:
            return 0.5, 0
        return float(row["win_rate"]), int(row["matchups"])

    def spell_delta(row: dict[str, Any]) -> dict[str, Any] | None:
        key = _slot_key(row)
        base_wr, base_n = solo_prior(key)
        full_key = (*key, int(row["spell_a"]), int(row["spell_b"]))
        nb_key = (key[0], key[1], int(row["spell_a"]), int(row["spell_b"]))
        prior = maps["spell_full"].get(full_key)
        level = "champ-role-build-spells"
        if prior is None or int(prior["matchups"]) < MIN_LOADOUT_N:
            prior = maps["spell_nb"].get(nb_key)
            level = "champ-role-spells"
        if prior is None or int(prior["matchups"]) < MIN_LOADOUT_N:
            return None
        return {
            "level": level,
            "wr": float(prior["win_rate"]),
            "n": int(prior["matchups"]),
            "base_wr": base_wr,
            "base_n": base_n,
            "delta": float(prior["win_rate"]) - base_wr,
            "team": "blue" if int(row["teamid"]) == 100 else "red",
            "champion": champs.get(int(row["championid"]), str(row["championid"])),
            "role": str(row["teamposition"]),
            "build": str(row["build"]),
            "spell_pair": _spell_pair_name(int(row["spell_a"]), int(row["spell_b"])),
        }

    def rune_delta(row: dict[str, Any]) -> dict[str, Any] | None:
        key = _slot_key(row)
        base_wr, base_n = solo_prior(key)
        rune_key = (
            int(row["primary_perk_1"] or 0),
            int(row["primary_style"] or 0),
            int(row["sub_style"] or 0),
        )
        full_key = (*key, *rune_key)
        nb_key = (key[0], key[1], *rune_key)
        prior = maps["rune_full"].get(full_key)
        level = "champ-role-build-keystone-tree"
        if prior is None or int(prior["matchups"]) < MIN_LOADOUT_N:
            prior = maps["rune_nb"].get(nb_key)
            level = "champ-role-keystone-tree"
        if prior is None or int(prior["matchups"]) < MIN_LOADOUT_N:
            return None
        return {
            "level": level,
            "wr": float(prior["win_rate"]),
            "n": int(prior["matchups"]),
            "base_wr": base_wr,
            "base_n": base_n,
            "delta": float(prior["win_rate"]) - base_wr,
            "team": "blue" if int(row["teamid"]) == 100 else "red",
            "champion": champs.get(int(row["championid"]), str(row["championid"])),
            "role": str(row["teamposition"]),
            "build": str(row["build"]),
            "rune_key": {
                "primary_perk_1": int(row["primary_perk_1"] or 0),
                "primary_style": int(row["primary_style"] or 0),
                "sub_style": int(row["sub_style"] or 0),
            },
        }

    def relationship_evidence() -> dict[str, Any]:
        blue_keys = [_slot_key(row) for row in slots[:5]]
        red_keys = [_slot_key(row) for row in slots[5:]]
        matchup_items: list[dict[str, Any]] = []
        for blue_key in blue_keys:
            for red_key in red_keys:
                item = _matchup_blue_wr(blue_key, red_key, maps["full_matchup"])
                level = "champ-role-build matchup"
                if item is None or int(item["matchups"]) < MIN_FULL_N:
                    prior = maps["nb_matchup"].get((blue_key[0], blue_key[1], red_key[0], red_key[1]))
                    if prior is not None:
                        item = {
                            "blue_wr": float(prior["blue_win_rate"]),
                            "matchups": int(prior["matchups"]),
                        }
                        level = "champ-role matchup"
                if item is None or int(item["matchups"]) < MIN_REL_N:
                    prior = maps["champ_matchup"].get((blue_key[0], red_key[0]))
                    if prior is not None:
                        item = {
                            "blue_wr": float(prior["blue_win_rate"]),
                            "matchups": int(prior["matchups"]),
                        }
                        level = "champion matchup"
                if item is None or int(item["matchups"]) < MIN_REL_N:
                    continue
                actual_wr = float(item["blue_wr"]) if actual == "blue" else 1.0 - float(item["blue_wr"])
                matchup_items.append(
                    {
                        "level": level,
                        "blue_wr": float(item["blue_wr"]),
                        "actual_wr": actual_wr,
                        "edge": actual_wr - 0.5,
                        "matchups": int(item["matchups"]),
                        "blue": _key_name(blue_key, champs),
                        "red": _key_name(red_key, champs),
                        "actual_side": actual,
                    }
                )

        blue_matchup_avg = _mean([item["blue_wr"] for item in matchup_items])
        actual_matchup_avg = None
        if blue_matchup_avg is not None:
            actual_matchup_avg = blue_matchup_avg if actual == "blue" else 1.0 - blue_matchup_avg

        synergy_items: list[dict[str, Any]] = []
        for side, team_keys in (("blue", blue_keys), ("red", red_keys)):
            for i in range(5):
                for j in range(i + 1, 5):
                    item = _synergy_wr(team_keys[i], team_keys[j], maps["full_synergy"])
                    level = "champ-role-build synergy"
                    if item is None or int(item["matchups"]) < MIN_FULL_N:
                        nb_left, nb_right = _canon2(
                            (team_keys[i][0], team_keys[i][1]),
                            (team_keys[j][0], team_keys[j][1]),
                        )
                        prior = maps["nb_synergy"].get((*nb_left, *nb_right))
                        if prior is not None:
                            item = {"wr": float(prior["win_rate"]), "matchups": int(prior["matchups"])}
                            level = "champ-role synergy"
                    if item is None or int(item["matchups"]) < MIN_REL_N:
                        champ_left, champ_right = sorted((team_keys[i][0], team_keys[j][0]))
                        prior = maps["champ_synergy"].get((champ_left, champ_right))
                        if prior is not None:
                            item = {"wr": float(prior["win_rate"]), "matchups": int(prior["matchups"])}
                            level = "champion synergy"
                    if item is None or int(item["matchups"]) < MIN_REL_N:
                        continue
                    synergy_items.append(
                        {
                            "side": side,
                            "level": level,
                            "wr": float(item["wr"]),
                            "edge": float(item["wr"]) - 0.5,
                            "matchups": int(item["matchups"]),
                            "pair": [_key_name(team_keys[i], champs), _key_name(team_keys[j], champs)],
                        }
                    )

        blue_synergy_avg = _mean([item["wr"] for item in synergy_items if item["side"] == "blue"])
        red_synergy_avg = _mean([item["wr"] for item in synergy_items if item["side"] == "red"])
        synergy_edge_for_actual = None
        if blue_synergy_avg is not None and red_synergy_avg is not None:
            synergy_edge_for_actual = (
                blue_synergy_avg - red_synergy_avg
                if actual == "blue"
                else red_synergy_avg - blue_synergy_avg
            )

        components = []
        if actual_matchup_avg is not None:
            components.append(actual_matchup_avg - 0.5)
        if synergy_edge_for_actual is not None:
            components.append(synergy_edge_for_actual)

        return {
            "actual_relationship_edge": _mean(components),
            "actual_matchup_avg_wr": actual_matchup_avg,
            "blue_matchup_avg_wr": blue_matchup_avg,
            "blue_synergy_avg_wr": blue_synergy_avg,
            "red_synergy_avg_wr": red_synergy_avg,
            "actual_synergy_edge_vs_predicted": synergy_edge_for_actual,
            "top_matchups_for_actual": sorted(
                matchup_items,
                key=lambda item: (item["edge"], item["matchups"]),
                reverse=True,
            )[:3],
            "top_synergies_for_actual": sorted(
                [item for item in synergy_items if item["side"] == actual],
                key=lambda item: (item["edge"], item["matchups"]),
                reverse=True,
            )[:3],
        }

    def side_delta(items: list[dict[str, Any]], side: str) -> float | None:
        return _mean([item["delta"] for item in items if item["team"] == side])

    blue_solo = [solo_prior(_slot_key(row))[0] for row in slots[:5]]
    red_solo = [solo_prior(_slot_key(row))[0] for row in slots[5:]]
    solo_blue_edge = (_mean(blue_solo) or 0.5) - (_mean(red_solo) or 0.5)
    actual_solo_edge = solo_blue_edge if actual == "blue" else -solo_blue_edge

    relationship = relationship_evidence()
    spell_deltas = [item for item in (spell_delta(row) for row in slots) if item is not None]
    rune_deltas = [item for item in (rune_delta(row) for row in slots) if item is not None]
    blue_spell = side_delta(spell_deltas, "blue")
    red_spell = side_delta(spell_deltas, "red")
    blue_rune = side_delta(rune_deltas, "blue")
    red_rune = side_delta(rune_deltas, "red")

    spell_edge = None
    if blue_spell is not None and red_spell is not None:
        spell_edge = blue_spell - red_spell if actual == "blue" else red_spell - blue_spell
    rune_edge = None
    if blue_rune is not None and red_rune is not None:
        rune_edge = blue_rune - red_rune if actual == "blue" else red_rune - blue_rune

    patch_values: list[dict[str, Any]] = []
    for row in slots:
        key = (
            int(row["season"]),
            int(row["patch"]),
            int(row["championid"]),
            str(row["teamposition"]),
            str(row["build"]),
        )
        prior = maps["patch_solo"].get(key)
        base_wr, _ = solo_prior(_slot_key(row))
        if prior is None or int(prior["matchups"]) < 30:
            continue
        patch_values.append(
            {
                "team": "blue" if int(row["teamid"]) == 100 else "red",
                "delta": float(prior["win_rate"]) - base_wr,
                "wr": float(prior["win_rate"]),
                "n": int(prior["matchups"]),
                "champion": champs.get(int(row["championid"]), str(row["championid"])),
                "role": str(row["teamposition"]),
                "build": str(row["build"]),
            }
        )
    blue_patch = side_delta(patch_values, "blue")
    red_patch = side_delta(patch_values, "red")
    patch_edge = None
    if blue_patch is not None and red_patch is not None:
        patch_edge = blue_patch - red_patch if actual == "blue" else red_patch - blue_patch

    top_spell_actual = sorted(
        [item for item in spell_deltas if item["team"] == actual],
        key=lambda item: (item["delta"], item["n"]),
        reverse=True,
    )[:3]
    top_rune_actual = sorted(
        [item for item in rune_deltas if item["team"] == actual],
        key=lambda item: (item["delta"], item["n"]),
        reverse=True,
    )[:3]

    unaccounted: list[str] = []
    rel_edge = relationship["actual_relationship_edge"]
    if rel_edge is not None and rel_edge >= 0.015:
        unaccounted.append("exact champion matchup/synergy priors disabled")
    elif any(
        item["edge"] >= 0.075 and item["matchups"] >= 300
        for item in relationship["top_matchups_for_actual"]
    ):
        unaccounted.append("strong individual champion matchup prior disabled")
    if spell_edge is not None and spell_edge >= 0.012:
        unaccounted.append("summoner spell conditioned prior absent")
    elif any(item["delta"] >= 0.045 and item["n"] >= 250 for item in top_spell_actual):
        unaccounted.append("strong slot summoner spell prior absent")
    if rune_edge is not None and rune_edge >= 0.012:
        unaccounted.append("rune/keystone conditioned prior absent")
    elif any(item["delta"] >= 0.04 and item["n"] >= 250 for item in top_rune_actual):
        unaccounted.append("strong slot rune/keystone prior absent")
    if patch_edge is not None and patch_edge >= 0.012:
        unaccounted.append("patch-conditioned champion/build prior absent")

    patch_label = f"S{int(slots[0]['season'])}.{int(slots[0]['patch'])}"
    return {
        "matchid": cand["matchid"],
        "split": cand["split"],
        "batch_position": cand["batch_position"],
        "pred_blue_win": cand["pred_blue_win"],
        "predicted_side_0_5": cand["predicted_side"],
        "predicted_side_threshold_0_516": predicted_threshold,
        "actual_winner": actual,
        "threshold_correct": predicted_threshold == actual,
        "patch": patch_label,
        "blue_p1_mean_from_candidate": cand.get("blue_p1_mean"),
        "red_p1_mean_from_candidate": cand.get("red_p1_mean"),
        "solo_blue_edge_train_table": solo_blue_edge,
        "actual_solo_edge_train_table": actual_solo_edge,
        "relationship": relationship,
        "spell_edge_for_actual": spell_edge,
        "top_spell_actual": top_spell_actual,
        "rune_edge_for_actual": rune_edge,
        "top_rune_actual": top_rune_actual,
        "patch_edge_for_actual_train_overlap": patch_edge,
        "top_patch_actual": sorted(
            [item for item in patch_values if item["team"] == actual],
            key=lambda item: (item["delta"], item["n"]),
            reverse=True,
        )[:3],
        "unaccounted_influences": unaccounted,
    }


def analyze_candidate_file(candidate_path: Path, output_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    data = json.loads(candidate_path.read_text(encoding="utf-8"))
    batch = data["batch"]
    matchids = [row["matchid"] for row in batch]
    client = get_client()
    champs = _load_champions(Path("database/clickhouse/support/championid_name_map.jsonl"))
    participant_rows = _load_batch_participants(client, matchids)
    if len(participant_rows) != 10 * len(matchids):
        raise RuntimeError(
            f"Expected {10 * len(matchids)} participant rows, got {len(participant_rows)}"
        )
    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in participant_rows:
        games[str(row["matchid"])].append(row)
    for rows in games.values():
        rows.sort(key=lambda item: int(item["participantid"]))

    keys = _build_key_sets(games)
    maps = _load_prior_maps(client, keys)
    prior_lookup_counts = {name: len(value) for name, value in maps.items()}

    reason_counter: Counter[str] = Counter()
    unaccounted_counter: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    for cand in batch:
        result = _analyze_game(cand, games[str(cand["matchid"])], maps, champs)
        results.append(result)
        if result["threshold_correct"]:
            reason_counter["Tuned 0.516 decision threshold fixes central blue-overcall/undercall"] += 1
        else:
            reason_counter[
                "Still wrong after tuned threshold; model score is central and available champion/build priors are near-tied or outweighed"
            ] += 1
        if result["patch"] in {"S16.9", "S16.10"}:
            reason_counter[
                "Patch/date is not an input; S16.9-S16.10 holdout has no same-patch train solo prior"
            ] += 1
        for label in result["unaccounted_influences"]:
            unaccounted_counter[label] += 1

    def rank_score(row: dict[str, Any]) -> float:
        rel = row["relationship"]["actual_relationship_edge"] or 0.0
        spell = row["spell_edge_for_actual"] or 0.0
        rune = row["rune_edge_for_actual"] or 0.0
        patch = row["patch_edge_for_actual_train_overlap"] or 0.0
        return len(row["unaccounted_influences"]) * 10.0 + rel + spell + rune + patch

    representative = sorted(
        [row for row in results if row["unaccounted_influences"]],
        key=rank_score,
        reverse=True,
    )[:20]
    output = {
        "source": {
            "candidate_path": str(candidate_path),
            "model_path": data.get("model_path"),
            "selection": data.get("selection"),
            "batch_number": data.get("batch_number"),
            "batch_size": data.get("batch_size"),
            "band": data.get("band"),
            "model_decision_threshold_used_for_accounted_check": MODEL_THRESHOLD,
            "identity_policy": (
                "No player identity fields are selected, emitted, aggregated, or used as evidence. "
                "PUUID appears only inside the ClickHouse rune join key and is not materialized."
            ),
        },
        "central_band_summary": data.get("summary"),
        "prior_lookup_counts": prior_lookup_counts,
        "batch_summary": {
            "games": len(results),
            "wrong_at_0_5_by_selection": sum(
                1 for row in results if row["predicted_side_0_5"] != row["actual_winner"]
            ),
            "correct_at_threshold_0_516": sum(1 for row in results if row["threshold_correct"]),
            "still_wrong_at_threshold_0_516": sum(
                1 for row in results if not row["threshold_correct"]
            ),
            "unaccounted_games": sum(1 for row in results if row["unaccounted_influences"]),
            "patch_counts": dict(Counter(row["patch"] for row in results)),
            "reason_counts": dict(reason_counter),
            "unaccounted_counts": dict(unaccounted_counter),
            "elapsed_seconds": time.monotonic() - started,
        },
        "representative_unaccounted_games": representative,
        "all_games": results,
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = analyze_candidate_file(args.candidates, args.output)
    print(json.dumps(output["batch_summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
