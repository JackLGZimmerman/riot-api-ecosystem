-- noqa: disable=AL09,LT02,LT05,RF02,RF03,ST09
-- 2v2 cross-team matchup win rates over all C(5,2) intra-team pairs.
-- Players within each team are sorted by (championid, teamposition, build);
-- the smaller team-tuple is canonicalised onto the left.
-- left_win_rate + right_win_rate = 1.0.
-- Built from ml_game_player_pivot so participant/item labels are joined and
-- role-pivoted once for all matchup aggregate builders.
-- Leakage-safe training prior: only train games contribute outcome counts.
-- Validation/test rows should join against split = 'train'; train feature rows
-- must subtract their current match contribution at feature-build time.

TRUNCATE TABLE game_data_filtered.matchup_2v2;

INSERT INTO game_data_filtered.matchup_2v2
WITH
[(1, 2), (1, 3), (1, 4), (1, 5),
 (2, 3), (2, 4), (2, 5),
 (3, 4), (3, 5),
 (4, 5)] AS combos2_idx,

match_combos AS (
    SELECT
        p.split AS split,
        p.blue_win AS blue_win,
        arrayMap(
            idx -> [
                p.blue_players[tupleElement(idx, 1)],
                p.blue_players[tupleElement(idx, 2)]
            ],
            combos2_idx
        ) AS blue_pairs,
        arrayMap(
            idx -> [
                p.red_players[tupleElement(idx, 1)],
                p.red_players[tupleElement(idx, 2)]
            ],
            combos2_idx
        ) AS red_pairs
    FROM game_data_filtered.ml_game_player_pivot AS p
    WHERE p.split = 'train'
),

cross_pairs AS (
    SELECT
        split,
        arraySort(tupleElement(cp, 1)) AS blue_pair,
        arraySort(tupleElement(cp, 2)) AS red_pair,
        blue_win
    FROM match_combos
    ARRAY JOIN arrayFlatten(arrayMap(
        bp -> arrayMap(rp -> (bp, rp), red_pairs),
        blue_pairs
    )) AS cp
),

canonical AS (
    SELECT
        split,
        blue_pair <= red_pair AS blue_is_left,
        if(blue_is_left, blue_pair, red_pair) AS left_pair,
        if(blue_is_left, red_pair, blue_pair) AS right_pair,
        if(blue_is_left, blue_win, toUInt8(1 - blue_win)) AS left_win,
        if(blue_is_left, toUInt8(1 - blue_win), blue_win) AS right_win
    FROM cross_pairs
)

SELECT
    split,
    tupleElement(left_pair[1], 1) AS left_championid_1,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(left_championid_1), '') AS left_championname_1,
    tupleElement(left_pair[1], 2) AS left_teamposition_1,
    tupleElement(left_pair[1], 3) AS left_build_1,
    tupleElement(left_pair[2], 1) AS left_championid_2,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(left_championid_2), '') AS left_championname_2,
    tupleElement(left_pair[2], 2) AS left_teamposition_2,
    tupleElement(left_pair[2], 3) AS left_build_2,
    tupleElement(right_pair[1], 1) AS right_championid_1,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(right_championid_1), '') AS right_championname_1,
    tupleElement(right_pair[1], 2) AS right_teamposition_1,
    tupleElement(right_pair[1], 3) AS right_build_1,
    tupleElement(right_pair[2], 1) AS right_championid_2,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(right_championid_2), '') AS right_championname_2,
    tupleElement(right_pair[2], 2) AS right_teamposition_2,
    tupleElement(right_pair[2], 3) AS right_build_2,
    count() AS matchups,
    sum(left_win) AS left_wins,
    sum(right_win) AS right_wins,
    toFloat32(left_wins / matchups) AS left_win_rate,
    toFloat32(right_wins / matchups) AS right_win_rate
FROM canonical
GROUP BY
    split,
    left_championid_1, left_teamposition_1, left_build_1,
    left_championid_2, left_teamposition_2, left_build_2,
    right_championid_1, right_teamposition_1, right_build_1,
    right_championid_2, right_teamposition_2, right_build_2;
