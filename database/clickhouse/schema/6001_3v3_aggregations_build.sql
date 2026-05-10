-- noqa: disable=AL09,LT02,LT05,RF02,RF03,ST09
-- 3v3 cross-team matchup win rates over all C(5,3) intra-team triples.
-- Players within each team are sorted by (championid, teamposition, build);
-- the smaller team-tuple is canonicalised onto the left.
-- left_win_rate + right_win_rate = 1.0.
-- Built from ml_game_player_pivot so participant/item labels are joined and
-- role-pivoted once for all matchup aggregate builders.
-- Leakage-safe training prior: only train games contribute outcome counts.
-- Validation/test rows should join against split = 'train'; train feature rows
-- must subtract their current match contribution at feature-build time.

TRUNCATE TABLE game_data_filtered.matchup_3v3;

INSERT INTO game_data_filtered.matchup_3v3
WITH
[(1, 2, 3), (1, 2, 4), (1, 2, 5),
 (1, 3, 4), (1, 3, 5), (1, 4, 5),
 (2, 3, 4), (2, 3, 5), (2, 4, 5),
 (3, 4, 5)] AS combos3_idx,

match_combos AS (
    SELECT
        p.split AS split,
        p.blue_win AS blue_win,
        arrayMap(
            idx -> [
                p.blue_players[tupleElement(idx, 1)],
                p.blue_players[tupleElement(idx, 2)],
                p.blue_players[tupleElement(idx, 3)]
            ],
            combos3_idx
        ) AS blue_triples,
        arrayMap(
            idx -> [
                p.red_players[tupleElement(idx, 1)],
                p.red_players[tupleElement(idx, 2)],
                p.red_players[tupleElement(idx, 3)]
            ],
            combos3_idx
        ) AS red_triples
    FROM game_data_filtered.ml_game_player_pivot AS p
    WHERE p.split = 'train'
),

cross_pairs AS (
    SELECT
        split,
        arraySort(tupleElement(cp, 1)) AS blue_triple,
        arraySort(tupleElement(cp, 2)) AS red_triple,
        blue_win
    FROM match_combos
    ARRAY JOIN arrayFlatten(arrayMap(
        bt -> arrayMap(rt -> (bt, rt), red_triples),
        blue_triples
    )) AS cp
),

canonical AS (
    SELECT
        split,
        blue_triple <= red_triple AS blue_is_left,
        if(blue_is_left, blue_triple, red_triple) AS left_triple,
        if(blue_is_left, red_triple, blue_triple) AS right_triple,
        if(blue_is_left, blue_win, toUInt8(1 - blue_win)) AS left_win,
        if(blue_is_left, toUInt8(1 - blue_win), blue_win) AS right_win
    FROM cross_pairs
)

SELECT
    split,
    tupleElement(left_triple[1], 1) AS left_championid_1,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(left_championid_1), '') AS left_championname_1,
    tupleElement(left_triple[1], 2) AS left_teamposition_1,
    tupleElement(left_triple[1], 3) AS left_build_1,
    tupleElement(left_triple[2], 1) AS left_championid_2,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(left_championid_2), '') AS left_championname_2,
    tupleElement(left_triple[2], 2) AS left_teamposition_2,
    tupleElement(left_triple[2], 3) AS left_build_2,
    tupleElement(left_triple[3], 1) AS left_championid_3,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(left_championid_3), '') AS left_championname_3,
    tupleElement(left_triple[3], 2) AS left_teamposition_3,
    tupleElement(left_triple[3], 3) AS left_build_3,
    tupleElement(right_triple[1], 1) AS right_championid_1,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(right_championid_1), '') AS right_championname_1,
    tupleElement(right_triple[1], 2) AS right_teamposition_1,
    tupleElement(right_triple[1], 3) AS right_build_1,
    tupleElement(right_triple[2], 1) AS right_championid_2,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(right_championid_2), '') AS right_championname_2,
    tupleElement(right_triple[2], 2) AS right_teamposition_2,
    tupleElement(right_triple[2], 3) AS right_build_2,
    tupleElement(right_triple[3], 1) AS right_championid_3,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(right_championid_3), '') AS right_championname_3,
    tupleElement(right_triple[3], 2) AS right_teamposition_3,
    tupleElement(right_triple[3], 3) AS right_build_3,
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
    left_championid_3, left_teamposition_3, left_build_3,
    right_championid_1, right_teamposition_1, right_build_1,
    right_championid_2, right_teamposition_2, right_build_2,
    right_championid_3, right_teamposition_3, right_build_3;
