-- noqa: disable=AL09,LT02,LT05,RF02,RF03,ST09
-- 3v2 cross-team matchup win rates: every C(5,3) intra-team trio on one
-- side crossed with every C(5,2) pair on the other side. Each match
-- contributes both directions (blue-trio vs red-pair and red-trio vs
-- blue-pair). Players within the trio/pair are sorted; 3v2 is intentionally
-- asymmetric so no left/right canonicalization is needed.
-- trio_win_rate + pair_win_rate = 1.0.
-- Built from ml_game_player_pivot so participant/item labels are joined and
-- role-pivoted once for all matchup aggregate builders.
-- Leakage-safe training prior: only train games contribute outcome counts.

TRUNCATE TABLE game_data_filtered.matchup_3v2;

INSERT INTO game_data_filtered.matchup_3v2
WITH
[(1, 2), (1, 3), (1, 4), (1, 5),
 (2, 3), (2, 4), (2, 5),
 (3, 4), (3, 5),
 (4, 5)] AS combos2_idx,
[(1, 2, 3), (1, 2, 4), (1, 2, 5),
 (1, 3, 4), (1, 3, 5), (1, 4, 5),
 (2, 3, 4), (2, 3, 5), (2, 4, 5),
 (3, 4, 5)] AS combos3_idx,

match_trios_pairs AS (
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
        ) AS blue_trios,
        arrayMap(
            idx -> [
                p.red_players[tupleElement(idx, 1)],
                p.red_players[tupleElement(idx, 2)],
                p.red_players[tupleElement(idx, 3)]
            ],
            combos3_idx
        ) AS red_trios,
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

both_directions AS (
    SELECT
        split,
        arraySort(tupleElement(entry, 1)) AS trio_arr,
        arraySort(tupleElement(entry, 2)) AS pair_arr,
        tupleElement(entry, 3) AS trio_win,
        tupleElement(entry, 4) AS pair_win
    FROM match_trios_pairs
    ARRAY JOIN arrayConcat(
        arrayFlatten(arrayMap(
            bt -> arrayMap(
                rp -> (bt, rp, blue_win, toUInt8(1 - blue_win)),
                red_pairs
            ),
            blue_trios
        )),
        arrayFlatten(arrayMap(
            rt -> arrayMap(
                bp -> (rt, bp, toUInt8(1 - blue_win), blue_win),
                blue_pairs
            ),
            red_trios
        ))
    ) AS entry
)

SELECT
    split,
    tupleElement(trio_arr[1], 1) AS trio_championid_1,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(trio_championid_1), '') AS trio_championname_1,
    tupleElement(trio_arr[1], 2) AS trio_teamposition_1,
    tupleElement(trio_arr[1], 3) AS trio_build_1,
    tupleElement(trio_arr[2], 1) AS trio_championid_2,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(trio_championid_2), '') AS trio_championname_2,
    tupleElement(trio_arr[2], 2) AS trio_teamposition_2,
    tupleElement(trio_arr[2], 3) AS trio_build_2,
    tupleElement(trio_arr[3], 1) AS trio_championid_3,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(trio_championid_3), '') AS trio_championname_3,
    tupleElement(trio_arr[3], 2) AS trio_teamposition_3,
    tupleElement(trio_arr[3], 3) AS trio_build_3,
    tupleElement(pair_arr[1], 1) AS pair_championid_1,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(pair_championid_1), '') AS pair_championname_1,
    tupleElement(pair_arr[1], 2) AS pair_teamposition_1,
    tupleElement(pair_arr[1], 3) AS pair_build_1,
    tupleElement(pair_arr[2], 1) AS pair_championid_2,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(pair_championid_2), '') AS pair_championname_2,
    tupleElement(pair_arr[2], 2) AS pair_teamposition_2,
    tupleElement(pair_arr[2], 3) AS pair_build_2,
    count() AS matchups,
    sum(trio_win) AS trio_wins,
    sum(pair_win) AS pair_wins,
    toFloat32(trio_wins / matchups) AS trio_win_rate,
    toFloat32(pair_wins / matchups) AS pair_win_rate
FROM both_directions
GROUP BY
    split,
    trio_championid_1, trio_teamposition_1, trio_build_1,
    trio_championid_2, trio_teamposition_2, trio_build_2,
    trio_championid_3, trio_teamposition_3, trio_build_3,
    pair_championid_1, pair_teamposition_1, pair_build_1,
    pair_championid_2, pair_teamposition_2, pair_build_2;
