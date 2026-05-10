-- noqa: disable=AL09,LT02,LT05,RF02,RF03,ST09
-- 3v1 cross-team matchup win rates: every C(5,3) intra-team trio on one
-- side crossed with every player on the other side. Each match contributes
-- both directions (lo-trio vs hi-single and hi-trio vs lo-single).
-- Players within the trio are sorted; trio vs single is intentionally
-- asymmetric so no left/right canonicalization is needed.
-- trio_win_rate + single_win_rate = 1.0.
-- Built from ml_game_player_pivot so participant/item labels are joined and
-- role-pivoted once for all matchup aggregate builders.
-- Leakage-safe training prior: only train games contribute outcome counts.

TRUNCATE TABLE game_data_filtered.matchup_3v1;

INSERT INTO game_data_filtered.matchup_3v1
WITH
[(1, 2, 3), (1, 2, 4), (1, 2, 5),
 (1, 3, 4), (1, 3, 5), (1, 4, 5),
 (2, 3, 4), (2, 3, 5), (2, 4, 5),
 (3, 4, 5)] AS combos3_idx,

match_trios_singles AS (
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
        p.blue_players AS blue_singles,
        p.red_players AS red_singles
    FROM game_data_filtered.ml_game_player_pivot AS p
    WHERE p.split = 'train'
),

both_directions AS (
    SELECT
        split,
        arraySort(tupleElement(entry, 1)) AS trio_arr,
        tupleElement(entry, 2) AS single,
        tupleElement(entry, 3) AS trio_win,
        tupleElement(entry, 4) AS single_win
    FROM match_trios_singles
    ARRAY JOIN arrayConcat(
        arrayFlatten(arrayMap(
            bt -> arrayMap(
                s -> (bt, s, blue_win, toUInt8(1 - blue_win)),
                red_singles
            ),
            blue_trios
        )),
        arrayFlatten(arrayMap(
            rt -> arrayMap(
                s -> (rt, s, toUInt8(1 - blue_win), blue_win),
                blue_singles
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
    tupleElement(single, 1) AS single_championid,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(single_championid), '') AS single_championname,
    tupleElement(single, 2) AS single_teamposition,
    tupleElement(single, 3) AS single_build,
    count() AS matchups,
    sum(trio_win) AS trio_wins,
    sum(single_win) AS single_wins,
    toFloat32(trio_wins / matchups) AS trio_win_rate,
    toFloat32(single_wins / matchups) AS single_win_rate
FROM both_directions
GROUP BY
    split,
    trio_championid_1, trio_teamposition_1, trio_build_1,
    trio_championid_2, trio_teamposition_2, trio_build_2,
    trio_championid_3, trio_teamposition_3, trio_build_3,
    single_championid, single_teamposition, single_build;
