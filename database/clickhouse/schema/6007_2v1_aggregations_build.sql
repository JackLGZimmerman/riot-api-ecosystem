-- noqa: disable=AL09,LT02,LT05,RF02,RF03,ST09
-- 2v1 cross-team matchup win rates: every C(5,2) intra-team pair on one
-- side crossed with every player on the other side. Each match contributes
-- both directions (lo-pair vs hi-single and hi-pair vs lo-single).
-- Players within the pair are sorted; pair vs single is intentionally
-- asymmetric so no left/right canonicalization is needed.
-- pair_win_rate + single_win_rate = 1.0.
-- Built from ml_game_player_pivot so participant/item labels are joined and
-- role-pivoted once for all matchup aggregate builders.
-- Leakage-safe training prior: only train games contribute outcome counts.
-- Test rows should join against split = 'train'; train feature rows
-- must subtract their current match contribution at feature-build time.

TRUNCATE TABLE game_data_filtered.matchup_2v1;

INSERT INTO game_data_filtered.matchup_2v1
WITH
[(1, 2), (1, 3), (1, 4), (1, 5),
 (2, 3), (2, 4), (2, 5),
 (3, 4), (3, 5),
 (4, 5)] AS combos2_idx,

match_pairs_singles AS (
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
        ) AS red_pairs,
        p.blue_players AS blue_singles,
        p.red_players AS red_singles
    FROM game_data_filtered.ml_game_player_pivot AS p
    WHERE p.split = 'train'
),

both_directions AS (
    SELECT
        split,
        arraySort(tupleElement(entry, 1)) AS pair_arr,
        tupleElement(entry, 2) AS single,
        tupleElement(entry, 3) AS pair_win,
        tupleElement(entry, 4) AS single_win
    FROM match_pairs_singles
    ARRAY JOIN arrayConcat(
        arrayFlatten(arrayMap(
            bp -> arrayMap(
                s -> (bp, s, blue_win, toUInt8(1 - blue_win)),
                red_singles
            ),
            blue_pairs
        )),
        arrayFlatten(arrayMap(
            rp -> arrayMap(
                s -> (rp, s, toUInt8(1 - blue_win), blue_win),
                blue_singles
            ),
            red_pairs
        ))
    ) AS entry
)

SELECT
    split,
    tupleElement(pair_arr[1], 1) AS pair_championid_1,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(pair_championid_1), '') AS pair_championname_1,
    tupleElement(pair_arr[1], 2) AS pair_teamposition_1,
    tupleElement(pair_arr[1], 3) AS pair_build_1,
    tupleElement(pair_arr[2], 1) AS pair_championid_2,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(pair_championid_2), '') AS pair_championname_2,
    tupleElement(pair_arr[2], 2) AS pair_teamposition_2,
    tupleElement(pair_arr[2], 3) AS pair_build_2,
    tupleElement(single, 1) AS single_championid,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(single_championid), '') AS single_championname,
    tupleElement(single, 2) AS single_teamposition,
    tupleElement(single, 3) AS single_build,
    count() AS matchups,
    sum(pair_win) AS pair_wins,
    sum(single_win) AS single_wins,
    toFloat32(pair_wins / matchups) AS pair_win_rate,
    toFloat32(single_wins / matchups) AS single_win_rate
FROM both_directions
GROUP BY
    split,
    pair_championid_1, pair_teamposition_1, pair_build_1,
    pair_championid_2, pair_teamposition_2, pair_build_2,
    single_championid, single_teamposition, single_build;
