-- noqa: disable=AL09,LT02,LT05,RF02,RF03,ST09
-- Same-team trio synergy priors. Each valid 5-player team contributes 10
-- trio rows; enemies are intentionally ignored.
-- Built from ml_game_player_pivot so participant/item labels are joined and
-- role-pivoted once for all matchup aggregate builders.

TRUNCATE TABLE game_data_filtered.synergy_3vx;

INSERT INTO game_data_filtered.synergy_3vx
WITH
[(1, 2, 3), (1, 2, 4), (1, 2, 5),
 (1, 3, 4), (1, 3, 5), (1, 4, 5),
 (2, 3, 4), (2, 3, 5), (2, 4, 5),
 (3, 4, 5)] AS combos3_idx,

train_games AS (
    SELECT
        split,
        blue_players,
        red_players,
        blue_win
    FROM game_data_filtered.ml_game_player_pivot
    WHERE split = 'train'
),

trio_expanded AS (
    SELECT
        split,
        tupleElement(trio_info, 2) AS team_win,
        arraySort(tupleElement(trio_info, 1)) AS trio_combo
    FROM train_games
    ARRAY JOIN arrayConcat(
        arrayMap(
            idx -> (
                [
                    blue_players[tupleElement(idx, 1)],
                    blue_players[tupleElement(idx, 2)],
                    blue_players[tupleElement(idx, 3)]
                ],
                blue_win
            ),
            combos3_idx
        ),
        arrayMap(
            idx -> (
                [
                    red_players[tupleElement(idx, 1)],
                    red_players[tupleElement(idx, 2)],
                    red_players[tupleElement(idx, 3)]
                ],
                toUInt8(1 - blue_win)
            ),
            combos3_idx
        )
    ) AS trio_info
)

SELECT
    split,
    tupleElement(trio_combo[1], 1) AS championid_1,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(championid_1), '') AS championname_1,
    tupleElement(trio_combo[1], 2) AS teamposition_1,
    tupleElement(trio_combo[1], 3) AS build_1,
    tupleElement(trio_combo[2], 1) AS championid_2,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(championid_2), '') AS championname_2,
    tupleElement(trio_combo[2], 2) AS teamposition_2,
    tupleElement(trio_combo[2], 3) AS build_2,
    tupleElement(trio_combo[3], 1) AS championid_3,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(championid_3), '') AS championname_3,
    tupleElement(trio_combo[3], 2) AS teamposition_3,
    tupleElement(trio_combo[3], 3) AS build_3,
    count() AS matchups,
    sum(team_win) AS wins,
    matchups - wins AS losses,
    toFloat32(wins / matchups) AS win_rate
FROM trio_expanded
GROUP BY
    split,
    championid_1, teamposition_1, build_1,
    championid_2, teamposition_2, build_2,
    championid_3, teamposition_3, build_3;
