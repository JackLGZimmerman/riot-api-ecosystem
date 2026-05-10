-- noqa: disable=AL09,LT02,LT05,RF02,RF03,ST09
-- Same-team pair synergy priors. Each valid 5-player team contributes 10
-- pair rows; enemies are intentionally ignored.
-- Built from ml_game_player_pivot so participant/item labels are joined and
-- role-pivoted once for all matchup aggregate builders.

TRUNCATE TABLE game_data_filtered.synergy_2vx;

INSERT INTO game_data_filtered.synergy_2vx
WITH
[(1, 2), (1, 3), (1, 4), (1, 5),
 (2, 3), (2, 4), (2, 5),
 (3, 4), (3, 5),
 (4, 5)] AS combos2_idx,

train_games AS (
    SELECT
        split,
        blue_players,
        red_players,
        blue_win
    FROM game_data_filtered.ml_game_player_pivot
    WHERE split = 'train'
),

pair_expanded AS (
    SELECT
        split,
        tupleElement(pair_info, 2) AS team_win,
        arraySort(tupleElement(pair_info, 1)) AS pair_combo
    FROM train_games
    ARRAY JOIN arrayConcat(
        arrayMap(
            idx -> (
                [
                    blue_players[tupleElement(idx, 1)],
                    blue_players[tupleElement(idx, 2)]
                ],
                blue_win
            ),
            combos2_idx
        ),
        arrayMap(
            idx -> (
                [
                    red_players[tupleElement(idx, 1)],
                    red_players[tupleElement(idx, 2)]
                ],
                toUInt8(1 - blue_win)
            ),
            combos2_idx
        )
    ) AS pair_info
)

SELECT
    split,
    tupleElement(pair_combo[1], 1) AS championid_1,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(championid_1), '') AS championname_1,
    tupleElement(pair_combo[1], 2) AS teamposition_1,
    tupleElement(pair_combo[1], 3) AS build_1,
    tupleElement(pair_combo[2], 1) AS championid_2,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(championid_2), '') AS championname_2,
    tupleElement(pair_combo[2], 2) AS teamposition_2,
    tupleElement(pair_combo[2], 3) AS build_2,
    count() AS matchups,
    sum(team_win) AS wins,
    matchups - wins AS losses,
    toFloat32(wins / matchups) AS win_rate
FROM pair_expanded
GROUP BY
    split,
    championid_1, teamposition_1, build_1,
    championid_2, teamposition_2, build_2;
