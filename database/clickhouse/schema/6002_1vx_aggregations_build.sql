-- noqa: disable=AL09,LT02,LT05,RF02,RF03,ST09
-- Same-team singleton synergy priors. Each valid 5-player team contributes 5
-- singleton rows; enemies are intentionally ignored.
-- Built from ml_game_player_pivot so participant/item labels are joined and
-- role-pivoted once for all matchup aggregate builders.

TRUNCATE TABLE game_data_filtered.synergy_1vx;

INSERT INTO game_data_filtered.synergy_1vx
WITH
train_games AS (
    SELECT
        split,
        blue_players,
        red_players,
        blue_win
    FROM game_data_filtered.ml_game_player_pivot
    WHERE split = 'train'
),

single_expanded AS (
    SELECT
        split,
        tupleElement(player_info, 2) AS team_win,
        tupleElement(player_info, 1) AS player
    FROM train_games
    ARRAY JOIN arrayConcat(
        arrayMap(
            player -> (player, blue_win),
            blue_players
        ),
        arrayMap(
            player -> (player, toUInt8(1 - blue_win)),
            red_players
        )
    ) AS player_info
)

SELECT
    split,
    tupleElement(player, 1) AS championid,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(championid), '') AS championname,
    tupleElement(player, 2) AS teamposition,
    tupleElement(player, 3) AS build,
    count() AS matchups,
    sum(team_win) AS wins,
    matchups - wins AS losses,
    toFloat32(wins / matchups) AS win_rate
FROM single_expanded
GROUP BY
    split,
    championid, teamposition, build;
