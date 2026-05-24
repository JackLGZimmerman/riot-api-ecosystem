-- noqa: disable=AL09,LT02,LT05,RF02,RF03,ST09
--
-- Same-team pair (2vx) synergy priors. Each valid 5-player team contributes
-- C(5, 2) = 10 pair rows; enemies are intentionally ignored.
-- Mirrors the structure of 6003_1vx_aggregations_build.sql: only the
-- (key, count, win_rate) priors are emitted; scaling-bin, time, and
-- profile-metric columns are intentionally excluded.
--
-- Canonicalised so the smaller (championid, teamposition, build) tuple is
-- in slot 1. Leakage-safe: only train games contribute outcome counts.

TRUNCATE TABLE game_data_filtered.synergy_2vx;

INSERT INTO game_data_filtered.synergy_2vx
SELECT
    'train' AS split,
    tupleElement(p1, 1) AS championid_1,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(championid_1),
        ''
    ) AS championname_1,
    tupleElement(p1, 2) AS teamposition_1,
    tupleElement(p1, 3) AS build_1,
    tupleElement(p2, 1) AS championid_2,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(championid_2),
        ''
    ) AS championname_2,
    tupleElement(p2, 2) AS teamposition_2,
    tupleElement(p2, 3) AS build_2,
    count() AS matchups,
    sum(team_win) AS wins,
    matchups - wins AS losses,
    toFloat32(wins / matchups) AS win_rate
FROM (
    SELECT
        if(pair.1 <= pair.2, pair.1, pair.2) AS p1,
        if(pair.1 <= pair.2, pair.2, pair.1) AS p2,
        pair.3 AS team_win
    FROM game_data_filtered.ml_game_player_pivot
    ARRAY JOIN [
        (blue_players[1], blue_players[2], blue_win),
        (blue_players[1], blue_players[3], blue_win),
        (blue_players[1], blue_players[4], blue_win),
        (blue_players[1], blue_players[5], blue_win),
        (blue_players[2], blue_players[3], blue_win),
        (blue_players[2], blue_players[4], blue_win),
        (blue_players[2], blue_players[5], blue_win),
        (blue_players[3], blue_players[4], blue_win),
        (blue_players[3], blue_players[5], blue_win),
        (blue_players[4], blue_players[5], blue_win),
        (red_players[1], red_players[2], toUInt8(1 - blue_win)),
        (red_players[1], red_players[3], toUInt8(1 - blue_win)),
        (red_players[1], red_players[4], toUInt8(1 - blue_win)),
        (red_players[1], red_players[5], toUInt8(1 - blue_win)),
        (red_players[2], red_players[3], toUInt8(1 - blue_win)),
        (red_players[2], red_players[4], toUInt8(1 - blue_win)),
        (red_players[2], red_players[5], toUInt8(1 - blue_win)),
        (red_players[3], red_players[4], toUInt8(1 - blue_win)),
        (red_players[3], red_players[5], toUInt8(1 - blue_win)),
        (red_players[4], red_players[5], toUInt8(1 - blue_win))
    ] AS pair
    WHERE split = 'train'
)
GROUP BY
    championid_1, teamposition_1, build_1,
    championid_2, teamposition_2, build_2;
