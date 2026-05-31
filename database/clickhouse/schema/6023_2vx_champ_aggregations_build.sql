-- noqa: disable=AL05,LT01,LT02,LT05,RF02,RF03,ST09
--
-- Coarsest champion-pair backoff for 6004: same-team 2vx synergy win rates
-- keyed on championid only. Each valid team contributes C(5,2)=10 pairs; the
-- championid pair is canonicalised smaller-first. Leakage-safe: train only.

TRUNCATE TABLE game_data_filtered.synergy_2vx_champ;

INSERT INTO game_data_filtered.synergy_2vx_champ
SELECT
    'train' AS split,
    least(c1, c2) AS championid_1,
    greatest(c1, c2) AS championid_2,
    count() AS matchups,
    toFloat32(sum(team_win) / count()) AS win_rate
FROM (
    SELECT
        tupleElement(pair.1, 1) AS c1,
        tupleElement(pair.2, 1) AS c2,
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
GROUP BY championid_1, championid_2;
