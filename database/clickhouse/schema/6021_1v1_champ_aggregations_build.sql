-- noqa: disable=AL05,LT01,LT02,LT05,RF02,ST09
--
-- Coarsest champion-pair backoff for 6000: 1v1 matchup win rates keyed on
-- championid only (lane-agnostic). 25 (blue, red) pairs per game; value is the
-- blue-perspective win rate. Stored directionally (no canonicalisation): the
-- cache builder reads blue_win_rate straight. Leakage-safe: train only.

TRUNCATE TABLE game_data_filtered.matchup_1v1_champ;

INSERT INTO game_data_filtered.matchup_1v1_champ
SELECT
    'train' AS split,
    tupleElement(pair.1, 1) AS blue_championid,
    tupleElement(pair.2, 1) AS red_championid,
    count() AS matchups,
    toFloat32(sum(pair.3) / count()) AS blue_win_rate
FROM game_data_filtered.ml_game_player_pivot AS p
ARRAY JOIN arrayFlatten(arrayMap(
    b -> arrayMap(
        r -> (b, r, p.blue_win),
        p.red_players
    ),
    p.blue_players
)) AS pair
WHERE p.split = 'train'
GROUP BY blue_championid, red_championid;
