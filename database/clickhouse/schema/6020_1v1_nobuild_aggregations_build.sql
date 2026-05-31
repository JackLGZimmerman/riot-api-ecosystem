-- noqa: disable=AL05,LT01,LT02,LT05,RF02,ST09
--
-- Backoff level for 6000: 1v1 matchup win rates with build dropped from both
-- members. Each ml_game_player_pivot row fans out into 25 (blue, red) pairs;
-- the value stored is the blue-perspective win rate for that (champ, role) pair.
-- Stored directionally (no canonicalisation): the cache builder reads
-- blue_win_rate straight, no orientation flip. Leakage-safe: train only.

TRUNCATE TABLE game_data_filtered.matchup_1v1_nobuild;

INSERT INTO game_data_filtered.matchup_1v1_nobuild
SELECT
    'train' AS split,
    tupleElement(pair.1, 1) AS blue_championid,
    tupleElement(pair.1, 2) AS blue_teamposition,
    tupleElement(pair.2, 1) AS red_championid,
    tupleElement(pair.2, 2) AS red_teamposition,
    count() AS matchups,
    toFloat32(sum(pair.3) / count()) AS blue_win_rate
FROM game_data_filtered.ml_game_player_pivot AS p
ARRAY JOIN arrayFlatten(arrayMap(
    b -> arrayMap(
        r -> (
            (tupleElement(b, 1), tupleElement(b, 2)),
            (tupleElement(r, 1), tupleElement(r, 2)),
            p.blue_win
        ),
        p.red_players
    ),
    p.blue_players
)) AS pair
WHERE p.split = 'train'
GROUP BY
    blue_championid, blue_teamposition,
    red_championid, red_teamposition;
