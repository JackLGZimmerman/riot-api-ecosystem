-- noqa: disable=AL05,LT01,LT02,LT05,RF02
--
-- Train-split per-player game count and win rate. The cache builder applies
-- leave-one-out adjustment on train rows and Empirical Bayes smoothing, so
-- this emits only the raw counts.

TRUNCATE TABLE game_data_filtered.player_1vx;

INSERT INTO game_data_filtered.player_1vx
(
    split,
    puuid,
    matchups,
    win_rate
)
SELECT
    'train' AS split,
    toString(ps.puuid) AS puuid,
    toUInt32(count()) AS matchups,
    toFloat32(countIf(ps.win > 0) / count()) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS s
    ON ps.matchid = s.matchid
WHERE
    s.split = 'train'
GROUP BY
    puuid;
