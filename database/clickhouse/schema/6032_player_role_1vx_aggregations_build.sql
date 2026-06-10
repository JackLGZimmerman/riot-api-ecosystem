-- noqa: disable=AL05,LT01,LT02,LT05,RF02
--
-- Train-split per-(player, role) game count. The cache builder applies
-- leave-one-out adjustment (count minus one) on train rows.

TRUNCATE TABLE game_data_filtered.player_role_1vx;

INSERT INTO game_data_filtered.player_role_1vx
(
    split,
    puuid,
    teamposition,
    matchups
)
SELECT
    'train' AS split,
    toString(ps.puuid) AS puuid,
    assumeNotNull(ps.teamposition) AS teamposition,
    toUInt32(count()) AS matchups
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS s
    ON ps.matchid = s.matchid
WHERE
    s.split = 'train'
GROUP BY
    puuid,
    teamposition;
