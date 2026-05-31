-- noqa: disable=AL05,LT01,LT02,LT05,RF02
--
-- Draft-time-safe champion/role priors for the ML model.
--
-- This mirrors 6003_1vx_aggregations_build.sql but removes the final
-- item-derived build label. It emits one constant build label ('unknown') so
-- downstream cache builds can use champion-role-only target encodings.

TRUNCATE TABLE game_data_filtered.synergy_1vx;

INSERT INTO game_data_filtered.synergy_1vx
(
    split,
    championid,
    championname,
    teamposition,
    build,
    matchups,
    win_rate
)
SELECT
    'train' AS split,
    championid,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(championid),
        ''
    ) AS championname,
    teamposition,
    'unknown' AS build,
    toUInt32(count()) AS matchups,
    toFloat32(sum(win) / count()) AS win_rate
FROM (
    SELECT
        assumeNotNull(ps.championid) AS championid,
        toString(ps.teamposition) AS teamposition,
        toUInt8(ps.win > 0) AS win
    FROM game_data_filtered.participant_stats AS ps
    INNER JOIN game_data_filtered.ml_game_split AS s
        ON ps.matchid = s.matchid
    WHERE
        s.split = 'train'

)
GROUP BY
    championid,
    teamposition;
