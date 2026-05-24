-- noqa: disable=AL05,LT01,LT02,LT05,RF02
--
-- Build-labeled draft-time champion/role priors for the ML model.
--
-- This emits only the count and win-rate prior for each
-- (championid, teamposition, build). It deliberately does not include the
-- previous scaling-bin, time, or profile-metric columns.

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
    build,
    toUInt32(count()) AS matchups,
    toFloat32(sum(win) / count()) AS win_rate
FROM (
    SELECT
        assumeNotNull(ps.championid) AS championid,
        toString(ps.teamposition) AS teamposition,
        toString(ivt.highest_value_label) AS build,
        toUInt8(ps.win > 0) AS win
    FROM game_data_filtered.participant_stats AS ps
    INNER JOIN game_data_filtered.ml_game_split AS s
        ON ps.matchid = s.matchid
    INNER JOIN game_data_filtered.participant_item_value_totals AS ivt
        ON
            ps.matchid = ivt.matchid
            AND ps.participantid = ivt.participantid
    WHERE
        s.split = 'train'

)
GROUP BY
    championid,
    teamposition,
    build;
