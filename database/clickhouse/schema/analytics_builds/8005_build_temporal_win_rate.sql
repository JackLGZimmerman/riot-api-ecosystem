-- noqa: disable=AL05,LT01,LT02,LT05,RF02,PRS
--
-- Ad-hoc inspection query: win rate per (championid, teamposition, build,
-- time_bin) using the participant's strongest temporal bin from
-- participant_scaling_weights. Unlike 6010_temporal_1vx_aggregations_build.sql,
-- this emits raw integer matchups rather than soft-attributed effective
-- sample sizes.
--
-- Requires game_data_filtered.participant_stats,
-- game_data_filtered.participant_item_value_totals,
-- game_data_filtered.participant_scaling_weights, and
-- game_data_filtered.ml_game_split to be populated.

SELECT
    assumeNotNull(ps.championid) AS championid,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(assumeNotNull(ps.championid)),
        ''
    ) AS championname,
    toString(ps.teamposition) AS teamposition,
    toString(ivt.highest_value_label) AS build,
    toString(psw.max_value_bin) AS time_bin,
    COUNT() AS matchups,
    toFloat32(sum(ps.win) / count()) AS win_rate
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS s
    ON ps.matchid = s.matchid
INNER JOIN game_data_filtered.participant_item_value_totals AS ivt
    ON
        ps.matchid = ivt.matchid
        AND ps.participantid = ivt.participantid
INNER JOIN game_data_filtered.participant_scaling_weights AS psw
    ON
        ps.matchid = psw.matchid
        AND ps.participantid = psw.participantid
WHERE s.split = 'train'
GROUP BY
    championid,
    teamposition,
    build,
    time_bin
HAVING matchups > 20
ORDER BY
    championid ASC,
    teamposition ASC,
    build ASC,
    time_bin ASC;
