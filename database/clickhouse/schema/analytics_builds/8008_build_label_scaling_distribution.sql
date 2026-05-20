-- Ad-hoc inspection query: build-label distribution per (championid,
-- teamposition) further stratified by scaling bin (early_mid / mid /
-- mid_late / late). Each row shows how often a champion+position+build is
-- played at a given scaling, what fraction of that build's instances fall in
-- the bin, and the win rate for that combination.
--
-- Requires:
--   - game_data_filtered.participant_item_value_totals  (see 5132)
--   - game_data_filtered.participant_scaling_weights    (see 5133)

WITH ps AS (
    SELECT
        matchid,
        participantid,
        win
    FROM game_data_filtered.participant_stats
)

SELECT
    ivt.championid,
    ivt.teamposition,
    ivt.highest_value_label,
    sw.max_value_bin,
    dictGet('game_data.championid_name_map_dict', 'name', toInt32(ivt.championid))
        AS champion_name,
    COUNT() AS instances,

    -- total rows per champion + position + build (across all scaling bins)
    SUM(COUNT())
        OVER (PARTITION BY ivt.championid, ivt.teamposition, ivt.highest_value_label)
        AS build_total_instances,

    -- percentage of this build's instances that fall in this scaling bin
    COUNT()
    / SUM(COUNT())
        OVER (PARTITION BY ivt.championid, ivt.teamposition, ivt.highest_value_label)
    * 100 AS pct_of_build,

    -- percentage of all (champion + position) instances
    COUNT()
    / SUM(COUNT())
        OVER (PARTITION BY ivt.championid, ivt.teamposition)
    * 100 AS pct_of_total,

    round(avg(toFloat64(ps.win)) * 100, 2) AS avg_win_rate

FROM game_data_filtered.participant_item_value_totals AS ivt
INNER JOIN game_data_filtered.participant_scaling_weights AS sw
    ON
        ivt.matchid = sw.matchid
        AND ivt.participantid = sw.participantid
INNER JOIN ps
    ON
        ivt.matchid = ps.matchid
        AND ivt.participantid = ps.participantid
GROUP BY
    ivt.championid,
    ivt.teamposition,
    ivt.highest_value_label,
    sw.max_value_bin
HAVING instances > 20
ORDER BY
    avg_win_rate DESC,
    ivt.championid ASC,
    ivt.teamposition ASC,
    ivt.highest_value_label ASC,
    sw.max_value_bin ASC
