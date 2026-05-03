-- Ad-hoc inspection query: build-label distribution per (championid,
-- teamposition) in the filtered dataset.  Useful for validating the rare
-- build threshold (stage 3 of the filter pipeline).
--
-- Requires game_data_filtered.participant_item_value_totals to be populated
-- (See .../5133_participant_item_value_totals_build.sql).

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
    dictGet('game_data.championid_name_map_dict', 'name', toInt32(ivt.championid))
        AS champion_name,
    COUNT() AS instances,

    -- total rows per champion + position
    SUM(COUNT())
        OVER (PARTITION BY ivt.championid, ivt.teamposition)
        AS total_instances,

    -- distinct labels per champion + position (replicated)
    -- uniqExact is not supported as a window function; after GROUP BY each row is
    -- already one distinct label, so count() over the partition is equivalent
    count()
        OVER (PARTITION BY ivt.championid, ivt.teamposition)
        AS distinct_labels,

    -- percentage of total
    COUNT()
    / SUM(COUNT()) OVER (PARTITION BY ivt.championid, ivt.teamposition)
    * 100 AS pct_of_total,
    round(avg(toFloat64(ps.win)) * 100, 2) AS avg_win_rate

FROM game_data_filtered.participant_item_value_totals AS ivt
INNER JOIN ps
    ON
        ivt.matchid = ps.matchid
        AND ivt.participantid = ps.participantid
GROUP BY
    ivt.championid,
    ivt.teamposition,
    ivt.highest_value_label
HAVING instances > 40
ORDER BY
    avg_win_rate DESC,
    ivt.championid ASC,
    ivt.teamposition ASC,
    ivt.highest_value_label ASC
