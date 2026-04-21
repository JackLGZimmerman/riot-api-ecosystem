-- Ad-hoc inspection query: build-label distribution per (championid,
-- teamposition) in the filtered dataset.  Useful for validating the rare
-- build threshold (stage 3 of the filter pipeline).
--
-- Requires game_data_filtered.participant_item_value_totals to be populated
-- (See .../5133_participant_item_value_totals_build.sql).

SELECT
    championid,
    teamposition,
    highest_value_label,
    dictGet('game_data.championid_name_map_dict', 'name', toInt32(championid))
        AS champion_name,
    COUNT() AS instances,

    -- total rows per champion + position
    SUM(COUNT()) OVER (PARTITION BY championid, teamposition) AS total_instances,

    -- distinct labels per champion + position (replicated)
    -- uniqExact is not supported as a window function; after GROUP BY each row is
    -- already one distinct label, so count() over the partition is equivalent
    count()
        OVER (PARTITION BY championid, teamposition)
        AS distinct_labels,

    -- percentage of total
    COUNT()
    / SUM(COUNT()) OVER (PARTITION BY championid, teamposition)
    * 100 AS pct_of_total

FROM game_data_filtered.participant_item_value_totals
GROUP BY
    championid,
    teamposition,
    highest_value_label
ORDER BY
    championid,
    teamposition,
    highest_value_label
