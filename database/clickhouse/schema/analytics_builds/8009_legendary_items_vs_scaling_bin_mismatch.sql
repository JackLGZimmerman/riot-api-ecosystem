-- Find games where the legendary-items bin (set by the gameduration
-- thresholds at which the population average crosses 3 / 4 / 5 legendary
-- items, per 8007) does NOT match the max_value_bin in
-- participant_scaling_weights (set by which median phase center is closest).
--
-- These are durations sitting near a bin boundary where the bin's median
-- pulls the midpoint between two adjacent centers across the threshold,
-- causing a player whose duration is in (e.g.) the "3-4 items" bin to be
-- weighted predominantly as "early_mid" rather than "mid".
--
-- Thresholds are recomputed inline so the query stays self-consistent with
-- the current data.
--
-- Requires:
--   - game_data_filtered.participant_stats          (see 5003)
--   - game_data_filtered.participant_scaling_weights (see 5133)

WITH
item_counts AS (
    SELECT
        ps.matchid,
        i.gameduration,
        arraySum(arrayMap(
            x -> if(
                x != 0
                AND dictHas(
                    'game_data.item_value_map_dict',
                    (toInt32(0), toString(''), toUInt32(x))
                ),
                1,
                0
            ),
            [
                ps.item0, ps.item1, ps.item2, ps.item3,
                ps.item4, ps.item5, ps.item6
            ]
        )) AS legendary_item_count
    FROM game_data_filtered.participant_stats AS ps
    ANY INNER JOIN game_data.info AS i FINAL
        ON ps.matchid = i.matchid
    WHERE i.gameduration >= 15 * 60
),

avg_by_exact_duration AS (
    SELECT
        gameduration AS duration_seconds,
        avg(legendary_item_count) AS avg_legendary_items
    FROM item_counts
    GROUP BY duration_seconds
),

thresholds AS (
    SELECT
        minIf(duration_seconds, avg_legendary_items >= 3) AS threshold_3,
        minIf(duration_seconds, avg_legendary_items >= 4) AS threshold_4,
        minIf(duration_seconds, avg_legendary_items >= 5) AS threshold_5
    FROM avg_by_exact_duration
)

SELECT
    gameduration,
    legendary_items_bin,
    max_value_bin,
    participant_rows,
    round(gameduration / 60.0, 2) AS gameduration_minutes,
    round(avg_early_mid, 4) AS avg_early_mid,
    round(avg_mid, 4) AS avg_mid,
    round(avg_mid_late, 4) AS avg_mid_late,
    round(avg_late, 4) AS avg_late
FROM (
    SELECT
        sw.gameduration,
        sw.max_value_bin,
        multiIf(
            sw.gameduration < t.threshold_3, '2-3 items',
            sw.gameduration < t.threshold_4, '3-4 items',
            sw.gameduration < t.threshold_5, '4-5 items',
            '5+ items'
        ) AS legendary_items_bin,
        avg(sw.early_mid) AS avg_early_mid,
        avg(sw.mid) AS avg_mid,
        avg(sw.mid_late) AS avg_mid_late,
        avg(sw.late) AS avg_late,
        count() AS participant_rows
    FROM game_data_filtered.participant_scaling_weights AS sw
    CROSS JOIN thresholds AS t
    GROUP BY
        sw.gameduration,
        legendary_items_bin,
        sw.max_value_bin
)
WHERE
    (legendary_items_bin = '2-3 items' AND max_value_bin != 'early_mid')
    OR (legendary_items_bin = '3-4 items' AND max_value_bin != 'mid')
    OR (legendary_items_bin = '4-5 items' AND max_value_bin != 'mid_late')
    OR (legendary_items_bin = '5+ items' AND max_value_bin != 'late')
ORDER BY gameduration ASC;
