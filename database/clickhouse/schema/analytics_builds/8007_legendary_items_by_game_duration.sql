-- Average legendary items completed per participant, bucketed into 2-minute bins.
-- Legendary items: item_value_map_dict entries with the sentinel key (championid=0,
-- teamposition='', itemid) — i.e. null/null rows in item_value_map.jsonl.
-- gameduration is in seconds (from game_data.info).
--
-- Outputs the boundaries of each bin plus the median gameduration within each
-- bin. The medians are the "true population center" of each bin (50% of games
-- in the bin sit on either side) and are consumed as phase centers by
-- table 5133 (participant_scaling_weights).

-- sqlfluff: disable=RF02

WITH item_counts AS (
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
                ps.item0,
                ps.item1,
                ps.item2,
                ps.item3,
                ps.item4,
                ps.item5,
                ps.item6
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
        avg(legendary_item_count) AS avg_legendary_items,
        count() AS sample_size
    FROM item_counts
    GROUP BY duration_seconds
),

thresholds AS (
    SELECT
        minIf(duration_seconds, avg_legendary_items >= 3) AS threshold_3_seconds,
        minIf(duration_seconds, avg_legendary_items >= 4) AS threshold_4_seconds,
        minIf(duration_seconds, avg_legendary_items >= 5) AS threshold_5_seconds
    FROM avg_by_exact_duration
),

bin_medians AS (
    SELECT
        quantileExactIf(0.5)(
            ic.gameduration,
            ic.gameduration < t.threshold_3_seconds
        ) AS median_bin_23,
        quantileExactIf(0.5)(
            ic.gameduration,
            ic.gameduration >= t.threshold_3_seconds
            AND ic.gameduration < t.threshold_4_seconds
        ) AS median_bin_34,
        quantileExactIf(0.5)(
            ic.gameduration,
            ic.gameduration >= t.threshold_4_seconds
            AND ic.gameduration < t.threshold_5_seconds
        ) AS median_bin_45,
        quantileExactIf(0.5)(
            ic.gameduration,
            ic.gameduration >= t.threshold_5_seconds
        ) AS median_bin_5plus
    FROM item_counts AS ic
    CROSS JOIN thresholds AS t
    WHERE ic.gameduration >= 18 * 60
),

games AS (
    SELECT DISTINCT matchid, gameduration
    FROM item_counts
    WHERE gameduration >= 18 * 60
),

bin_counts AS (
    SELECT
        countIf(g.gameduration < t.threshold_3_seconds) AS games_bin_23,
        countIf(g.gameduration >= t.threshold_3_seconds AND g.gameduration < t.threshold_4_seconds) AS games_bin_34,
        countIf(g.gameduration >= t.threshold_4_seconds AND g.gameduration < t.threshold_5_seconds) AS games_bin_45,
        countIf(g.gameduration >= t.threshold_5_seconds) AS games_bin_5plus,
        count() AS games_total
    FROM games AS g
    CROSS JOIN thresholds AS t
)

SELECT
    bin_label,
    bin_start_seconds,
    bin_end_seconds,
    median_seconds,
    game_count,

    round(bin_start_seconds / 60, 2) AS bin_start_minutes,

    if(
        isNull(bin_end_seconds),
        '40+',
        toString(round(bin_end_seconds / 60, 2))
    ) AS bin_end_minutes,

    round(median_seconds / 60, 2) AS median_minutes,

    round(100.0 * game_count / games_total, 2) AS pct_of_games

FROM
    (
        SELECT
            '2-3 items' AS bin_label,
            18 * 60 AS bin_start_seconds,
            threshold_3_seconds AS bin_end_seconds,
            median_bin_23 AS median_seconds,
            games_bin_23 AS game_count,
            games_total
        FROM thresholds
        CROSS JOIN bin_medians
        CROSS JOIN bin_counts

        UNION ALL

        SELECT
            '3-4 items' AS bin_label,
            threshold_3_seconds AS bin_start_seconds,
            threshold_4_seconds AS bin_end_seconds,
            median_bin_34 AS median_seconds,
            games_bin_34 AS game_count,
            games_total
        FROM thresholds
        CROSS JOIN bin_medians
        CROSS JOIN bin_counts

        UNION ALL

        SELECT
            '4-5 items' AS bin_label,
            threshold_4_seconds AS bin_start_seconds,
            threshold_5_seconds AS bin_end_seconds,
            median_bin_45 AS median_seconds,
            games_bin_45 AS game_count,
            games_total
        FROM thresholds
        CROSS JOIN bin_medians
        CROSS JOIN bin_counts

        UNION ALL

        SELECT
            '5+ items' AS bin_label,
            threshold_5_seconds AS bin_start_seconds,
            NULL AS bin_end_seconds,
            median_bin_5plus AS median_seconds,
            games_bin_5plus AS game_count,
            games_total
        FROM thresholds
        CROSS JOIN bin_medians
        CROSS JOIN bin_counts
    )
ORDER BY bin_start_seconds;
