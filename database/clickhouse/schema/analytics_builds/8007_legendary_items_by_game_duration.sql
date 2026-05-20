-- Average legendary items completed per participant, bucketed by game duration.
-- Legendary items: item_value_map_dict entries with the sentinel key
-- (championid=0, teamposition='', itemid) -- null/null rows in item_value_map.jsonl.
-- gameduration is seconds (game_data.info).
--
-- For each target N in {3, 4, 5}, finds the minimum gameduration at which the
-- per-participant average legendary-item count first reaches N. The three
-- thresholds partition games into 4 bins (the games below t3 already have
-- avg >= 2, so the first bin starts at the 16.5-min floor):
--
--   2-3 items  : 16.5 min .. t3
--   3-4 items  : t3       .. t4
--   4-5 items  : t4       .. t5
--   5+ items   : t5       .. inf
--
-- avg_legendary_items in the output is the achieved per-participant average
-- inside each bin: it should sit close to (N + 0.5) for the middle bins,
-- confirming the threshold search did what the bin label claims.

-- sqlfluff: disable=RF02

WITH participant_legendary AS (
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
    INNER JOIN game_data.info AS i
        ON ps.matchid = i.matchid
    WHERE i.gameduration >= 16.5 * 60
),

thresholds AS (
    SELECT
        toFloat64(16.5 * 60) AS t_min,
        minIf(duration_seconds, avg_count >= 3) AS t3,
        minIf(duration_seconds, avg_count >= 4) AS t4,
        minIf(duration_seconds, avg_count >= 5) AS t5
    FROM (
        SELECT
            gameduration AS duration_seconds,
            avg(legendary_item_count) AS avg_count
        FROM participant_legendary
        GROUP BY duration_seconds
    )
),

binned AS (
    SELECT
        pl.matchid,
        pl.gameduration,
        pl.legendary_item_count,
        multiIf(
            pl.gameduration < t.t3, 1,
            pl.gameduration < t.t4, 2,
            pl.gameduration < t.t5, 3,
            4
        ) AS bin_idx,
        multiIf(
            pl.gameduration < t.t3, t.t_min,
            pl.gameduration < t.t4, t.t3,
            pl.gameduration < t.t5, t.t4,
            t.t5
        ) AS bin_start_s,
        multiIf(
            pl.gameduration < t.t3, toNullable(t.t3),
            pl.gameduration < t.t4, toNullable(t.t4),
            pl.gameduration < t.t5, toNullable(t.t5),
            CAST(NULL AS Nullable(Float64))
        ) AS bin_end_s
    FROM participant_legendary AS pl
    CROSS JOIN thresholds AS t
),

per_bin AS (
    SELECT
        bin_idx,
        bin_start_s,
        bin_end_s,
        avg(legendary_item_count) AS avg_legendary_items,
        quantileExact(0.5)(gameduration) AS median_s,
        uniqExact(matchid) AS games,
        count() AS participants
    FROM binned
    GROUP BY bin_idx, bin_start_s, bin_end_s
)

SELECT
    multiIf(
        bin_idx = 1, '2-3 items',
        bin_idx = 2, '3-4 items',
        bin_idx = 3, '4-5 items',
        '5+ items'
    ) AS bin_label,

    round(bin_start_s / 60, 2) AS bin_start_min,
    if(
        isNull(bin_end_s),
        'inf',
        toString(round(bin_end_s / 60, 2))
    ) AS bin_end_min,

    round(avg_legendary_items, 3) AS avg_legendary_items,
    round(median_s / 60, 2) AS median_min,

    games,
    participants,
    round(100.0 * games / sum(games) OVER (), 2) AS pct_games

FROM per_bin
ORDER BY bin_idx;
