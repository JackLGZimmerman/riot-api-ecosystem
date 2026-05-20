-- noqa: disable=PRS
-- Populate game_data_filtered.participant_scaling_weights.
--
-- For each (matchid, participantid) we emit four soft-attribution scaling
-- weights (early_mid, mid, mid_late, late) over the game's temporal phases.
-- The weights sum to 1.0 and at most TWO adjacent phases ever carry a
-- non-zero value. max_value_bin records the phase carrying the largest
-- weight. This adds scaling granularity to the (championid, teamposition,
-- build) grouping captured by 5132 (participant_item_value_totals).
--
-- Phase boundaries come from analytics_builds/8007: average legendary items
-- completed per participant defines four bins (2-3 / 3-4 / 4-5 / 5+ items),
-- delimited at the gameduration thresholds where the population average
-- crosses 3, 4 and 5 legendary items. Phase CENTERS are the median
-- gameduration within each bin -- i.e. the time value where 50% of games in
-- the bin sit below and 50% above. The four centers are:
--   c_early_mid = median of games in the 2-3 items bin
--   c_mid       = median of games in the 3-4 items bin
--   c_mid_late  = median of games in the 4-5 items bin
--   c_late      = median of games in the 5+ items bin
--
-- Weight assignment (linear interpolation between the two nearest centers):
--   d <= c_early_mid                  -> early_mid = 1
--   c_early_mid < d < c_mid           -> early_mid + mid     (split)
--   c_mid       <= d < c_mid_late     -> mid       + mid_late (split)
--   c_mid_late  <= d < c_late         -> mid_late  + late    (split)
--   d >= c_late                       -> late = 1
-- The split weight is (distance_to_far_center / distance_between_centers).
--
-- Requires game_data_filtered.participant_stats and game_data.info to be
-- populated.

TRUNCATE TABLE game_data_filtered.participant_scaling_weights;

INSERT INTO game_data_filtered.participant_scaling_weights
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
    ANY INNER JOIN game_data.info AS i
        ON ps.matchid = i.matchid
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
),

phase_centers AS (
    SELECT
        toFloat64(quantileExactIf(0.5) (
            ic.gameduration,
            ic.gameduration < t.threshold_3
        )) AS c_early_mid,
        toFloat64(quantileExactIf(0.5) (
            ic.gameduration,
            ic.gameduration >= t.threshold_3
            AND ic.gameduration < t.threshold_4
        )) AS c_mid,
        toFloat64(quantileExactIf(0.5) (
            ic.gameduration,
            ic.gameduration >= t.threshold_4
            AND ic.gameduration < t.threshold_5
        )) AS c_mid_late,
        toFloat64(quantileExactIf(0.5) (
            ic.gameduration,
            ic.gameduration >= t.threshold_5
        )) AS c_late
    FROM item_counts AS ic
    CROSS JOIN thresholds AS t
),

weighted AS (
    SELECT
        ps.matchid,
        ps.teamid,
        ps.participantid,
        ps.puuid,
        ps.championid,
        ps.teamposition,
        i.gameduration,

        toFloat32(multiIf(
            i.gameduration <= pc.c_early_mid, 1.0,
            i.gameduration >= pc.c_mid, 0.0,
            (pc.c_mid - i.gameduration) / (pc.c_mid - pc.c_early_mid)
        )) AS early_mid,

        toFloat32(multiIf(
            i.gameduration <= pc.c_early_mid, 0.0,
            i.gameduration < pc.c_mid,
            (i.gameduration - pc.c_early_mid) / (pc.c_mid - pc.c_early_mid),
            i.gameduration < pc.c_mid_late,
            (pc.c_mid_late - i.gameduration) / (pc.c_mid_late - pc.c_mid),
            0.0
        )) AS mid,

        toFloat32(multiIf(
            i.gameduration <= pc.c_mid, 0.0,
            i.gameduration < pc.c_mid_late,
            (i.gameduration - pc.c_mid) / (pc.c_mid_late - pc.c_mid),
            i.gameduration < pc.c_late,
            (pc.c_late - i.gameduration) / (pc.c_late - pc.c_mid_late),
            0.0
        )) AS mid_late,

        toFloat32(multiIf(
            i.gameduration < pc.c_mid_late, 0.0,
            i.gameduration < pc.c_late,
            (i.gameduration - pc.c_mid_late) / (pc.c_late - pc.c_mid_late),
            1.0
        )) AS late
    FROM game_data_filtered.participant_stats AS ps
    ANY INNER JOIN game_data.info AS i
        ON ps.matchid = i.matchid
    CROSS JOIN phase_centers AS pc
),

final AS (
    SELECT
        *,
        multiIf(
            early_mid = greatest(early_mid, mid, mid_late, late), 'early_mid',
            mid = greatest(mid, mid_late, late), 'mid',
            mid_late = greatest(mid_late, late), 'mid_late',
            'late'
        ) AS max_value_bin
    FROM weighted
)

SELECT *
FROM final;
