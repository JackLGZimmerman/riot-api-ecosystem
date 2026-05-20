-- Legendary-item scaling bins with deterministic probabilistic boundary
-- assignment ("smoothing"), derived from 8007_legendary_items_by_game_duration.
--
-- Legendary items: item_value_map_dict entries with the sentinel key
-- (championid=0, teamposition='', itemid) -- null/null rows in item_value_map.jsonl.
-- gameduration is seconds (game_data.info).
--
-- The thresholds t3/t4/t5 and the four bins are unchanged from 8007:
--
--   2-3 items  : 16.5 min .. t3
--   3-4 items  : t3       .. t4
--   4-5 items  : t4       .. t5
--   5+ items   : t5       .. inf
--
-- SMOOTHING
-- ---------
-- Strict assignment places a row in whichever bin its gameduration falls in.
-- That makes the bin counts jump discontinuously at t3/t4/t5: a one-second
-- difference in game length flips a whole row between bins.
--
-- Instead of a fixed window around each boundary, the transition between two
-- adjacent bins runs from the MEDIAN of the lower bin to the MEDIAN of the
-- higher bin (strict-bin medians, computed once in bin_medians):
--
--   * Below the lowest bin median           -> always the lowest bin.
--   * Above the highest bin median          -> always the highest bin.
--   * Between two adjacent bin medians m_lo and m_hi the row is in a
--     transition. Let x = (gameduration - m_lo) / (m_hi - m_lo) be the
--     normalised position in [0, 1]. The probability of landing in the
--     HIGHER bin follows a normal CDF (an S-curve) centred at x = 0.5:
--         prob_higher = 0.5 * (1 + erf((x - 0.5) / (sigma * sqrt(2))))
--     This makes movement exceedingly unlikely near the bin medians
--     (x -> 0 or x -> 1) and concentrates the smoothing mid-transition.
--     transition_sigma (in normalised x units) sets how sharp the S-curve
--     is; smaller = sharper. At m_lo prob_higher ~ 0, at m_hi ~ 1.
--   * A stable 0-1 value is produced from cityHash64(matchid, participantid,
--     boundary_id, hash_seed). If hash_value < prob_higher the whole row goes
--     to the higher bin, otherwise the whole row goes to the lower bin.
--
-- Properties: every row lands in exactly one bin (no fractional / weighted
-- counts), the assignment is idempotent across runs (no rand()).
--
-- The delta_* columns expose the smoothing effect by comparing the strict and
-- smoothed counts side by side.

-- sqlfluff: disable=RF02

WITH participant_legendary AS (
    SELECT
        ps.matchid,
        ps.participantid,
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

-- Single-row CTE: bin thresholds plus the smoothing seed.
thresholds AS (
    SELECT
        toFloat64(16.5 * 60) AS t_min,
        minIf(duration_seconds, avg_count >= 3) AS t3,
        minIf(duration_seconds, avg_count >= 4) AS t4,
        minIf(duration_seconds, avg_count >= 5) AS t5,
        -- Smoothing constants. hash_seed reshuffles the deterministic
        -- assignment; transition_sigma sets the S-curve sharpness (smaller
        -- sigma -> steeper, even less movement near the bin medians).
        toUInt64(2654435761) AS hash_seed,
        toFloat64(0.15) AS transition_sigma
    FROM (
        SELECT
            gameduration AS duration_seconds,
            avg(legendary_item_count) AS avg_count
        FROM participant_legendary
        GROUP BY duration_seconds
    )
),

-- Single-row CTE: median gameduration of each strict bin. These medians are
-- the endpoints of the smoothing transitions.
bin_medians AS (
    SELECT
        quantileExactIf(0.5)(gameduration, strict_bin_idx = 1) AS m1,
        quantileExactIf(0.5)(gameduration, strict_bin_idx = 2) AS m2,
        quantileExactIf(0.5)(gameduration, strict_bin_idx = 3) AS m3,
        quantileExactIf(0.5)(gameduration, strict_bin_idx = 4) AS m4
    FROM (
        SELECT
            pl.gameduration,
            multiIf(
                pl.gameduration < t.t3, 1,
                pl.gameduration < t.t4, 2,
                pl.gameduration < t.t5, 3,
                4
            ) AS strict_bin_idx
        FROM participant_legendary AS pl
        CROSS JOIN thresholds AS t
    )
),

-- One row per participant with both the strict bin and the smoothed bin.
-- The transition for each boundary spans the medians of the two adjacent bins;
-- outside the m1..m4 span the row keeps its strict (extreme) bin.
assigned AS (
    SELECT
        pl.matchid,
        pl.gameduration,
        pl.legendary_item_count,
        multiIf(
            pl.gameduration < t.t3, 1,
            pl.gameduration < t.t4, 2,
            pl.gameduration < t.t5, 3,
            4
        ) AS strict_bin_idx,
        multiIf(
            pl.gameduration < bm.m1, 1,
            pl.gameduration < bm.m2,
            if(
                toFloat64(cityHash64(
                    pl.matchid, pl.participantid, toUInt8(3), t.hash_seed
                )) / 1.8446744073709552e19
                < 0.5 * (1 + erf(
                    ((pl.gameduration - bm.m1) / (bm.m2 - bm.m1) - 0.5)
                    / (t.transition_sigma * sqrt(2))
                )),
                2, 1
            ),
            pl.gameduration < bm.m3,
            if(
                toFloat64(cityHash64(
                    pl.matchid, pl.participantid, toUInt8(4), t.hash_seed
                )) / 1.8446744073709552e19
                < 0.5 * (1 + erf(
                    ((pl.gameduration - bm.m2) / (bm.m3 - bm.m2) - 0.5)
                    / (t.transition_sigma * sqrt(2))
                )),
                3, 2
            ),
            pl.gameduration < bm.m4,
            if(
                toFloat64(cityHash64(
                    pl.matchid, pl.participantid, toUInt8(5), t.hash_seed
                )) / 1.8446744073709552e19
                < 0.5 * (1 + erf(
                    ((pl.gameduration - bm.m3) / (bm.m4 - bm.m3) - 0.5)
                    / (t.transition_sigma * sqrt(2))
                )),
                4, 3
            ),
            4
        ) AS smoothed_bin_idx
    FROM participant_legendary AS pl
    CROSS JOIN thresholds AS t
    CROSS JOIN bin_medians AS bm
),

-- Smoothed aggregates: avg / median / counts follow the smoothed assignment.
smoothed_per_bin AS (
    SELECT
        smoothed_bin_idx AS bin_idx,
        avg(legendary_item_count) AS avg_legendary_items,
        quantileExact(0.5)(gameduration) AS median_s,
        uniqExact(matchid) AS smoothed_games,
        count() AS smoothed_participants
    FROM assigned
    GROUP BY smoothed_bin_idx
),

-- Strict counts for the side-by-side delta columns.
strict_per_bin AS (
    SELECT
        strict_bin_idx AS bin_idx,
        uniqExact(matchid) AS strict_games,
        count() AS strict_participants
    FROM assigned
    GROUP BY strict_bin_idx
)

SELECT
    s.bin_idx AS bin_idx,

    multiIf(
        bin_idx = 1, '2-3 items',
        bin_idx = 2, '3-4 items',
        bin_idx = 3, '4-5 items',
        '5+ items'
    ) AS bin_label,

    round(multiIf(
        bin_idx = 1, t.t_min,
        bin_idx = 2, t.t3,
        bin_idx = 3, t.t4,
        t.t5
    ) / 60, 2) AS bin_start_min,
    multiIf(
        bin_idx = 1, toString(round(t.t3 / 60, 2)),
        bin_idx = 2, toString(round(t.t4 / 60, 2)),
        bin_idx = 3, toString(round(t.t5 / 60, 2)),
        'inf'
    ) AS bin_end_min,

    round(s.avg_legendary_items, 3) AS avg_legendary_items,
    round(s.median_s / 60, 2) AS median_min,

    s.smoothed_games AS games,
    s.smoothed_participants AS participants,
    round(100.0 * s.smoothed_games / sum(s.smoothed_games) OVER (), 2) AS pct_games,

    -- Strict vs smoothed comparison.
    st.strict_games AS strict_games,
    s.smoothed_games AS smoothed_games,
    toInt64(s.smoothed_games) - toInt64(st.strict_games) AS delta_games,

    st.strict_participants AS strict_participants,
    s.smoothed_participants AS smoothed_participants,
    toInt64(s.smoothed_participants) - toInt64(st.strict_participants) AS delta_participants,

    round(100.0 * st.strict_games / sum(st.strict_games) OVER (), 2) AS strict_pct_games,
    round(100.0 * s.smoothed_games / sum(s.smoothed_games) OVER (), 2) AS smoothed_pct_games,
    round(smoothed_pct_games - strict_pct_games, 2) AS delta_pct_games

FROM smoothed_per_bin AS s
INNER JOIN strict_per_bin AS st ON s.bin_idx = st.bin_idx
CROSS JOIN thresholds AS t
ORDER BY bin_idx;
