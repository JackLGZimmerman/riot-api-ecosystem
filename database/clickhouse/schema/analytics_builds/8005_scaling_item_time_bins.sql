-- noqa: disable=AL09,LT02,LT05,PRS
--
-- Scaling time-bin calibration and deterministic match assignment.
--
-- "Completed items" are final inventory slots item0-item6 whose item id exists
-- in the generic item-value map key (championid = 0, teamposition = '').
-- The timing source is game_data_filtered.participant_stats only; timeline item
-- purchase events are intentionally not used here. The first bin starts at the
-- observed minimum eligible game length, not at a hard-coded base timer.

DROP TABLE IF EXISTS game_data_filtered.scaling_match_item_times_stg;

CREATE TABLE game_data_filtered.scaling_match_item_times_stg
ENGINE = MergeTree
ORDER BY matchid
AS
WITH participant_item_counts AS (
    SELECT
        matchid,
        timeplayed,
        toUInt8(arrayCount(
            item_id -> dictHas(
                'game_data.item_value_map_dict',
                (toInt32(0), '', item_id)
            ),
            [
                toUInt32(item0), toUInt32(item1), toUInt32(item2),
                toUInt32(item3), toUInt32(item4), toUInt32(item5),
                toUInt32(item6)
            ]
        )) AS completed_items
    FROM game_data_filtered.participant_stats
)

SELECT
    matchid,
    toFloat64(max(timeplayed)) / 60.0 AS game_minutes,
    avg(toFloat64(completed_items)) AS avg_completed_items
FROM participant_item_counts
GROUP BY matchid
HAVING count() = 10;

DROP TABLE IF EXISTS game_data_filtered.scaling_time_bins;

CREATE TABLE game_data_filtered.scaling_time_bins
(
    bin_index UInt8,
    bin_label LowCardinality(String),
    item_count_from Float32,
    item_count_to_exclusive Nullable(Float32),
    match_count UInt64,
    avg_game_minutes Float32,
    from_minutes Float32,
    to_minutes Float32,
    centroid_minutes Float32
)
ENGINE = MergeTree
ORDER BY bin_index;

INSERT INTO game_data_filtered.scaling_time_bins
WITH
bin_defs AS (
    SELECT
        tupleElement(bin_def, 1) AS bin_index,
        tupleElement(bin_def, 2) AS bin_label,
        tupleElement(bin_def, 3) AS item_count_from,
        tupleElement(bin_def, 4) AS item_count_to_exclusive
    FROM (
        SELECT arrayJoin([
            tuple(toUInt8(1), '2-3 items', toFloat32(2), toNullable(toFloat32(3))),
            tuple(toUInt8(2), '3-4 items', toFloat32(3), toNullable(toFloat32(4))),
            tuple(toUInt8(3), '4-5 items', toFloat32(4), toNullable(toFloat32(5))),
            tuple(toUInt8(4), '5+ items', toFloat32(5), CAST(NULL, 'Nullable(Float32)'))
        ]) AS bin_def
    )
),

raw_bins AS (
    SELECT
        d.bin_index,
        d.bin_label,
        d.item_count_from,
        d.item_count_to_exclusive,
        count() AS match_count,
        avg(m.game_minutes) AS avg_game_minutes
    FROM game_data_filtered.scaling_match_item_times_stg AS m
    INNER JOIN bin_defs AS d
        ON
            m.avg_completed_items >= d.item_count_from
            AND (
                isNull(d.item_count_to_exclusive)
                OR m.avg_completed_items < assumeNotNull(d.item_count_to_exclusive)
            )
    GROUP BY
        d.bin_index,
        d.bin_label,
        d.item_count_from,
        d.item_count_to_exclusive
),

endpoints AS (
    SELECT
        (
            SELECT min(game_minutes)
            FROM game_data_filtered.scaling_match_item_times_stg
        ) AS min_game_minutes,
        maxIf(avg_game_minutes, bin_index = 1) AS bin_1_to_minutes,
        maxIf(avg_game_minutes, bin_index = 2) AS bin_2_to_minutes,
        maxIf(avg_game_minutes, bin_index = 3) AS bin_3_to_minutes
    FROM raw_bins
)

SELECT
    bin_index,
    bin_label,
    item_count_from,
    item_count_to_exclusive,
    toUInt64(match_count) AS match_count,
    toFloat32(avg_game_minutes) AS avg_game_minutes,
    toFloat32(from_minutes) AS from_minutes,
    toFloat32(avg_game_minutes) AS to_minutes,
    toFloat32((from_minutes + avg_game_minutes) / 2.0) AS centroid_minutes
FROM (
    SELECT
        r.bin_index,
        r.bin_label,
        r.item_count_from,
        r.item_count_to_exclusive,
        r.match_count,
        r.avg_game_minutes,
        multiIf(
            r.bin_index = 1, e.min_game_minutes,
            r.bin_index = 2, e.bin_1_to_minutes,
            r.bin_index = 3, e.bin_2_to_minutes,
            e.bin_3_to_minutes
        ) AS from_minutes
    FROM raw_bins AS r
    CROSS JOIN endpoints AS e
)
ORDER BY bin_index;

DROP TABLE IF EXISTS game_data_filtered.match_scaling_time_bins;

CREATE TABLE game_data_filtered.match_scaling_time_bins
(
    matchid String,
    game_minutes Float32,
    avg_completed_items Float32,
    lower_bin_index UInt8,
    lower_bin_label LowCardinality(String),
    lower_centroid_minutes Float32,
    upper_bin_index UInt8,
    upper_bin_label LowCardinality(String),
    upper_centroid_minutes Float32,
    lower_probability Float32,
    upper_probability Float32,
    sample_value Float32,
    assigned_bin_index UInt8,
    assigned_bin_label LowCardinality(String),
    assigned_from_minutes Float32,
    assigned_to_minutes Float32,
    assigned_centroid_minutes Float32
)
ENGINE = MergeTree
ORDER BY matchid;

INSERT INTO game_data_filtered.match_scaling_time_bins
WITH
candidate_bins AS (
    SELECT
        m.matchid,
        m.game_minutes,
        m.avg_completed_items,
        min(b.bin_index) AS first_bin_index,
        max(b.bin_index) AS last_bin_index,
        min(b.centroid_minutes) AS first_centroid_minutes,
        max(b.centroid_minutes) AS last_centroid_minutes,
        maxIf(b.bin_index, b.centroid_minutes <= m.game_minutes)
            AS lower_candidate_bin_index,
        minIf(b.bin_index, b.centroid_minutes >= m.game_minutes)
            AS upper_candidate_bin_index
    FROM game_data_filtered.scaling_match_item_times_stg AS m
    CROSS JOIN game_data_filtered.scaling_time_bins AS b
    GROUP BY
        m.matchid,
        m.game_minutes,
        m.avg_completed_items
),

assignment_bounds AS (
    SELECT
        matchid,
        game_minutes,
        avg_completed_items,
        multiIf(
            game_minutes <= first_centroid_minutes, first_bin_index,
            game_minutes >= last_centroid_minutes, last_bin_index,
            lower_candidate_bin_index
        ) AS lower_bin_index,
        multiIf(
            game_minutes <= first_centroid_minutes, first_bin_index,
            game_minutes >= last_centroid_minutes, last_bin_index,
            upper_candidate_bin_index
        ) AS upper_bin_index
    FROM candidate_bins
),

score_inputs AS (
    SELECT
        b.matchid,
        b.game_minutes,
        b.avg_completed_items,
        b.lower_bin_index,
        lower_bin.bin_label AS lower_bin_label,
        toFloat64(lower_bin.centroid_minutes) AS lower_centroid_minutes,
        b.upper_bin_index,
        upper_bin.bin_label AS upper_bin_label,
        toFloat64(upper_bin.centroid_minutes) AS upper_centroid_minutes,
        greatest(
            (upper_centroid_minutes - lower_centroid_minutes) / 3.0,
            0.25
        ) AS sigma,
        toFloat64(cityHash64(b.matchid) % 1000000) / 1000000.0 AS sample_value
    FROM assignment_bounds AS b
    INNER JOIN game_data_filtered.scaling_time_bins AS lower_bin
        ON b.lower_bin_index = lower_bin.bin_index
    INNER JOIN game_data_filtered.scaling_time_bins AS upper_bin
        ON b.upper_bin_index = upper_bin.bin_index
),

normal_weights AS (
    SELECT
        *,
        if(
            lower_bin_index = upper_bin_index,
            toFloat64(1),
            exp(
                -pow(game_minutes - lower_centroid_minutes, 2)
                / (2.0 * pow(sigma, 2))
            )
        ) AS lower_weight,
        if(
            lower_bin_index = upper_bin_index,
            toFloat64(0),
            exp(
                -pow(game_minutes - upper_centroid_minutes, 2)
                / (2.0 * pow(sigma, 2))
            )
        ) AS upper_weight
    FROM score_inputs
),

assignment_probabilities AS (
    SELECT
        *,
        if(
            lower_bin_index = upper_bin_index,
            toFloat64(1),
            lower_weight / (lower_weight + upper_weight)
        ) AS lower_probability,
        if(
            lower_bin_index = upper_bin_index,
            toFloat64(0),
            upper_weight / (lower_weight + upper_weight)
        ) AS upper_probability
    FROM normal_weights
),

assigned AS (
    SELECT
        *,
        if(
            lower_bin_index = upper_bin_index
            OR sample_value < lower_probability,
            lower_bin_index,
            upper_bin_index
        ) AS assigned_bin_index
    FROM assignment_probabilities
)

SELECT
    a.matchid,
    toFloat32(a.game_minutes) AS game_minutes,
    toFloat32(a.avg_completed_items) AS avg_completed_items,
    a.lower_bin_index,
    a.lower_bin_label,
    toFloat32(a.lower_centroid_minutes) AS lower_centroid_minutes,
    a.upper_bin_index,
    a.upper_bin_label,
    toFloat32(a.upper_centroid_minutes) AS upper_centroid_minutes,
    toFloat32(a.lower_probability) AS lower_probability,
    toFloat32(a.upper_probability) AS upper_probability,
    toFloat32(a.sample_value) AS sample_value,
    a.assigned_bin_index,
    assigned_bin.bin_label AS assigned_bin_label,
    assigned_bin.from_minutes AS assigned_from_minutes,
    assigned_bin.to_minutes AS assigned_to_minutes,
    assigned_bin.centroid_minutes AS assigned_centroid_minutes
FROM assigned AS a
INNER JOIN game_data_filtered.scaling_time_bins AS assigned_bin
    ON a.assigned_bin_index = assigned_bin.bin_index;

-- Calibration table for documentation.
SELECT
    bin_index,
    bin_label,
    item_count_from,
    item_count_to_exclusive,
    match_count,
    round(avg_game_minutes, 2) AS avg_game_minutes,
    round(from_minutes, 2) AS from_minutes,
    round(to_minutes, 2) AS to_minutes,
    round(centroid_minutes, 2) AS centroid_minutes
FROM game_data_filtered.scaling_time_bins
ORDER BY bin_index;

-- Build sanity checks.
SELECT
    count() AS bins,
    countIf(
        bin_index = 1
        AND abs(from_minutes - (
            SELECT min(game_minutes)
            FROM game_data_filtered.match_scaling_time_bins
        )) < 0.001
    ) AS bins_starting_at_min_game_length,
    countIf(to_minutes > from_minutes) AS bins_with_positive_width,
    countIf(centroid_minutes > from_minutes AND centroid_minutes < to_minutes)
        AS bins_with_internal_centroid
FROM game_data_filtered.scaling_time_bins;

SELECT
    count() AS assigned_rows,
    uniqExact(matchid) AS distinct_matches,
    count() - uniqExact(matchid) AS duplicate_match_rows,
    countIf(
        lower_probability < 0
        OR lower_probability > 1
        OR upper_probability < 0
        OR upper_probability > 1
    ) AS invalid_probability_rows,
    max(abs((lower_probability + upper_probability) - 1.0))
        AS max_probability_sum_error
FROM game_data_filtered.match_scaling_time_bins;

DROP TABLE IF EXISTS game_data_filtered.scaling_role_build_profiles_stg;

CREATE TABLE game_data_filtered.scaling_role_build_profiles_stg
ENGINE = MergeTree
ORDER BY (championid, teamposition, build)
AS
WITH
role_build_bin_rows AS (
    SELECT
        assumeNotNull(ps.championid) AS championid,
        dictGetOrDefault(
            'game_data.championid_name_map_dict',
            'name',
            toInt32(championid),
            ''
        ) AS championname,
        toString(ps.teamposition) AS teamposition,
        toString(ivt.highest_value_label) AS build,
        mt.assigned_bin_index AS bin_index,
        count() AS games,
        sum(toUInt64(ps.win)) AS wins
    FROM game_data_filtered.participant_stats AS ps
    INNER JOIN game_data_filtered.participant_item_value_totals AS ivt
        ON
            ps.matchid = ivt.matchid
            AND ps.participantid = ivt.participantid
    INNER JOIN game_data_filtered.match_scaling_time_bins AS mt
        ON ps.matchid = mt.matchid
    WHERE
        isNotNull(ps.championid)
        AND toString(ivt.highest_value_label) != 'none'
    GROUP BY
        championid,
        championname,
        teamposition,
        build,
        bin_index
)

SELECT
    championid,
    championname,
    teamposition,
    build,
    bin_1_games,
    bin_1_wins,
    bin_2_games,
    bin_2_wins,
    bin_4_games,
    bin_4_wins,
    bin_1_games + bin_2_games AS early_games,
    bin_1_wins + bin_2_wins AS early_wins
FROM (
    SELECT
        championid,
        championname,
        teamposition,
        build,
        sumIf(games, bin_index = 1) AS bin_1_games,
        sumIf(wins, bin_index = 1) AS bin_1_wins,
        sumIf(games, bin_index = 2) AS bin_2_games,
        sumIf(wins, bin_index = 2) AS bin_2_wins,
        sumIf(games, bin_index = 4) AS bin_4_games,
        sumIf(wins, bin_index = 4) AS bin_4_wins
    FROM role_build_bin_rows
    GROUP BY
        championid,
        championname,
        teamposition,
        build
);

-- Champion/build examples that skew early: bins 1-2 beat bin 4.
SELECT
    championid,
    championname,
    teamposition,
    build,
    early_games,
    round(100.0 * early_wins / early_games, 2) AS early_wr,
    bin_4_games AS late_games,
    round(100.0 * bin_4_wins / bin_4_games, 2) AS late_wr,
    round(
        100.0 * ((early_wins / early_games) - (bin_4_wins / bin_4_games)),
        2
    ) AS early_minus_late_pp
FROM game_data_filtered.scaling_role_build_profiles_stg
WHERE
    early_games >= 500
    AND bin_4_games >= 200
    AND early_wins / early_games > bin_4_wins / bin_4_games
ORDER BY early_minus_late_pp DESC
LIMIT 10;

-- Champion/build examples that skew late: bin 4 beats bins 1-2.
SELECT
    championid,
    championname,
    teamposition,
    build,
    early_games,
    round(100.0 * early_wins / early_games, 2) AS early_wr,
    bin_4_games AS late_games,
    round(100.0 * bin_4_wins / bin_4_games, 2) AS late_wr,
    round(
        100.0 * ((bin_4_wins / bin_4_games) - (early_wins / early_games)),
        2
    ) AS late_minus_early_pp
FROM game_data_filtered.scaling_role_build_profiles_stg
WHERE
    early_games >= 500
    AND bin_4_games >= 200
    AND bin_4_wins / bin_4_games > early_wins / early_games
ORDER BY late_minus_early_pp DESC
LIMIT 10;

DROP TABLE IF EXISTS game_data_filtered.scaling_role_build_profiles_stg;

DROP TABLE IF EXISTS game_data_filtered.scaling_match_item_times_stg;
