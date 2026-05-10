-- noqa: disable=LT01,LT05,PRS

DROP TABLE IF EXISTS game_data_filtered.matchup_2v2;

CREATE TABLE IF NOT EXISTS game_data_filtered.matchup_2v2
(
    split LowCardinality(String),
    left_championid_1 Int32,
    left_championname_1 LowCardinality(String),
    left_teamposition_1 LowCardinality(String),
    left_build_1 LowCardinality(String),
    left_championid_2 Int32,
    left_championname_2 LowCardinality(String),
    left_teamposition_2 LowCardinality(String),
    left_build_2 LowCardinality(String),
    right_championid_1 Int32,
    right_championname_1 LowCardinality(String),
    right_teamposition_1 LowCardinality(String),
    right_build_1 LowCardinality(String),
    right_championid_2 Int32,
    right_championname_2 LowCardinality(String),
    right_teamposition_2 LowCardinality(String),
    right_build_2 LowCardinality(String),
    matchups UInt64,
    left_wins UInt64,
    right_wins UInt64,
    left_win_rate Float32,
    right_win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    left_championid_1, left_teamposition_1, left_build_1,
    left_championid_2, left_teamposition_2, left_build_2,
    right_championid_1, right_teamposition_1, right_build_1,
    right_championid_2, right_teamposition_2, right_build_2
);
