-- noqa: disable=LT01,LT05,PRS

DROP TABLE IF EXISTS game_data_filtered.matchup_1v1;

CREATE TABLE IF NOT EXISTS game_data_filtered.matchup_1v1
(
    split LowCardinality(String),
    left_championid Int32,
    left_championname LowCardinality(String),
    left_teamposition LowCardinality(String),
    left_build LowCardinality(String),
    right_championid Int32,
    right_championname LowCardinality(String),
    right_teamposition LowCardinality(String),
    right_build LowCardinality(String),
    matchups UInt64,
    left_wins UInt64,
    right_wins UInt64,
    left_win_rate Float32,
    right_win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    left_championid, left_teamposition, left_build,
    right_championid, right_teamposition, right_build
);
