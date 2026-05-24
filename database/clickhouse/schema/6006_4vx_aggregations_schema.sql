-- noqa: disable=LT01,LT05,PRS

DROP TABLE IF EXISTS game_data_filtered.synergy_4vx;

CREATE TABLE IF NOT EXISTS game_data_filtered.synergy_4vx
(
    split LowCardinality(String),
    championid_1 Int32,
    championname_1 LowCardinality(String),
    teamposition_1 LowCardinality(String),
    build_1 LowCardinality(String),
    championid_2 Int32,
    championname_2 LowCardinality(String),
    teamposition_2 LowCardinality(String),
    build_2 LowCardinality(String),
    championid_3 Int32,
    championname_3 LowCardinality(String),
    teamposition_3 LowCardinality(String),
    build_3 LowCardinality(String),
    championid_4 Int32,
    championname_4 LowCardinality(String),
    teamposition_4 LowCardinality(String),
    build_4 LowCardinality(String),
    matchups UInt64,
    wins UInt64,
    losses UInt64,
    win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    championid_1, teamposition_1, build_1,
    championid_2, teamposition_2, build_2,
    championid_3, teamposition_3, build_3,
    championid_4, teamposition_4, build_4
);
