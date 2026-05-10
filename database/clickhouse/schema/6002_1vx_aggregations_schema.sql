-- noqa: disable=LT01,LT05,PRS

DROP TABLE IF EXISTS game_data_filtered.synergy_1vx;

CREATE TABLE IF NOT EXISTS game_data_filtered.synergy_1vx
(
    split LowCardinality(String),
    championid Int32,
    championname LowCardinality(String),
    teamposition LowCardinality(String),
    build LowCardinality(String),
    matchups UInt64,
    wins UInt64,
    losses UInt64,
    win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    championid, teamposition, build
);
