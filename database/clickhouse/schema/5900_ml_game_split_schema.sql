-- noqa: disable=LT01,LT05,PRS
-- Persistent 80/10/10 chronological split labels for ML and leakage-safe
-- aggregate builds. The build file populates this from filtered participant
-- rows, using game_data_filtered.info only for ordering timestamps.

DROP TABLE IF EXISTS game_data_filtered.ml_game_split;

CREATE TABLE IF NOT EXISTS game_data_filtered.ml_game_split
(
    matchid String,
    split LowCardinality(String),
    split_index UInt64,
    total_games UInt64,
    gamestarttimestamp UInt64,
    gamecreation UInt64,
    participant_count UInt8
)
ENGINE = MergeTree
ORDER BY (split, split_index, matchid);
