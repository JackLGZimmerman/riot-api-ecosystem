-- noqa: disable=LT01,LT05,PRS
-- Persistent per-patch chronological 80/20 train/test split labels for ML and
-- leakage-safe aggregate builds. The build file orders current valid_game_ids
-- with source game_data.info because game_data_filtered.info is no longer
-- mirrored.

DROP TABLE IF EXISTS game_data_filtered.ml_game_split;

CREATE TABLE IF NOT EXISTS game_data_filtered.ml_game_split
(
    matchid String,
    split LowCardinality(String),
)
ENGINE = MergeTree
ORDER BY (split, matchid);
