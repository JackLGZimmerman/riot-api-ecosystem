CREATE DATABASE IF NOT EXISTS game_data;

CREATE TABLE IF NOT EXISTS game_data.data_timestamps
(
    name String,
    stored_at UInt32
)
ENGINE = ReplacingMergeTree(stored_at)
ORDER BY name;