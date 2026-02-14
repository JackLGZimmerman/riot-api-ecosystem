CREATE TABLE IF NOT EXISTS game_data.data_timestamps
(
    name String,
    run_id UUID,
    stored_at UInt32
)
ENGINE = MergeTree
ORDER BY (name, run_id);
