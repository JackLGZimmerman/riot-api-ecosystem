CREATE TABLE IF NOT EXISTS game_data.data_timestamps
(
    name String,
    run_id Uuid,
    stored_at Uint32
)
ENGINE = MERGETREE
ORDER BY (name, run_id);
