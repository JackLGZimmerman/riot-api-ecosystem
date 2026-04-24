CREATE TABLE IF NOT EXISTS game_data.tl_pause_end
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    realtimestamp UInt64
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
