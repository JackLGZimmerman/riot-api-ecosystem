CREATE TABLE IF NOT EXISTS game_data.tl_level_up
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    participantid UInt8,
    level UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
