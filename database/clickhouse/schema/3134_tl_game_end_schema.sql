CREATE TABLE IF NOT EXISTS game_data.tl_game_end
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    winningteam UInt8,
    gameid Nullable (UInt64),
    realtimestamp UInt64
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, winningteam, realtimestamp);
