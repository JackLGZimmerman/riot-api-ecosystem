CREATE TABLE IF NOT EXISTS game_data.tl_item_destroyed
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    participantid UInt8,
    itemid UInt32
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
