CREATE TABLE IF NOT EXISTS game_data.tl_item_undo
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    participantid UInt8,
    beforeid UInt32,
    afterid UInt32,
    goldgain Int32
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid, beforeid, afterid);
