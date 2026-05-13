CREATE TABLE IF NOT EXISTS game_data.tl_item_purchased
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    participantid UInt8,
    itemid UInt32
)
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid, itemid);
