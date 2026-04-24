CREATE TABLE IF NOT EXISTS game_data.tl_ward_placed
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    creatorid UInt8,
    wardtype LowCardinality (String)
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
