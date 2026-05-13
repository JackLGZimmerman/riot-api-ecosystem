CREATE TABLE IF NOT EXISTS game_data.tl_ward_kill
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    killerid Int8,
    wardtype LowCardinality (String)
)
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, killerid, wardtype);
