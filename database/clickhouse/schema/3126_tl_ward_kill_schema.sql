CREATE TABLE IF NOT EXISTS game_data.tl_ward_kill
(
    run_id UUID,
    matchid String CODEC (ZSTD(3)),
    frame_timestamp UInt32,
    timestamp UInt64,
    killerid Int8,
    wardtype LowCardinality (String)
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, killerid);
