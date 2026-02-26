CREATE TABLE IF NOT EXISTS game_data.tl_champion_special_kill
(
    run_id UUID,
    matchid UInt64,
    frame_timestamp UInt32,
    timestamp UInt64,
    type LowCardinality (String),
    killtype String,
    killerid Int8,
    position_x Int16,
    position_y Int16,
    multikilllength Nullable (UInt8)
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
