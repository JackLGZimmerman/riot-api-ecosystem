CREATE TABLE IF NOT EXISTS game_data.tl_champion_special_kill
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    type LowCardinality (String),
    killtype String,
    killerid Int8,
    position_x Int16,
    position_y Int16,
    multikilllength Nullable (UInt8)
)
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, killtype, killerid);
