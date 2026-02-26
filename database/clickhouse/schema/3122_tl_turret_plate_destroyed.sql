CREATE TABLE IF NOT EXISTS game_data.tl_turret_plate_destroyed
(
    run_id UUID,
    matchid UInt64,
    frame_timestamp UInt32,
    timestamp UInt64,
    type LowCardinality (String),
    killerid Int8,
    lanetype LowCardinality (String),
    position_x Int16,
    position_y Int16,
    teamid UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
