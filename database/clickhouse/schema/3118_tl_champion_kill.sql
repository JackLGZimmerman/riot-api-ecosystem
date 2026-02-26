CREATE TABLE IF NOT EXISTS game_data.tl_champion_kill
(
    run_id UUID,
    matchid UInt64,
    frame_timestamp UInt32,
    timestamp UInt64,
    type LowCardinality (String),
    champion_kill_event_id String,
    killerid Int8,
    victimid Int8,
    bounty UInt16,
    killstreaklength UInt8,
    shutdownbounty UInt16,
    position_x Int16,
    position_y Int16
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, champion_kill_event_id, run_id);
