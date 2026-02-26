CREATE TABLE IF NOT EXISTS game_data.tl_elite_monster_kill
(
    run_id UUID,
    matchid UInt64,
    frame_timestamp UInt32,
    timestamp UInt64,
    type LowCardinality (String),
    assistingparticipantids Array (UInt8),
    bounty UInt16,
    killerid Int8,
    killerteamid Int8,
    monstersubtype LowCardinality (Nullable (String)),
    monstertype LowCardinality (String),
    position_x Int16,
    position_y Int16
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
