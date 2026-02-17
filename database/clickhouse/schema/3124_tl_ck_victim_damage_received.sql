CREATE TABLE IF NOT EXISTS game_data.tl_ck_victim_damage_received
(
    run_id UUID,
    basic UInt8,
    magicdamage UInt16,
    name LowCardinality (String),
    participantid UInt8,
    physicaldamage UInt16,
    spellname LowCardinality (String),
    spellslot Int8,
    truedamage UInt32,
    type String,
    gameid UInt64,
    frame_timestamp UInt32,
    timestamp UInt64,
    champion_kill_event_id String,
    direction LowCardinality (String),
    idx UInt8
)
ENGINE = MergeTree
ORDER BY (gameid, frame_timestamp, timestamp, champion_kill_event_id, idx, run_id);
