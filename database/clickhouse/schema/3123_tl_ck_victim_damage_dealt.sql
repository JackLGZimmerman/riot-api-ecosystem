CREATE TABLE IF NOT EXISTS game_data.tl_ck_victim_damage_dealt
(
    run_id UUID,
    basic Bool,
    magicdamage UInt16,
    name LowCardinality (String),
    participantid Enum8 (
        '1' = 1,
        '2' = 2,
        '3' = 3,
        '4' = 4,
        '5' = 5,
        '6' = 6,
        '7' = 7,
        '8' = 8,
        '9' = 9,
        '10' = 10
    ),
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
    idx UInt32
)
ENGINE = MergeTree
ORDER BY (gameid, frame_timestamp, timestamp, champion_kill_event_id, idx, run_id);
