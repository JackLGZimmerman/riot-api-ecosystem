CREATE TABLE IF NOT EXISTS game_data.tl_elite_monster_kill
(
    run_id UUID,
    gameId UInt64,
    frame_timestamp UInt32,
    timestamp UInt64,
    type LowCardinality(String),
    assistingParticipantIds Array(UInt8),
    bounty UInt16,
    killerId Int8,
    killerTeamId Enum('100'=1, '200'=2),
    monsterSubType LowCardinality(String),
    monsterType LowCardinality(String),
    position_x Int16,
    position_y Int16
)
ENGINE = MergeTree
ORDER BY (gameId, frame_timestamp, timestamp, run_id);
