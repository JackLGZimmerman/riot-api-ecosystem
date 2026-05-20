CREATE TABLE IF NOT EXISTS game_data.tl_skill_level_up
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    participantid UInt8,
    skillslot UInt8,
    leveluptype LowCardinality (String)
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid, skillslot, leveluptype);
