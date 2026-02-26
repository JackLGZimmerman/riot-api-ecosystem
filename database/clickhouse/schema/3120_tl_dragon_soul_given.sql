CREATE TABLE IF NOT EXISTS game_data.tl_dragon_soul_given
(
    run_id UUID,
    matchid UInt64,
    frame_timestamp UInt32,
    timestamp UInt64,
    type LowCardinality (String),
    name LowCardinality (String),
    teamid UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
