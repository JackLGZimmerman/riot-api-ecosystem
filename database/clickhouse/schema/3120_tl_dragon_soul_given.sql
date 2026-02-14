CREATE TABLE IF NOT EXISTS game_data.tl_dragon_soul_given
(
    run_id UUID,
    gameid UInt64,
    frame_timestamp UInt32,
    timestamp UInt64,
    type LowCardinality (String),
    name LowCardinality (String),
    teamid Enum8 ('100' = 1, '200' = 2)
)
ENGINE = MergeTree
ORDER BY (gameid, frame_timestamp, timestamp, run_id);
