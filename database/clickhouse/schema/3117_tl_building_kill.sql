CREATE TABLE IF NOT EXISTS game_data.tl_building_kill
(
    run_id UUID,
    gameid UInt64,
    frame_timestamp UInt32,
    timestamp UInt64,
    type LowCardinality (String),
    bounty UInt16,
    buildingtype LowCardinality (String),
    killerid Int8,
    lanetype LowCardinality (String),
    position_x Int16,
    position_y Int16,
    teamid Enum8 ('100' = 1, '200' = 2),
    towertype LowCardinality (String)
)
ENGINE = MergeTree
ORDER BY (gameid, frame_timestamp, timestamp, run_id);
