CREATE TABLE IF NOT EXISTS game_data.info
(
    run_id UUID,
    endofgameresult LowCardinality (String),
    gamecreation UInt32,
    gameduration UInt16,
    gameendtimestamp UInt64,
    gameid UInt64,
    gamestarttimestamp UInt64,
    gametype LowCardinality (String),
    gameversion LowCardinality (String),
    season UInt8,
    patch UInt8,
    subversion UInt8,
    mapid UInt8,
    platformid LowCardinality (String),
    queueid UInt8
)
ENGINE = MergeTree
ORDER BY (gameid, run_id);
