CREATE TABLE IF NOT EXISTS game_data.info
(
    run_id UUID,
    endofgameresult LowCardinality (String),
    gamecreation UInt64,
    gameduration UInt16,
    gameendtimestamp UInt64,
    matchid UInt64,
    gamestarttimestamp UInt64,
    gametype LowCardinality (String),
    gameversion LowCardinality (String),
    season UInt8,
    patch UInt8,
    subversion String,
    mapid UInt8,
    platformid LowCardinality (String),
    queueid Int16
)
ENGINE = MergeTree
ORDER BY (matchid, run_id);
