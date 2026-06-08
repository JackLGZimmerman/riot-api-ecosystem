CREATE TABLE IF NOT EXISTS game_data.tl_champion_transform
(
    run_id UUID,
    matchid String CODEC (ZSTD(3)),
    frame_timestamp UInt32,
    timestamp UInt64,
    participantid UInt8,
    transformtype LowCardinality (String)
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid);
