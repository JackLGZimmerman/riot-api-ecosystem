CREATE TABLE IF NOT EXISTS game_data.tl_feat_update
(
    run_id UUID,
    matchid String CODEC (ZSTD(3)),
    frame_timestamp UInt32,
    timestamp UInt64,
    teamid UInt8,
    feattype UInt8,
    featvalue Int32
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, teamid);
