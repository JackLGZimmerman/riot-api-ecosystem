CREATE TABLE IF NOT EXISTS game_data.tl_feat_update
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    teamid UInt8,
    feattype UInt8,
    featvalue Int32
)
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, teamid, feattype, featvalue);
