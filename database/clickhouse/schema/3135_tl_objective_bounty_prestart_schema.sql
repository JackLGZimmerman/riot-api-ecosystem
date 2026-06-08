CREATE TABLE IF NOT EXISTS game_data.tl_objective_bounty_prestart
(
    run_id UUID,
    matchid String CODEC (ZSTD(3)),
    frame_timestamp UInt32,
    timestamp UInt64,
    teamid UInt8,
    actualstarttime UInt64
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, teamid);
