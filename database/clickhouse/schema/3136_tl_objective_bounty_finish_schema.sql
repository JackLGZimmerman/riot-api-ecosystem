CREATE TABLE IF NOT EXISTS game_data.tl_objective_bounty_finish
(
    run_id UUID,
    matchid String,
    frame_timestamp UInt32,
    timestamp UInt64,
    teamid UInt8
)
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, teamid);
