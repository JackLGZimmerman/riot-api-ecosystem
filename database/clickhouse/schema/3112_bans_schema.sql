CREATE TABLE IF NOT EXISTS game_data.bans
(
    run_id UUID,
    matchid String CODEC (ZSTD(3)),
    teamid UInt8,
    pickturn UInt8,
    championid Nullable (Int32)
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, pickturn);
