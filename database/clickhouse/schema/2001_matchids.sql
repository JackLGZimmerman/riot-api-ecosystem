CREATE TABLE IF NOT EXISTS game_data.matchids
(
    run_id UUID,
    matchid STRING CODEC (ZSTD(3))
)
ENGINE = MERGETREE
ORDER BY (run_id, matchid);
