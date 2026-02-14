CREATE TABLE IF NOT EXISTS game_data.matchids
(
    run_id UUID,
    matchid String CODEC (ZSTD(3))
)
ENGINE = MergeTree
ORDER BY (run_id, matchid);
