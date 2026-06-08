CREATE TABLE IF NOT EXISTS game_data.matchdata_matchids (
    run_id UUID,
    matchid String CODEC (ZSTD(3))
)
ENGINE = ReplacingMergeTree
ORDER BY (matchid);
