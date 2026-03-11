CREATE TABLE IF NOT EXISTS game_data.matchids
(
    run_id UUID,
    matchid String CODEC (ZSTD(3)),
    queue_type LowCardinality(String)
)
ENGINE = MergeTree
ORDER BY (run_id, matchid, queue_type);
