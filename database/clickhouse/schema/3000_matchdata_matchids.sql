CREATE TABLE IF NOT EXISTS game_data.matchdata_matchids (
    run_id UUID,
    matchid String,
    status LowCardinality(String) DEFAULT 'pending',
    last_error String DEFAULT ''
)
ENGINE = MergeTree
ORDER BY (status, matchid, run_id)
