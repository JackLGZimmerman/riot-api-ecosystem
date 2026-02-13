CREATE TABLE IF NOT EXISTS 'game_data.matchdata_matchids' (
    run_id UUID,
    match_id String
)
ENGINE = MergeTree
ORDER BY run_id
