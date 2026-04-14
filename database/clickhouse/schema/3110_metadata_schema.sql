CREATE TABLE IF NOT EXISTS game_data.metadata
(
    run_id UUID,
    matchid String,
    dataversion UInt8,
    participants Array (FixedString (78))
)
ENGINE = MergeTree
ORDER BY (matchid, run_id);
