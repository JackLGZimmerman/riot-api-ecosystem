CREATE TABLE IF NOT EXISTS game_data_filtered.valid_game_ids
(
    matchid UInt64
)
ENGINE = MergeTree
ORDER BY matchid;
