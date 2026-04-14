CREATE TABLE IF NOT EXISTS game_data_filtered.valid_game_ids
(
    matchid String
)
ENGINE = MergeTree
ORDER BY matchid;
