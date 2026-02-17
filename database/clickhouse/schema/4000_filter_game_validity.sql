CREATE TABLE IF NOT EXISTS game_data.filter_game_validity
(
    gameid UInt64,
    rule_mask UInt32,
    is_valid UInt8
)
ENGINE = MergeTree
ORDER BY (gameid);
