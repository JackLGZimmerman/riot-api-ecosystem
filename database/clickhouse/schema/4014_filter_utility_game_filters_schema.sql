DROP TABLE IF EXISTS game_data.filter_utility_game_filters;

CREATE TABLE game_data.filter_utility_game_filters
(
    matchid String,
    game_time_lte_18 UInt8
)
ENGINE = MergeTree
ORDER BY matchid;
