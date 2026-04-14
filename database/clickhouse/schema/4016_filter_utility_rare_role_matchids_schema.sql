DROP TABLE IF EXISTS game_data.filter_utility_rare_role_matchids;

CREATE TABLE game_data.filter_utility_rare_role_matchids
(
    matchid String,
    has_rare_role UInt8
)
ENGINE = MergeTree
ORDER BY matchid;
