CREATE TABLE IF NOT EXISTS game_data_filtered.matchup_1v1
(
    matchid String,
    left_teamid UInt8,
    right_teamid UInt8,
    left_champion Int32,
    right_champion Int32,
    left_team_position LowCardinality (String),
    right_team_position LowCardinality (String),
    left_build String,
    right_build String,
    left_win UInt8,
    right_win UInt8,
    left_metrics Map (String, Int64),
    right_metrics Map (String, Int64),
    left_flags Map (String, UInt8),
    right_flags Map (String, UInt8)
)
ENGINE = MergeTree
ORDER BY (left_champion, right_champion, matchid);
