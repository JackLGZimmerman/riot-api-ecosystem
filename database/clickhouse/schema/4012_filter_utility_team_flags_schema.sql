DROP TABLE IF EXISTS game_data.filter_utility_team_flags;

CREATE TABLE game_data.filter_utility_team_flags
(
    matchid String,
    teamid UInt8,
    team_kills UInt16,
    team_damage_to_champions UInt32,
    team_kills_to_deaths UInt8,
    team_non_utility_avg_cs_per_min_gt_2_5_below_enemy UInt8,
    team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, teamid);
