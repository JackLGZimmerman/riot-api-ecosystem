DROP TABLE IF EXISTS game_data.filter_utility_game_flags;

CREATE TABLE game_data.filter_utility_game_flags
(
    matchid String,
    player_low_kda UInt8,
    player_gold_spent UInt8,
    no_contribution_kda UInt8,
    bad_summoner_usage UInt8,
    player_high_winrate UInt8,
    team_kills_to_deaths UInt8,
    solo_carried UInt8,
    too_little_damage UInt8,
    low_minions_killed UInt8,
    team_non_utility_avg_cs_per_min_gt_2_5_below_enemy UInt8,
    team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy UInt8,
    sold_all_items UInt8,
    grief_build UInt8,
    game_time_lte_18 UInt8,
    low_champion_teamposition_history UInt8,
    any_filter_triggered UInt8
)
ENGINE = MergeTree
ORDER BY matchid;
