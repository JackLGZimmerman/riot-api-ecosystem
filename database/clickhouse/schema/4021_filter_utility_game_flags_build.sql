TRUNCATE TABLE game_data.filter_utility_game_flags;

INSERT INTO game_data.filter_utility_game_flags
(
    matchid,
    player_low_kda,
    player_gold_spent,
    no_contribution_kda,
    bad_summoner_usage,
    player_high_winrate,
    team_kills_to_deaths,
    solo_carried,
    too_little_damage,
    low_minions_killed,
    team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy,
    sold_all_items,
    grief_build,
    game_time_lte_18,
    low_champion_teamposition_history,
    any_filter_triggered
)
SELECT
    p.matchid,
    max(p.player_low_kda) AS player_low_kda,
    max(p.player_gold_spent) AS player_gold_spent,
    max(p.no_contribution_kda) AS no_contribution_kda,
    max(p.bad_summoner_usage) AS bad_summoner_usage,
    max(p.player_high_winrate) AS player_high_winrate,
    max(p.team_kills_to_deaths) AS team_kills_to_deaths,
    max(p.solo_carried) AS solo_carried,
    max(p.too_little_damage) AS too_little_damage,
    max(p.low_minions_killed) AS low_minions_killed,
    max(p.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy)
        AS team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    max(p.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy)
        AS team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy,
    max(p.sold_all_items) AS sold_all_items,
    max(p.grief_build) AS grief_build,
    max(p.game_time_lte_18) AS game_time_lte_18,
    max(rrm.has_rare_role) AS low_champion_teamposition_history,
    (
        max(p.player_low_kda)
        OR max(p.player_gold_spent)
        OR max(p.no_contribution_kda)
        OR max(p.bad_summoner_usage)
        OR max(p.player_high_winrate)
        OR max(p.team_kills_to_deaths)
        OR max(p.solo_carried)
        OR max(p.too_little_damage)
        OR max(p.low_minions_killed)
        OR max(p.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy)
        OR max(p.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy)
        OR max(p.sold_all_items)
        OR max(p.grief_build)
        OR max(p.game_time_lte_18)
        OR max(rrm.has_rare_role)
    ) AS any_filter_triggered
FROM game_data.filter_utility_participant_flags AS p
LEFT JOIN game_data.filter_utility_rare_role_matchids AS rrm
    ON p.matchid = rrm.matchid
GROUP BY p.matchid;
