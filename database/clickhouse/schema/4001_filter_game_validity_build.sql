TRUNCATE TABLE game_data.filter_game_validity;

INSERT INTO game_data.filter_game_validity
(
    matchid,
    teamid,
    participantid,
    player_rule_mask,
    team_rule_mask,
    game_rule_mask,
    rule_mask,
    is_valid
)
SELECT
    ps.matchid,
    ps.teamid,
    ps.participantid,
    pf.player_low_kda * 1
    + pf.player_gold_spent * 2
    + pf.no_contribution_kda * 4
    + pf.bad_summoner_usage * 8
    + pf.player_high_winrate * 16
    + pf.solo_carried * 64
    + pf.too_little_damage * 128
    + pf.low_minions_killed * 256
    + pf.sold_all_items * 2048
    + pf.grief_build * 4096 AS player_rule_mask,
    pf.team_kills_to_deaths * 32
    + pf.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy * 512
    + pf.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy * 1024
        AS team_rule_mask,
    pf.game_time_lte_18 * 8192
    + rrm.has_rare_role * 16384 AS game_rule_mask,
    gf.player_low_kda * 1
    + gf.player_gold_spent * 2
    + gf.no_contribution_kda * 4
    + gf.bad_summoner_usage * 8
    + gf.player_high_winrate * 16
    + gf.team_kills_to_deaths * 32
    + gf.solo_carried * 64
    + gf.too_little_damage * 128
    + gf.low_minions_killed * 256
    + gf.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy * 512
    + gf.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy * 1024
    + gf.sold_all_items * 2048
    + gf.grief_build * 4096
    + gf.game_time_lte_18 * 8192
    + gf.low_champion_teamposition_history * 16384 AS rule_mask,
    gf.any_filter_triggered = 0 AS is_valid
FROM game_data.participant_stats AS ps
INNER JOIN game_data.filter_utility_game_flags AS gf
    ON ps.matchid = gf.matchid
ANY LEFT JOIN game_data.filter_utility_participant_flags AS pf
    ON
        ps.matchid = pf.matchid
        AND ps.teamid = pf.teamid
        AND ps.participantid = pf.participantid
ANY LEFT JOIN game_data.filter_utility_rare_role_matchids AS rrm
    ON ps.matchid = rrm.matchid;
