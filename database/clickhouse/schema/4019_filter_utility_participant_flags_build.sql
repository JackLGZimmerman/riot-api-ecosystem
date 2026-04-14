TRUNCATE TABLE game_data.filter_utility_participant_flags;

INSERT INTO game_data.filter_utility_participant_flags
(
    matchid,
    teamid,
    participantid,
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
    game_time_lte_18
)
SELECT
    ps.matchid,
    ps.teamid,
    ps.participantid,
    (ps.kills + ps.assists) * 5 < ps.deaths
        AS player_low_kda,
    ps.goldearned > 0 AND ps.goldspent * 100 < ps.goldearned * 60
        AS player_gold_spent,
    ps.kills + ps.assists = 0 AND ps.deaths > 4 AS no_contribution_kda,
    ps.summoner1casts = 0 OR ps.summoner2casts = 0 AS bad_summoner_usage,
    pl.wins + pl.losses > 40
    AND pl.wins * 100 > (pl.wins + pl.losses) * 70 AS player_high_winrate,
    tf.team_kills_to_deaths,
    tf.team_kills > 0 AND ps.kills * 100 > tf.team_kills * 65 AS solo_carried,
    (
        ps.teamposition != 'UTILITY'
        AND tf.team_damage_to_champions > 0
        AND ps.totaldamagedealttochampions * 1000 < tf.team_damage_to_champions * 75
    ) AS too_little_damage,
    (
        ps.teamposition != 'UTILITY'
        AND ps.timeplayed > 0
        AND (ps.totalminionskilled + ps.neutralminionskilled) * 60.0 / ps.timeplayed
        < 4.5
    ) AS low_minions_killed,
    tf.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    tf.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy,
    (
        ps.item0 = 0
        AND ps.item1 = 0
        AND ps.item2 = 0
        AND ps.item3 = 0
        AND ps.item4 = 0
        AND ps.item5 = 0
    ) AS sold_all_items,
    (
        ps.item0 = ps.item1
        AND ps.item1 = ps.item2
        AND ps.item2 = ps.item3
        AND ps.item3 = ps.item4
        AND ps.item4 = ps.item5
    ) AS grief_build,
    gf.game_time_lte_18
FROM game_data.participant_stats AS ps
ANY LEFT JOIN game_data.filter_utility_players_latest AS pl
    ON ps.puuid = pl.puuid
ANY LEFT JOIN game_data.filter_utility_team_flags AS tf
    ON
        ps.matchid = tf.matchid
        AND ps.teamid = tf.teamid
ANY LEFT JOIN game_data.filter_utility_game_filters AS gf
    ON ps.matchid = gf.matchid;
