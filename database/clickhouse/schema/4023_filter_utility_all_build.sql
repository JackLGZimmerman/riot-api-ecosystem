-- Utility stage rebuild entrypoint for the ad hoc filter reporting pipeline.
-- Run with clickhouse-client --multiquery after 4022_filter_utility_all_schema.sql.

TRUNCATE TABLE game_data.filter_utility_players_latest;

INSERT INTO game_data.filter_utility_players_latest
(
    puuid,
    wins,
    losses
)
SELECT
    puuid,
    argMax(wins, updated_at) AS wins,
    argMax(losses, updated_at) AS losses
FROM game_data.players
GROUP BY puuid;

TRUNCATE TABLE game_data.filter_utility_team_flags;

INSERT INTO game_data.filter_utility_team_flags
(
    matchid,
    teamid,
    team_kills,
    team_damage_to_champions,
    team_kills_to_deaths,
    team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy
)
WITH
team_base AS (
    SELECT
        matchid,
        teamid,
        sum(kills) AS team_kills,
        sum(deaths) AS team_deaths,
        sum(totaldamagedealttochampions) AS team_damage_to_champions,
        avgIf(
            (totalminionskilled + neutralminionskilled) * 60.0 / timeplayed,
            teamposition != 'UTILITY' AND timeplayed > 0
        ) AS team_non_utility_avg_cs_per_min,
        sumIf(
            totaldamagedealttochampions,
            teamposition != 'UTILITY'
        ) AS team_non_utility_damage_to_champions
    FROM game_data.participant_stats
    GROUP BY
        matchid,
        teamid
)

SELECT
    tb.matchid,
    tb.teamid,
    tb.team_kills,
    tb.team_damage_to_champions,
    tb.team_kills * 3 < tb.team_deaths AS team_kills_to_deaths,
    enemy.team_non_utility_avg_cs_per_min - tb.team_non_utility_avg_cs_per_min > 2.5
        AS team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    tb.team_non_utility_damage_to_champions
    / enemy.team_non_utility_damage_to_champions < (1.0 / 3.0)
        AS team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy
FROM team_base AS tb
LEFT JOIN team_base AS enemy
    ON
        tb.matchid = enemy.matchid
        AND tb.teamid != enemy.teamid;

TRUNCATE TABLE game_data.filter_utility_game_filters;

INSERT INTO game_data.filter_utility_game_filters
(
    matchid,
    game_time_lte_18
)
SELECT
    matchid,
    max(timeplayed) <= 18 * 60 AS game_time_lte_18
FROM game_data.participant_stats
GROUP BY matchid;

TRUNCATE TABLE game_data.filter_utility_rare_role_matchids;

INSERT INTO game_data.filter_utility_rare_role_matchids
(
    matchid,
    has_rare_role
)
WITH
champion_teamposition_pick_counts AS (
    SELECT
        championid,
        teamposition,
        count() AS champion_teamposition_picks
    FROM game_data.participant_stats
    WHERE
        championid IS NOT NULL
        AND teamposition != 'UNKNOWN'
    GROUP BY
        championid,
        teamposition
),

champion_pick_totals AS (
    SELECT
        championid,
        sum(champion_teamposition_picks) AS champion_picks
    FROM champion_teamposition_pick_counts
    GROUP BY championid
),

rare_champion_teampositions AS (
    SELECT
        ctpc.championid,
        ctpc.teamposition
    FROM champion_teamposition_pick_counts AS ctpc
    INNER JOIN champion_pick_totals AS cpt
        USING (championid)
    WHERE ctpc.champion_teamposition_picks * 1000 < cpt.champion_picks * 6
),

player_rare_champion_teamposition_pick_counts AS (
    SELECT
        ps.puuid,
        ps.championid,
        ps.teamposition,
        count() AS player_champion_teamposition_picks
    FROM game_data.participant_stats AS ps
    INNER JOIN rare_champion_teampositions AS rct
        ON
            ps.championid = rct.championid
            AND ps.teamposition = rct.teamposition
    GROUP BY
        ps.puuid,
        ps.championid,
        ps.teamposition
    HAVING player_champion_teamposition_picks < 30
)

SELECT
    ps.matchid,
    1 AS has_rare_role
FROM game_data.participant_stats AS ps
INNER JOIN player_rare_champion_teamposition_pick_counts AS prctpc
    ON
        ps.puuid = prctpc.puuid
        AND ps.championid = prctpc.championid
        AND ps.teamposition = prctpc.teamposition
GROUP BY ps.matchid;

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
    (ps.kills + ps.assists) * 5 < ps.deaths AS player_low_kda,
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
        AND (
            ps.totalminionskilled + ps.neutralminionskilled
        ) * 60.0 / ps.timeplayed < 4.5
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
    max(
        p.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy
    ) AS team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    max(
        p.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy
    ) AS team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy,
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
