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
        sumIf(totaldamagedealttochampions, teamposition != 'UTILITY')
            AS team_non_utility_damage_to_champions
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
