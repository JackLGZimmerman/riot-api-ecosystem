WITH
player_filters AS (
    SELECT
        matchid,
        (kills + assists) * 5 < coalesce(deaths, 1) as player_low_kda,
        kills + assists = 0 AND deaths > 4 as no_contribution_kda,
        summoner1casts = 0 or summoner2casts = 0 as bad_summoner_usage,
        kills / sum(kills) OVER (PARTITION BY matchid, teamid) > 0.65 as solo_carried,
        teamposition != 'UTILITY' AND totaldamagedealttochampions / sum(totaldamagedealttochampions) OVER (PARTITION BY matchid, teamid) < 0.075 as too_little_damage,
        teamposition != 'UTILITY' and timeplayed > 0 and (totalminionskilled + neutralminionskilled) * 60.0 / timeplayed < 4.5 as low_minions_killed,
        (item0 = item1 AND item1 = item2 AND item2 = item3 AND item3 = item4 AND item4 = item5) as grief_build,
        item0 = 0 and item1 = 0 and item2=0 and item3 =0 and item4=0 and item5=0 as sold_all_items,
        goldspent / goldearned < 0.6 as player_gold_spent
    FROM game_data.participant_stats
),
team_filters AS (
    SELECT
        matchid,
        teamid,
        (sum(kills)) * 3 < sum(deaths) as team_kills_to_deaths
    FROM game_data.participant_stats
    GROUP BY matchid, teamid
),
game_filters AS (
    SELECT
        matchid,
        max(timeplayed) <= 15 * 60 AS game_time_lte_15
    FROM game_data.participant_stats
    GROUP BY matchid
),
tot AS (
    SELECT countDistinct(matchid) AS total_games
    FROM game_data.participant_stats
),
filters AS (
    SELECT
        row_number() OVER (ORDER BY tuple()) AS rn,
        rule_value
    FROM
    (
        SELECT countDistinctIf(matchid, player_low_kda) AS rule_value FROM player_filters
        UNION ALL SELECT countDistinctIf(matchid, player_gold_spent) FROM player_filters
        UNION ALL SELECT countDistinctIf(matchid, no_contribution_kda) FROM player_filters
        UNION ALL SELECT countDistinctIf(matchid, bad_summoner_usage) FROM player_filters
        UNION ALL SELECT countDistinctIf(matchid, team_kills_to_deaths) FROM team_filters
        UNION ALL SELECT countDistinctIf(matchid, solo_carried) FROM player_filters
        UNION ALL SELECT countDistinctIf(matchid, too_little_damage) FROM player_filters
        UNION ALL SELECT countDistinctIf(matchid, low_minions_killed) FROM player_filters
        UNION ALL SELECT countDistinctIf(matchid, sold_all_items) FROM player_filters
        UNION ALL SELECT countDistinctIf(matchid, grief_build) FROM player_filters
        UNION ALL SELECT countDistinctIf(matchid, game_time_lte_15) FROM game_filters
        UNION ALL
        SELECT countDistinct(matchid)
        FROM
        (
            SELECT matchid FROM player_filters WHERE player_low_kda
            UNION DISTINCT
            SELECT matchid FROM player_filters WHERE player_gold_spent
            UNION DISTINCT
            SELECT matchid FROM player_filters WHERE no_contribution_kda
            UNION DISTINCT
            SELECT matchid FROM player_filters WHERE bad_summoner_usage
            UNION DISTINCT
            SELECT matchid FROM team_filters WHERE team_kills_to_deaths
            UNION DISTINCT
            SELECT matchid FROM player_filters WHERE solo_carried
            UNION DISTINCT
            SELECT matchid FROM player_filters WHERE too_little_damage
            UNION DISTINCT
            SELECT matchid FROM player_filters WHERE low_minions_killed
            UNION DISTINCT
            SELECT matchid FROM player_filters WHERE sold_all_items
            UNION DISTINCT
            SELECT matchid FROM player_filters WHERE grief_build
            UNION DISTINCT
            SELECT matchid FROM game_filters WHERE game_time_lte_15
        )
    )
),
rule_base AS (
    SELECT
        row_number() OVER (ORDER BY tuple()) AS rn,
        rule_name
    FROM
    (
        SELECT '01-kda-lt-0.2' AS rule_name
        UNION ALL SELECT '02-spent-lt-60%-earned'
        UNION ALL SELECT '03-kills+assists-is-0-and-deaths-gt-4'
        UNION ALL SELECT '04-either-summoner-not-cast'
        UNION ALL SELECT '05-team-kd-lt-0.33'
        UNION ALL SELECT '06-player-kills-gt-65%-team-kills'
        UNION ALL SELECT '07-non-utility-dmg-share-lt-7.5%'
        UNION ALL SELECT '08-non-utility-cs-per-min-lt-4.5'
        UNION ALL SELECT '09-all-items-0'
        UNION ALL SELECT '10-at-least-5-same-items'
        UNION ALL SELECT '11-game-time-lte-15'
        UNION ALL SELECT '12-any-filter-triggered'
    )
)
SELECT
    rb.rule_name AS filter,
    f.rule_value AS number_of_games,
    round(100.0 * f.rule_value / nullIf(t.total_games, 0), 2) AS pct_of_total_games,
    t.total_games
FROM rule_base rb
INNER JOIN filters f USING (rn)
CROSS JOIN tot t
ORDER BY rn;
