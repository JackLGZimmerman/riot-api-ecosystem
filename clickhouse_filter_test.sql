-- Active reporting query.
-- Build the utility stages first, then run this query against the precomputed
-- helper table:
--   database/clickhouse/schema/4022_filter_utility_all_schema.sql
--   database/clickhouse/schema/4023_filter_utility_all_build.sql
-- The individual stage files remain available when you want to rerun one layer only.

WITH
tot AS (
    SELECT count() AS total_games
    FROM game_data.filter_utility_game_flags
),

rules AS (
    SELECT
        tupleElement(rule, 1) AS rn,
        tupleElement(rule, 2) AS filter,
        tupleElement(rule, 3) AS number_of_games
    FROM (
        SELECT
            arrayJoin([
                tuple(1, '01-kda-lt-0.2', countIf(player_low_kda)),
                tuple(2, '02-spent-lt-60%-earned', countIf(player_gold_spent)),
                tuple(
                    3,
                    '03-kills+assists-is-0-and-deaths-gt-4',
                    countIf(no_contribution_kda)
                ),
                tuple(4, '04-either-summoner-not-cast', countIf(bad_summoner_usage)),
                tuple(
                    5,
                    '05-player-games-gr-40-winrate-gt-70%',
                    countIf(player_high_winrate)
                ),
                tuple(
                    6,
                    '06-team-kd-ratio-lt-0.33-vs-enemy',
                    countIf(team_kills_to_deaths)
                ),
                tuple(7, '07-player-kills-gt-65%-team-kills', countIf(solo_carried)),
                tuple(
                    8, '08-non-utility-dmg-share-lt-7.5%', countIf(too_little_damage)
                ),
                tuple(
                    9, '09-non-utility-cs-per-min-lt-4.5', countIf(low_minions_killed)
                ),
                tuple(
                    10,
                    '10-team-non-utility-avg-cs-per-min-gt-2.5-below-enemy',
                    countIf(team_non_utility_avg_cs_per_min_gt_2_5_below_enemy)
                ),
                tuple(
                    11,
                    '11-team-non-utility-dmg-to-champs-ratio-lt-1/3-vs-enemy',
                    countIf(team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy)
                ),
                tuple(12, '12-all-items-0', countIf(sold_all_items)),
                tuple(13, '13-all-items-same', countIf(grief_build)),
                tuple(14, '14-game-time-lte-18', countIf(game_time_lte_18)),
                tuple(
                    15,
                    concat(
                        '15-player-champion+position-lt-30-picks',
                        '-and-position-lt-0.6%-of-champion-picks'
                    ),
                    countIf(low_champion_teamposition_history)
                ),
                tuple(
                    16,
                    '16-any-filter-triggered',
                    countIf(
                        player_low_kda
                        OR player_gold_spent
                        OR no_contribution_kda
                        OR bad_summoner_usage
                        OR player_high_winrate
                        OR team_kills_to_deaths
                        OR solo_carried
                        OR too_little_damage
                        OR low_minions_killed
                        OR team_non_utility_avg_cs_per_min_gt_2_5_below_enemy
                        OR team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy
                        OR sold_all_items
                        OR grief_build
                        OR game_time_lte_18
                        OR low_champion_teamposition_history
                    )
                )
            ]) AS rule
        FROM game_data.filter_utility_game_flags
    )
)

SELECT
    r.filter,
    r.number_of_games,
    t.total_games,
    round(100.0 * r.number_of_games / nullIf(t.total_games, 0), 2) AS pct_of_total_games
FROM rules AS r
CROSS JOIN tot AS t
ORDER BY r.rn;
