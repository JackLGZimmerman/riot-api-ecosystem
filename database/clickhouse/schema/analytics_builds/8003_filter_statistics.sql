-- Report per-rule match counts plus a survivor tally for each filter stage.
-- Run against the final filter_stg_game_flags / stage-valid helper tables
-- produced by database/clickhouse/schema/4000_filter_build.sql.

WITH
tot AS (
    SELECT count() AS total_games
    FROM game_data.filter_stg_game_flags
),

rules AS (
    SELECT
        tupleElement(rule, 1) AS rn,
        tupleElement(rule, 2) AS filter,
        tupleElement(rule, 3) AS number_of_games
    FROM (
        SELECT
            arrayJoin([
                tuple(1, '01-kda-lt-0.30', countIf(player_low_kda)),
                tuple(2, '02-spent-lt-50%-earned', countIf(player_gold_spent)),
                tuple(3, '03-kill-participation-low', countIf(kill_participation_low)),
                tuple(
                    4,
                    '04-player-games-gt-40-winrate-gt-70%',
                    countIf(player_high_winrate)
                ),
                tuple(
                    6,
                    '06-team-kd-ratio-lt-0.50-vs-enemy',
                    countIf(team_kills_to_deaths)
                ),
                tuple(
                    7,
                    '07-winning-player-kills-gt-75%-team-kills',
                    countIf(solo_carried)
                ),
                tuple(8, '08-non-utility-dmg-share-lt-2%', countIf(too_little_damage)),
                tuple(9, '09-non-utility-cs-per-min-lt-4.0', countIf(low_minions_killed)),
                tuple(
                    10,
                    '10-team-non-utility-avg-cs-per-min-gt-1.0-below-enemy',
                    countIf(team_non_utility_avg_cs_per_min_gt_1_0_below_enemy)
                ),
                tuple(
                    11,
                    '11-team-non-utility-dmg-to-champs-ratio-lt-1/2-vs-enemy',
                    countIf(team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy)
                ),
                tuple(12, '12-low-build-value-lt-1.0', countIf(low_build_value))
            ]) AS rule
        FROM game_data.filter_stg_game_flags
    )
)

SELECT
    r.filter,
    r.number_of_games,
    t.total_games,
    round(100.0 * r.number_of_games / nullIf(t.total_games, 0), 2) AS pct_of_total_games
FROM rules AS r
CROSS JOIN tot AS t

UNION ALL

SELECT
    'any-filter-triggered' AS filter,
    countIf(any_filter_triggered) AS number_of_games,
    count() AS total_games,
    round(
        100.0 * countIf(any_filter_triggered) / nullIf(count(), 0), 2
    ) AS pct_of_total_games
FROM game_data.filter_stg_game_flags

UNION ALL

-- Stage-level survivor counts (pool sizes passed to the next stage).
SELECT
    'stage1-survivors' AS filter,
    count() AS number_of_games,
    (SELECT tot.total_games FROM tot) AS total_games,
    round(
        100.0 * count() / nullIf((SELECT tot.total_games FROM tot), 0), 2
    ) AS pct_of_total_games
FROM game_data.filter_stg_stage1_valid_matchids

UNION ALL

SELECT
    'final-survivors' AS filter,
    countIf(any_filter_triggered = 0) AS number_of_games,
    count() AS total_games,
    round(
        100.0 * countIf(any_filter_triggered = 0) / nullIf(count(), 0), 2
    ) AS pct_of_total_games
FROM game_data.filter_stg_game_flags

ORDER BY filter;
