-- Report per-rule match counts plus a survivor tally for each filter stage.
-- Run against the final filter_stg_game_flags / stage-valid helper tables
-- produced by database/clickhouse/schema/4001_filter_build.sql.

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
                tuple(1, '01-kda-lt-0.2', countIf(player_low_kda)),
                tuple(2, '02-spent-lt-50%-earned', countIf(player_gold_spent)),
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
                tuple(7, '07-player-kills-gt-75%-team-kills', countIf(solo_carried)),
                tuple(
                    8, '08-non-utility-dmg-share-lt-5%', countIf(too_little_damage)
                ),
                tuple(
                    9, '09-non-utility-cs-per-min-lt-4', countIf(low_minions_killed)
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
                    '15-rare-role-champion-position-lt-0.4-pct-lt-30-games',
                    countIf(has_rare_role)
                ),
                tuple(
                    16,
                    '16-rare-build-label-lt-8-games',
                    countIf(rare_build_label)
                )
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
    '17-any-filter-triggered' AS filter,
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
    'stage2-survivors' AS filter,
    count() AS number_of_games,
    (SELECT tot.total_games FROM tot) AS total_games,
    round(
        100.0 * count() / nullIf((SELECT tot.total_games FROM tot), 0), 2
    ) AS pct_of_total_games
FROM game_data.filter_stg_stage2_valid_matchids

UNION ALL

SELECT
    'stage3-survivors' AS filter,
    countIf(any_filter_triggered = 0) AS number_of_games,
    count() AS total_games,
    round(
        100.0 * countIf(any_filter_triggered = 0) / nullIf(count(), 0), 2
    ) AS pct_of_total_games
FROM game_data.filter_stg_game_flags

ORDER BY filter;
