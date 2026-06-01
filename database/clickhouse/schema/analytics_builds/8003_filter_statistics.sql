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
                tuple(1, '01-kda-lt-0.20', countIf(player_low_kda)),
                tuple(2, '02-spent-lt-50%-earned-on-loss', countIf(player_gold_spent)),
                tuple(
                    3,
                    '03-suspect-player-suffix-wr-gte-85%',
                    countIf(player_high_winrate)
                ),
                tuple(
                    4,
                    '04-team-kd-ratio-lt-0.40-vs-enemy',
                    countIf(team_kills_to_deaths)
                ),
                tuple(
                    5,
                    '05-winning-player-kills-gt-75%-team-kills',
                    countIf(solo_carried)
                ),
                tuple(6, '06-non-utility-dmg-share-lt-2%', countIf(too_little_damage)),
                tuple(
                    7,
                    '07-non-utility-cs-per-min-lt-3.0',
                    countIf(low_minions_killed)
                ),
                tuple(
                    8,
                    '08-team-non-utility-avg-cs-per-min-gt-2.0-below-enemy',
                    countIf(team_non_utility_avg_cs_per_min_gt_1_0_below_enemy)
                ),
                tuple(
                    9,
                    '09-team-non-utility-dmg-to-champs-ratio-lt-1/2-vs-enemy',
                    countIf(team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy)
                ),
                tuple(10, '10-low-build-value-lt-0.5', countIf(low_build_value)),
                tuple(
                    11,
                    '11-unknown-teamposition',
                    countIf(unknown_teamposition)
                ),
                tuple(
                    12,
                    '12-game-ruining-behavior',
                    countIf(game_ruining_behavior)
                ),
                tuple(
                    13,
                    '13-was-severe-transgressor',
                    countIf(was_severe_transgressor)
                ),
                tuple(
                    14,
                    '14-caused-game-end-from-ignb-surrender',
                    countIf(caused_game_end_from_ignb_surrender)
                ),
                tuple(
                    15,
                    '15-team-ignb-surrendered',
                    countIf(team_ignb_surrendered)
                ),
                tuple(
                    16,
                    '16-was-premade-with-ignb-game-end-causer',
                    countIf(was_premade_with_ignb_game_end_causer)
                ),
                tuple(
                    17,
                    '17-was-premade-with-severe-transgressor',
                    countIf(was_premade_with_severe_transgressor)
                ),
                tuple(
                    18,
                    '18-zero-spell-casts-loss',
                    countIf(zero_spell_casts_loss)
                ),
                tuple(
                    20,
                    '20-zero-item-purchases-loss',
                    countIf(zero_item_purchases_loss)
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
