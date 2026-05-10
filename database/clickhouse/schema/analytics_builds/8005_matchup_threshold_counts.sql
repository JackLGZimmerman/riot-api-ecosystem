-- noqa: disable=AL03,LT14
-- Row counts per 6xxx aggregate at successive `matchups` thresholds, train
-- split only. Justifies the `matchups >= 5` cutoff applied in the
-- 6901_ml_interaction_counts build: low-support rows dominate the large
-- matchup/synergy right-hand sides and inflate the join hash table without
-- contributing usable signal. Re-run after rebuilding 6xxx to refresh.

SELECT
    table_name,
    rows_total,
    rows_ge2,
    rows_ge3,
    rows_ge4,
    rows_ge5,
    round(100.0 * rows_ge2 / nullIf(rows_total, 0), 2) AS pct_ge2,
    round(100.0 * rows_ge3 / nullIf(rows_total, 0), 2) AS pct_ge3,
    round(100.0 * rows_ge4 / nullIf(rows_total, 0), 2) AS pct_ge4,
    round(100.0 * rows_ge5 / nullIf(rows_total, 0), 2) AS pct_ge5
FROM (
    SELECT
        'matchup_1v1' AS table_name,
        count() AS rows_total,
        countIf(matchups >= 2) AS rows_ge2,
        countIf(matchups >= 3) AS rows_ge3,
        countIf(matchups >= 4) AS rows_ge4,
        countIf(matchups >= 5) AS rows_ge5
    FROM game_data_filtered.matchup_1v1 WHERE split = 'train'
    UNION ALL
    SELECT
        'matchup_2v2',
        count(),
        countIf(matchups >= 2),
        countIf(matchups >= 3),
        countIf(matchups >= 4),
        countIf(matchups >= 5)
    FROM game_data_filtered.matchup_2v2 WHERE split = 'train'
    UNION ALL
    SELECT
        'matchup_2v1',
        count(),
        countIf(matchups >= 2),
        countIf(matchups >= 3),
        countIf(matchups >= 4),
        countIf(matchups >= 5)
    FROM game_data_filtered.matchup_2v1 WHERE split = 'train'
    UNION ALL
    SELECT
        'matchup_3v1',
        count(),
        countIf(matchups >= 2),
        countIf(matchups >= 3),
        countIf(matchups >= 4),
        countIf(matchups >= 5)
    FROM game_data_filtered.matchup_3v1 WHERE split = 'train'
    UNION ALL
    SELECT
        'matchup_3v2',
        count(),
        countIf(matchups >= 2),
        countIf(matchups >= 3),
        countIf(matchups >= 4),
        countIf(matchups >= 5)
    FROM game_data_filtered.matchup_3v2 WHERE split = 'train'
    UNION ALL
    SELECT
        'matchup_3v3',
        count(),
        countIf(matchups >= 2),
        countIf(matchups >= 3),
        countIf(matchups >= 4),
        countIf(matchups >= 5)
    FROM game_data_filtered.matchup_3v3 WHERE split = 'train'
    UNION ALL
    SELECT
        'synergy_1vx',
        count(),
        countIf(matchups >= 2),
        countIf(matchups >= 3),
        countIf(matchups >= 4),
        countIf(matchups >= 5)
    FROM game_data_filtered.synergy_1vx WHERE split = 'train'
    UNION ALL
    SELECT
        'synergy_2vx',
        count(),
        countIf(matchups >= 2),
        countIf(matchups >= 3),
        countIf(matchups >= 4),
        countIf(matchups >= 5)
    FROM game_data_filtered.synergy_2vx WHERE split = 'train'
    UNION ALL
    SELECT
        'synergy_3vx',
        count(),
        countIf(matchups >= 2),
        countIf(matchups >= 3),
        countIf(matchups >= 4),
        countIf(matchups >= 5)
    FROM game_data_filtered.synergy_3vx WHERE split = 'train'
    UNION ALL
    SELECT
        'synergy_4vx',
        count(),
        countIf(matchups >= 2),
        countIf(matchups >= 3),
        countIf(matchups >= 4),
        countIf(matchups >= 5)
    FROM game_data_filtered.synergy_4vx WHERE split = 'train'
)
ORDER BY table_name;
