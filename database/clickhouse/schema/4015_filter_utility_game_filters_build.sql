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
