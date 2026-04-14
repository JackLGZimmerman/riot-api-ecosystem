TRUNCATE TABLE game_data_filtered.valid_game_ids;

INSERT INTO game_data_filtered.valid_game_ids
(
    matchid
)
SELECT matchid
FROM game_data.filter_game_validity
GROUP BY matchid
HAVING max(rule_mask) = 0;
