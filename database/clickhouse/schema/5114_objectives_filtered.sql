CREATE TABLE IF NOT EXISTS game_data_filtered.objectives
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.objectives AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;
