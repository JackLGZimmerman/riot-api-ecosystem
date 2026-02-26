CREATE TABLE IF NOT EXISTS game_data_filtered.participant_stats
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.participant_stats AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;
