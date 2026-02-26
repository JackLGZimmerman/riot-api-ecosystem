CREATE TABLE IF NOT EXISTS game_data_filtered.tl_payload_event
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_payload_event AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;
