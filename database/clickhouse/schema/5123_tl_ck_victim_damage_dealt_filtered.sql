CREATE TABLE IF NOT EXISTS game_data_filtered.tl_ck_victim_damage_dealt
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_ck_victim_damage_dealt AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;
