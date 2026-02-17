CREATE VIEW IF NOT EXISTS game_data_filtered.tl_ck_victim_damage_dealt AS
SELECT t.*
FROM game_data.tl_ck_victim_damage_dealt AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.gameid = v.gameid;
