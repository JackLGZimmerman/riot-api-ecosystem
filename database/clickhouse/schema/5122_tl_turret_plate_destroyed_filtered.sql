CREATE VIEW IF NOT EXISTS game_data_filtered.tl_turret_plate_destroyed AS
SELECT t.*
FROM game_data.tl_turret_plate_destroyed AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.gameid = v.gameid;
