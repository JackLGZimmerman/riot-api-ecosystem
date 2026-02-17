CREATE VIEW IF NOT EXISTS game_data_filtered.tl_dragon_soul_given AS
SELECT t.*
FROM game_data.tl_dragon_soul_given AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.gameid = v.gameid;
