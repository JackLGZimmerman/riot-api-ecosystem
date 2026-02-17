CREATE VIEW IF NOT EXISTS game_data_filtered.info AS
SELECT t.*
FROM game_data.info AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.gameid = v.gameid;
