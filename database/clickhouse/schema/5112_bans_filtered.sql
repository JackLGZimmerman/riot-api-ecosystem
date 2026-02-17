CREATE VIEW IF NOT EXISTS game_data_filtered.bans AS
SELECT t.*
FROM game_data.bans AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.gameid = v.gameid;
