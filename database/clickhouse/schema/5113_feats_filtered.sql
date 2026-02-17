CREATE VIEW IF NOT EXISTS game_data_filtered.feats AS
SELECT t.*
FROM game_data.feats AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.gameId = v.gameid;
