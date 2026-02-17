CREATE VIEW IF NOT EXISTS game_data_filtered.tl_elite_monster_kill AS
SELECT t.*
FROM game_data.tl_elite_monster_kill AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.gameId = v.gameid;
