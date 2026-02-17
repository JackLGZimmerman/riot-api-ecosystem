CREATE VIEW IF NOT EXISTS game_data_filtered.participant_perk_ids AS
SELECT t.*
FROM game_data.participant_perk_ids AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.gameid = v.gameid;
