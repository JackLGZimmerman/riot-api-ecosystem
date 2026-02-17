CREATE VIEW IF NOT EXISTS game_data_filtered.participant_challenges AS
SELECT t.*
FROM game_data.participant_challenges AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.gameid = v.gameid;
