CREATE VIEW IF NOT EXISTS game_data_filtered.tl_payload_event AS
SELECT t.*
FROM game_data.tl_payload_event AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.gameid = v.gameid;
