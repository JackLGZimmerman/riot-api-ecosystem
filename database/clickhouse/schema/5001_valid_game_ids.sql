CREATE VIEW IF NOT EXISTS game_data_filtered.valid_game_ids AS
SELECT DISTINCT gameid
FROM game_data.filter_game_validity
WHERE is_valid = 1;
