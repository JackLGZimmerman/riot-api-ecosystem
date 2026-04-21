TRUNCATE TABLE game_data_filtered.valid_game_ids;

INSERT INTO game_data_filtered.valid_game_ids (matchid)
SELECT matchid
FROM game_data.filter_stg_game_flags
WHERE any_filter_triggered = 0;
