SELECT *
FROM game_data_filtered.participant_stats
WHERE championid IS NULL OR teamposition IS NULL
