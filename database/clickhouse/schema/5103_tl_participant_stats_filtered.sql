CREATE VIEW IF NOT EXISTS game_data_filtered.tl_participant_stats AS
-- Source table has no gameid column, so this view is currently passthrough.
SELECT t.*
FROM game_data.tl_participant_stats AS t;
