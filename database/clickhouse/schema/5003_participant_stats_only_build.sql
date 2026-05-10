-- Fast filtered-table refresh for filter iteration.
--
-- Rebuilds ONLY game_data_filtered.participant_stats from the current
-- game_data_filtered.valid_game_ids. Use after 4000/4001/5002 when validating
-- filter changes that remove games from the pool and you do not need tl_* or
-- derived 52XX tables refreshed.
TRUNCATE TABLE game_data_filtered.participant_stats;

INSERT INTO game_data_filtered.participant_stats
SELECT t.*
FROM game_data.participant_stats AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);
