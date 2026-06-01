-- noqa: disable=PRS
-- Fast filtered-table refresh for filter iteration.
--
-- Rebuilds game_data_filtered.participant_stats. Timeline profiling tables are
-- no longer mirrored into game_data_filtered.

SYSTEM STOP MERGES;
SET max_threads = 2;
SET max_block_size = 8192;
SET max_insert_block_size = 32768;

TRUNCATE TABLE game_data_filtered.participant_stats;

INSERT INTO game_data_filtered.participant_stats
SELECT t.*
FROM game_data.participant_stats AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

SYSTEM START MERGES;
