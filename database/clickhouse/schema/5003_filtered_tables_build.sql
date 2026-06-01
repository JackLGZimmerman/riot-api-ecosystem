-- noqa: disable=PRS
-- Populate the persistent game_data_filtered tables used by the production ML
-- feature path. The filtered database no longer mirrors every raw game_data.*
-- table; raw snapshots and timeline tables are intentionally left in game_data.
--
-- Run after 5001_valid_game_ids_build.sql.

SYSTEM STOP MERGES;
SET max_threads = 2;
SET max_block_size = 8192;
SET max_insert_block_size = 32768;

TRUNCATE TABLE game_data_filtered.participant_stats;

INSERT INTO game_data_filtered.participant_stats
SELECT t.*
FROM game_data.participant_stats AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

SYSTEM START MERGES;
