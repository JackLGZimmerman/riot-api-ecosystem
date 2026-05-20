-- noqa: disable=PRS,RF01
-- Fast filtered-table refresh for filter iteration.
--
-- Rebuilds the small ML-ordering table, participant_stats, and the timeline
-- event tables required for profiling (champion kills, item buy/sell/undo,
-- elite-monster kills, building kills, and per-minute participant stats) from
-- the current game_data_filtered.valid_game_ids. Use after 4000/4001/5002
-- when validating filter changes and you do not need the full 5003 rebuild.

SYSTEM STOP MERGES;
SET max_threads = 2;
SET max_block_size = 8192;
SET max_insert_block_size = 32768;

TRUNCATE TABLE game_data_filtered.info;

INSERT INTO game_data_filtered.info
SELECT t.*
FROM game_data.info AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);

TRUNCATE TABLE game_data_filtered.participant_stats;

INSERT INTO game_data_filtered.participant_stats
SELECT t.*
FROM game_data.participant_stats AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_participant_stats;

INSERT INTO game_data_filtered.tl_participant_stats
SELECT t.*
FROM game_data.tl_participant_stats AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_champion_kill;

INSERT INTO game_data_filtered.tl_champion_kill
SELECT t.*
FROM game_data.tl_champion_kill AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);

TRUNCATE TABLE game_data_filtered.tl_item_purchased;

INSERT INTO game_data_filtered.tl_item_purchased
SELECT t.*
FROM game_data.tl_item_purchased AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);

TRUNCATE TABLE game_data_filtered.tl_item_sold;

INSERT INTO game_data_filtered.tl_item_sold
SELECT t.*
FROM game_data.tl_item_sold AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);

TRUNCATE TABLE game_data_filtered.tl_item_undo;

INSERT INTO game_data_filtered.tl_item_undo
SELECT t.*
FROM game_data.tl_item_undo AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);

TRUNCATE TABLE game_data_filtered.tl_elite_monster_kill;

INSERT INTO game_data_filtered.tl_elite_monster_kill
SELECT t.*
FROM game_data.tl_elite_monster_kill AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);

TRUNCATE TABLE game_data_filtered.tl_building_kill;

INSERT INTO game_data_filtered.tl_building_kill
SELECT t.*
FROM game_data.tl_building_kill AS t
WHERE t.matchid IN (SELECT vgi.matchid FROM game_data_filtered.valid_game_ids AS vgi);

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

SYSTEM START MERGES;
