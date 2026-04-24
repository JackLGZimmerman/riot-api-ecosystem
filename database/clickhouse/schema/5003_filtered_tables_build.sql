-- Populate persistent game_data_filtered.* tables by SEMI-joining each source
-- game_data.* table against game_data_filtered.valid_game_ids.  valid_game_ids
-- is the small build-side hash table, so no full matchid set is materialised.
-- Derived analytical tables (participant_item_value_totals, the per-minute
-- tl_* aggregates) are NOT populated here — see
-- database/clickhouse/schema/analytics_builds/ for those builds.
-- Run after 5002_valid_game_ids_build.sql.
--
-- Memory note: the clickhouse container is capped at 5 GiB.  To stay under
-- that cap we (1) pause background merges for the duration of the build so
-- they don't compete for RSS with the active INSERT, and (2) drop mark/
-- uncompressed/compiled caches and run SYSTEM JEMALLOC PURGE between
-- statements so freed memory is returned to the OS.

SYSTEM STOP MERGES;

TRUNCATE TABLE game_data_filtered.metadata;
INSERT INTO game_data_filtered.metadata
SELECT t.*
FROM game_data.metadata AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.info;
INSERT INTO game_data_filtered.info
SELECT t.*
FROM game_data.info AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.bans;
INSERT INTO game_data_filtered.bans
SELECT t.*
FROM game_data.bans AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.feats;
INSERT INTO game_data_filtered.feats
SELECT t.*
FROM game_data.feats AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.objectives;
INSERT INTO game_data_filtered.objectives
SELECT t.*
FROM game_data.objectives AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.participant_stats;
INSERT INTO game_data_filtered.participant_stats
SELECT t.*
FROM game_data.participant_stats AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.participant_challenges;
INSERT INTO game_data_filtered.participant_challenges
SELECT t.*
FROM game_data.participant_challenges AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.participant_perk_values;
INSERT INTO game_data_filtered.participant_perk_values
SELECT t.*
FROM game_data.participant_perk_values AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.participant_perk_ids;
INSERT INTO game_data_filtered.participant_perk_ids
SELECT t.*
FROM game_data.participant_perk_ids AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_participant_stats;
INSERT INTO game_data_filtered.tl_participant_stats
SELECT t.*
FROM game_data.tl_participant_stats AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_ward_placed;
INSERT INTO game_data_filtered.tl_ward_placed
SELECT t.*
FROM game_data.tl_ward_placed AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_ward_kill;
INSERT INTO game_data_filtered.tl_ward_kill
SELECT t.*
FROM game_data.tl_ward_kill AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_item_purchased;
INSERT INTO game_data_filtered.tl_item_purchased
SELECT t.*
FROM game_data.tl_item_purchased AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_item_sold;
INSERT INTO game_data_filtered.tl_item_sold
SELECT t.*
FROM game_data.tl_item_sold AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_item_destroyed;
INSERT INTO game_data_filtered.tl_item_destroyed
SELECT t.*
FROM game_data.tl_item_destroyed AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_item_undo;
INSERT INTO game_data_filtered.tl_item_undo
SELECT t.*
FROM game_data.tl_item_undo AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_level_up;
INSERT INTO game_data_filtered.tl_level_up
SELECT t.*
FROM game_data.tl_level_up AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_skill_level_up;
INSERT INTO game_data_filtered.tl_skill_level_up
SELECT t.*
FROM game_data.tl_skill_level_up AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_pause_end;
INSERT INTO game_data_filtered.tl_pause_end
SELECT t.*
FROM game_data.tl_pause_end AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_game_end;
INSERT INTO game_data_filtered.tl_game_end
SELECT t.*
FROM game_data.tl_game_end AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_objective_bounty_prestart;
INSERT INTO game_data_filtered.tl_objective_bounty_prestart
SELECT t.*
FROM game_data.tl_objective_bounty_prestart AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_objective_bounty_finish;
INSERT INTO game_data_filtered.tl_objective_bounty_finish
SELECT t.*
FROM game_data.tl_objective_bounty_finish AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_feat_update;
INSERT INTO game_data_filtered.tl_feat_update
SELECT t.*
FROM game_data.tl_feat_update AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_champion_transform;
INSERT INTO game_data_filtered.tl_champion_transform
SELECT t.*
FROM game_data.tl_champion_transform AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_building_kill;
INSERT INTO game_data_filtered.tl_building_kill
SELECT t.*
FROM game_data.tl_building_kill AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_champion_kill;
INSERT INTO game_data_filtered.tl_champion_kill
SELECT t.*
FROM game_data.tl_champion_kill AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_champion_special_kill;
INSERT INTO game_data_filtered.tl_champion_special_kill
SELECT t.*
FROM game_data.tl_champion_special_kill AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_dragon_soul_given;
INSERT INTO game_data_filtered.tl_dragon_soul_given
SELECT t.*
FROM game_data.tl_dragon_soul_given AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_elite_monster_kill;
INSERT INTO game_data_filtered.tl_elite_monster_kill
SELECT t.*
FROM game_data.tl_elite_monster_kill AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_turret_plate_destroyed;
INSERT INTO game_data_filtered.tl_turret_plate_destroyed
SELECT t.*
FROM game_data.tl_turret_plate_destroyed AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_ck_victim_damage_dealt;
INSERT INTO game_data_filtered.tl_ck_victim_damage_dealt
SELECT t.*
FROM game_data.tl_ck_victim_damage_dealt AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_ck_victim_damage_received;
INSERT INTO game_data_filtered.tl_ck_victim_damage_received
SELECT t.*
FROM game_data.tl_ck_victim_damage_received AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
SETTINGS max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

SYSTEM START MERGES;
