-- Populate persistent game_data_filtered.* tables by filtering each source
-- game_data.* table against game_data_filtered.valid_game_ids.
-- valid_game_ids is small, so ClickHouse can keep the membership set resident
-- while streaming each source table.
-- Derived analytical tables (participant_item_value_totals, the per-minute
-- tl_* aggregates) are NOT populated here -- see
-- database/clickhouse/schema/analytics_builds/ for those builds.
-- Run after 5001_valid_game_ids_build.sql.
--
-- Runtime note: the clickhouse container is currently capped at 10 GiB and
-- 2 CPUs. We pause background merges for the duration of the build, use
-- `WHERE matchid IN (...)` with `max_threads = 2` to use both CPUs, and only
-- drop caches after the largest copy phases so RSS stays bounded without
-- paying the purge cost after every table.

SYSTEM STOP MERGES;
SET max_threads = 2;
SET max_block_size = 8192;
SET max_insert_block_size = 32768;

TRUNCATE TABLE game_data_filtered.metadata;
INSERT INTO game_data_filtered.metadata
SELECT t.*
FROM game_data.metadata AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.info;
INSERT INTO game_data_filtered.info
SELECT t.*
FROM game_data.info AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.bans;
INSERT INTO game_data_filtered.bans
SELECT t.*
FROM game_data.bans AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.feats;
INSERT INTO game_data_filtered.feats
SELECT t.*
FROM game_data.feats AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.objectives;
INSERT INTO game_data_filtered.objectives
SELECT t.*
FROM game_data.objectives AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.participant_stats;
INSERT INTO game_data_filtered.participant_stats
SELECT t.*
FROM game_data.participant_stats AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.participant_challenges;
INSERT INTO game_data_filtered.participant_challenges
SELECT t.*
FROM game_data.participant_challenges AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.participant_perk_values;
INSERT INTO game_data_filtered.participant_perk_values
SELECT t.*
FROM game_data.participant_perk_values AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.participant_perk_ids;
INSERT INTO game_data_filtered.participant_perk_ids
SELECT t.*
FROM game_data.participant_perk_ids AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

-- Largest/widest non-timeline copies are done; trim caches before timeline scans.
SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_participant_stats;
INSERT INTO game_data_filtered.tl_participant_stats
SELECT t.*
FROM game_data.tl_participant_stats AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

-- The participant stats scan is the heaviest single copy in the rebuild.
SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_ward_placed;
INSERT INTO game_data_filtered.tl_ward_placed
SELECT t.*
FROM game_data.tl_ward_placed AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_ward_kill;
INSERT INTO game_data_filtered.tl_ward_kill
SELECT t.*
FROM game_data.tl_ward_kill AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_item_purchased;
INSERT INTO game_data_filtered.tl_item_purchased
SELECT t.*
FROM game_data.tl_item_purchased AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_item_sold;
INSERT INTO game_data_filtered.tl_item_sold
SELECT t.*
FROM game_data.tl_item_sold AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_item_destroyed;
INSERT INTO game_data_filtered.tl_item_destroyed
SELECT t.*
FROM game_data.tl_item_destroyed AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_item_undo;
INSERT INTO game_data_filtered.tl_item_undo
SELECT t.*
FROM game_data.tl_item_undo AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_level_up;
INSERT INTO game_data_filtered.tl_level_up
SELECT t.*
FROM game_data.tl_level_up AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_skill_level_up;
INSERT INTO game_data_filtered.tl_skill_level_up
SELECT t.*
FROM game_data.tl_skill_level_up AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

-- The ward/item/level tables are the next biggest sustained scan group.
SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_pause_end;
INSERT INTO game_data_filtered.tl_pause_end
SELECT t.*
FROM game_data.tl_pause_end AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_game_end;
INSERT INTO game_data_filtered.tl_game_end
SELECT t.*
FROM game_data.tl_game_end AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_objective_bounty_prestart;
INSERT INTO game_data_filtered.tl_objective_bounty_prestart
SELECT t.*
FROM game_data.tl_objective_bounty_prestart AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_objective_bounty_finish;
INSERT INTO game_data_filtered.tl_objective_bounty_finish
SELECT t.*
FROM game_data.tl_objective_bounty_finish AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_feat_update;
INSERT INTO game_data_filtered.tl_feat_update
SELECT t.*
FROM game_data.tl_feat_update AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_champion_transform;
INSERT INTO game_data_filtered.tl_champion_transform
SELECT t.*
FROM game_data.tl_champion_transform AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_building_kill;
INSERT INTO game_data_filtered.tl_building_kill
SELECT t.*
FROM game_data.tl_building_kill AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_champion_kill;
INSERT INTO game_data_filtered.tl_champion_kill
SELECT t.*
FROM game_data.tl_champion_kill AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_champion_special_kill;
INSERT INTO game_data_filtered.tl_champion_special_kill
SELECT t.*
FROM game_data.tl_champion_special_kill AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_dragon_soul_given;
INSERT INTO game_data_filtered.tl_dragon_soul_given
SELECT t.*
FROM game_data.tl_dragon_soul_given AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_elite_monster_kill;
INSERT INTO game_data_filtered.tl_elite_monster_kill
SELECT t.*
FROM game_data.tl_elite_monster_kill AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_turret_plate_destroyed;
INSERT INTO game_data_filtered.tl_turret_plate_destroyed
SELECT t.*
FROM game_data.tl_turret_plate_destroyed AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

-- The combat/objective event group is complete; reset caches before damage rows.
SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.tl_ck_victim_damage_dealt;
INSERT INTO game_data_filtered.tl_ck_victim_damage_dealt
SELECT t.*
FROM game_data.tl_ck_victim_damage_dealt AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

TRUNCATE TABLE game_data_filtered.tl_ck_victim_damage_received;
INSERT INTO game_data_filtered.tl_ck_victim_damage_received
SELECT t.*
FROM game_data.tl_ck_victim_damage_received AS t
WHERE t.matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids);

-- Final cleanup before background merges resume.
SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

SYSTEM START MERGES;
