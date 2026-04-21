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
-- they don't compete for RSS with the active INSERT, (2) drop mark/
-- uncompressed/compiled caches and run SYSTEM JEMALLOC PURGE between
-- statements so freed memory is returned to the OS, and (3) chunk the
-- participant_challenges INSERT by cityHash64(matchid) because its dynamic
-- JSON payload column (~8.9 GiB uncompressed) cannot be streamed in one shot.

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

-- participant_challenges carries a dynamic JSON payload column whose
-- per-block working set blows the 5 GiB cap on a single streaming read.
-- Split the insert into 4 cityHash64(matchid) buckets so each chunk only
-- touches ~1/4 of the source, and purge caches between chunks.
TRUNCATE TABLE game_data_filtered.participant_challenges;

INSERT INTO game_data_filtered.participant_challenges
SELECT t.*
FROM game_data.participant_challenges AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
WHERE cityHash64(t.matchid) % 4 = 0
SETTINGS max_threads = 1, max_block_size = 2048, max_insert_block_size = 8192;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

INSERT INTO game_data_filtered.participant_challenges
SELECT t.*
FROM game_data.participant_challenges AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
WHERE cityHash64(t.matchid) % 4 = 1
SETTINGS max_threads = 1, max_block_size = 2048, max_insert_block_size = 8192;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

INSERT INTO game_data_filtered.participant_challenges
SELECT t.*
FROM game_data.participant_challenges AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
WHERE cityHash64(t.matchid) % 4 = 2
SETTINGS max_threads = 1, max_block_size = 2048, max_insert_block_size = 8192;

SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

INSERT INTO game_data_filtered.participant_challenges
SELECT t.*
FROM game_data.participant_challenges AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid -- noqa: PRS
WHERE cityHash64(t.matchid) % 4 = 3
SETTINGS max_threads = 1, max_block_size = 2048, max_insert_block_size = 8192;

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

-- tl_payload_event skipped: ClickHouse LOGICAL_ERROR (Code 49) on Object-type
-- payload column serialization. Table is excluded from filtered dataset.

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
