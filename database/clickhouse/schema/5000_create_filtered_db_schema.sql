CREATE DATABASE IF NOT EXISTS game_data_filtered;

-- Persistent MergeTree tables: filtered snapshots of game_data base tables.
-- Populated by the corresponding build script after the filter pipeline runs.
-- ORDER BY keys mirror the source tables.
DROP TABLE IF EXISTS game_data_filtered.metadata;
CREATE TABLE IF NOT EXISTS game_data_filtered.metadata
AS game_data.metadata
ENGINE = MergeTree
ORDER BY (matchid, run_id);
DROP TABLE IF EXISTS game_data_filtered.info;
CREATE TABLE IF NOT EXISTS game_data_filtered.info
AS game_data.info
ENGINE = MergeTree
ORDER BY (matchid, run_id);
DROP TABLE IF EXISTS game_data_filtered.bans;
CREATE TABLE IF NOT EXISTS game_data_filtered.bans
AS game_data.bans
ENGINE = MergeTree
ORDER BY (matchid, teamid, pickturn, run_id);
DROP TABLE IF EXISTS game_data_filtered.feats;
CREATE TABLE IF NOT EXISTS game_data_filtered.feats
AS game_data.feats
ENGINE = MergeTree
ORDER BY (matchid, teamid, feattype, run_id);
DROP TABLE IF EXISTS game_data_filtered.objectives;
CREATE TABLE IF NOT EXISTS game_data_filtered.objectives
AS game_data.objectives
ENGINE = MergeTree
ORDER BY (matchid, teamid, objectivetype, run_id);
DROP TABLE IF EXISTS game_data_filtered.participant_stats;
CREATE TABLE IF NOT EXISTS game_data_filtered.participant_stats
AS game_data.participant_stats
ENGINE = MergeTree
ORDER BY (matchid, participantid, puuid, run_id);
DROP TABLE IF EXISTS game_data_filtered.participant_challenges;
CREATE TABLE IF NOT EXISTS game_data_filtered.participant_challenges
AS game_data.participant_challenges
ENGINE = MergeTree
ORDER BY (matchid, teamid, puuid, run_id);
DROP TABLE IF EXISTS game_data_filtered.participant_perk_values;
CREATE TABLE IF NOT EXISTS game_data_filtered.participant_perk_values
AS game_data.participant_perk_values
ENGINE = MergeTree
ORDER BY (matchid, teamid, puuid, run_id);
DROP TABLE IF EXISTS game_data_filtered.participant_perk_ids;
CREATE TABLE IF NOT EXISTS game_data_filtered.participant_perk_ids
AS game_data.participant_perk_ids
ENGINE = MergeTree
ORDER BY (matchid, teamid, puuid, run_id);
DROP TABLE IF EXISTS game_data_filtered.tl_participant_stats;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_participant_stats
AS game_data.tl_participant_stats
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, participantid, run_id);
DROP TABLE IF EXISTS game_data_filtered.tl_payload_event;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_payload_event
AS game_data.tl_payload_event
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, type, run_id);
DROP TABLE IF EXISTS game_data_filtered.tl_building_kill;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_building_kill
AS game_data.tl_building_kill
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
DROP TABLE IF EXISTS game_data_filtered.tl_champion_kill;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_champion_kill
AS game_data.tl_champion_kill
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, champion_kill_event_id, run_id);
DROP TABLE IF EXISTS game_data_filtered.tl_champion_special_kill;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_champion_special_kill
AS game_data.tl_champion_special_kill
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
DROP TABLE IF EXISTS game_data_filtered.tl_dragon_soul_given;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_dragon_soul_given
AS game_data.tl_dragon_soul_given
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
DROP TABLE IF EXISTS game_data_filtered.tl_elite_monster_kill;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_elite_monster_kill
AS game_data.tl_elite_monster_kill
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
DROP TABLE IF EXISTS game_data_filtered.tl_turret_plate_destroyed;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_turret_plate_destroyed
AS game_data.tl_turret_plate_destroyed
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, run_id);
DROP TABLE IF EXISTS game_data_filtered.tl_ck_victim_damage_dealt;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_ck_victim_damage_dealt
AS game_data.tl_ck_victim_damage_dealt
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, champion_kill_event_id, idx, run_id);
DROP TABLE IF EXISTS game_data_filtered.tl_ck_victim_damage_received;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_ck_victim_damage_received
AS game_data.tl_ck_victim_damage_received
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, champion_kill_event_id, idx, run_id);

DROP TABLE IF EXISTS game_data_filtered.participant_item_value_totals;
CREATE TABLE IF NOT EXISTS game_data_filtered.participant_item_value_totals
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    puuid FixedString (78),
    championid Nullable (Int32),
    teamposition LowCardinality (String),
    attack_damage Float32,
    ability_power Float32,
    lethality Float32,
    on_hit Float32,
    crit Float32,
    utility_enchanter Float32,
    utility_protection Float32,
    ar_tank Float32,
    mr_tank Float32,
    ad_off_tank Float32,
    ap_off_tank Float32,
    highest_value Float32,
    highest_value_label LowCardinality (String)
)
ENGINE = MergeTree
ORDER BY (matchid, participantid, puuid);
