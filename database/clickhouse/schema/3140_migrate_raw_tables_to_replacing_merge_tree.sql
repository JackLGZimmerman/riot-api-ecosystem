-- One-shot migration for raw matchdata tables from MergeTree/run_id keys to
-- ReplacingMergeTree natural identity keys.
--
-- Run only while the matchdata pipeline is stopped. The live table is swapped
-- atomically with its rebuilt copy, and the previous table is retained as
-- <table>__pre_rmt for validation/rollback. Drop those backups manually after
-- validating counts and downstream builds.
--
-- Duplicate payloads from the pre-RMT era are expected to be identical except
-- for run_id, so OPTIMIZE FINAL may keep any duplicate row version safely.

SET max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768;
SET receive_timeout = 3600, send_timeout = 3600;

DROP TABLE IF EXISTS game_data.matchdata_matchids__rmt_new;
CREATE TABLE game_data.matchdata_matchids__rmt_new AS game_data.matchdata_matchids
ENGINE = ReplacingMergeTree
ORDER BY (matchid);
INSERT INTO game_data.matchdata_matchids__rmt_new SELECT * FROM game_data.matchdata_matchids;
OPTIMIZE TABLE game_data.matchdata_matchids__rmt_new FINAL;
EXCHANGE TABLES game_data.matchdata_matchids AND game_data.matchdata_matchids__rmt_new;
RENAME TABLE game_data.matchdata_matchids__rmt_new TO game_data.matchdata_matchids__pre_rmt;

DROP TABLE IF EXISTS game_data.participant_perk_values__rmt_new;
CREATE TABLE game_data.participant_perk_values__rmt_new AS game_data.participant_perk_values
ENGINE = ReplacingMergeTree
ORDER BY (matchid, teamid, puuid);
INSERT INTO game_data.participant_perk_values__rmt_new SELECT * FROM game_data.participant_perk_values;
OPTIMIZE TABLE game_data.participant_perk_values__rmt_new FINAL;
EXCHANGE TABLES game_data.participant_perk_values AND game_data.participant_perk_values__rmt_new;
RENAME TABLE game_data.participant_perk_values__rmt_new TO game_data.participant_perk_values__pre_rmt;

DROP TABLE IF EXISTS game_data.participant_perk_ids__rmt_new;
CREATE TABLE game_data.participant_perk_ids__rmt_new AS game_data.participant_perk_ids
ENGINE = ReplacingMergeTree
ORDER BY (matchid, teamid, puuid);
INSERT INTO game_data.participant_perk_ids__rmt_new SELECT * FROM game_data.participant_perk_ids;
OPTIMIZE TABLE game_data.participant_perk_ids__rmt_new FINAL;
EXCHANGE TABLES game_data.participant_perk_ids AND game_data.participant_perk_ids__rmt_new;
RENAME TABLE game_data.participant_perk_ids__rmt_new TO game_data.participant_perk_ids__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_participant_stats__rmt_new;
CREATE TABLE game_data.tl_participant_stats__rmt_new AS game_data.tl_participant_stats
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, participantid);
INSERT INTO game_data.tl_participant_stats__rmt_new SELECT * FROM game_data.tl_participant_stats;
OPTIMIZE TABLE game_data.tl_participant_stats__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_participant_stats AND game_data.tl_participant_stats__rmt_new;
RENAME TABLE game_data.tl_participant_stats__rmt_new TO game_data.tl_participant_stats__pre_rmt;

DROP TABLE IF EXISTS game_data.metadata__rmt_new;
CREATE TABLE game_data.metadata__rmt_new AS game_data.metadata
ENGINE = ReplacingMergeTree
ORDER BY (matchid);
INSERT INTO game_data.metadata__rmt_new SELECT * FROM game_data.metadata;
OPTIMIZE TABLE game_data.metadata__rmt_new FINAL;
EXCHANGE TABLES game_data.metadata AND game_data.metadata__rmt_new;
RENAME TABLE game_data.metadata__rmt_new TO game_data.metadata__pre_rmt;

DROP TABLE IF EXISTS game_data.info__rmt_new;
CREATE TABLE game_data.info__rmt_new AS game_data.info
ENGINE = ReplacingMergeTree
ORDER BY (matchid);
INSERT INTO game_data.info__rmt_new SELECT * FROM game_data.info;
OPTIMIZE TABLE game_data.info__rmt_new FINAL;
EXCHANGE TABLES game_data.info AND game_data.info__rmt_new;
RENAME TABLE game_data.info__rmt_new TO game_data.info__pre_rmt;

DROP TABLE IF EXISTS game_data.bans__rmt_new;
CREATE TABLE game_data.bans__rmt_new AS game_data.bans
ENGINE = ReplacingMergeTree
ORDER BY (matchid, teamid, pickturn);
INSERT INTO game_data.bans__rmt_new SELECT * FROM game_data.bans;
OPTIMIZE TABLE game_data.bans__rmt_new FINAL;
EXCHANGE TABLES game_data.bans AND game_data.bans__rmt_new;
RENAME TABLE game_data.bans__rmt_new TO game_data.bans__pre_rmt;

DROP TABLE IF EXISTS game_data.feats__rmt_new;
CREATE TABLE game_data.feats__rmt_new AS game_data.feats
ENGINE = ReplacingMergeTree
ORDER BY (matchid, teamid, feattype);
INSERT INTO game_data.feats__rmt_new SELECT * FROM game_data.feats;
OPTIMIZE TABLE game_data.feats__rmt_new FINAL;
EXCHANGE TABLES game_data.feats AND game_data.feats__rmt_new;
RENAME TABLE game_data.feats__rmt_new TO game_data.feats__pre_rmt;

DROP TABLE IF EXISTS game_data.objectives__rmt_new;
CREATE TABLE game_data.objectives__rmt_new AS game_data.objectives
ENGINE = ReplacingMergeTree
ORDER BY (matchid, teamid, objectivetype);
INSERT INTO game_data.objectives__rmt_new SELECT * FROM game_data.objectives;
OPTIMIZE TABLE game_data.objectives__rmt_new FINAL;
EXCHANGE TABLES game_data.objectives AND game_data.objectives__rmt_new;
RENAME TABLE game_data.objectives__rmt_new TO game_data.objectives__pre_rmt;

DROP TABLE IF EXISTS game_data.participant_stats__rmt_new;
CREATE TABLE game_data.participant_stats__rmt_new AS game_data.participant_stats
ENGINE = ReplacingMergeTree
ORDER BY (matchid, participantid, puuid);
INSERT INTO game_data.participant_stats__rmt_new SELECT * FROM game_data.participant_stats;
OPTIMIZE TABLE game_data.participant_stats__rmt_new FINAL;
EXCHANGE TABLES game_data.participant_stats AND game_data.participant_stats__rmt_new;
RENAME TABLE game_data.participant_stats__rmt_new TO game_data.participant_stats__pre_rmt;

DROP TABLE IF EXISTS game_data.participant_challenges__rmt_new;
CREATE TABLE game_data.participant_challenges__rmt_new AS game_data.participant_challenges
ENGINE = ReplacingMergeTree
ORDER BY (matchid, teamid, puuid);
INSERT INTO game_data.participant_challenges__rmt_new SELECT * FROM game_data.participant_challenges;
OPTIMIZE TABLE game_data.participant_challenges__rmt_new FINAL;
EXCHANGE TABLES game_data.participant_challenges AND game_data.participant_challenges__rmt_new;
RENAME TABLE game_data.participant_challenges__rmt_new TO game_data.participant_challenges__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_building_kill__rmt_new;
CREATE TABLE game_data.tl_building_kill__rmt_new AS game_data.tl_building_kill
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, lanetype, buildingtype, killerid);
INSERT INTO game_data.tl_building_kill__rmt_new SELECT * FROM game_data.tl_building_kill;
OPTIMIZE TABLE game_data.tl_building_kill__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_building_kill AND game_data.tl_building_kill__rmt_new;
RENAME TABLE game_data.tl_building_kill__rmt_new TO game_data.tl_building_kill__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_champion_kill__rmt_new;
CREATE TABLE game_data.tl_champion_kill__rmt_new AS game_data.tl_champion_kill
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, champion_kill_event_id);
INSERT INTO game_data.tl_champion_kill__rmt_new SELECT * FROM game_data.tl_champion_kill;
OPTIMIZE TABLE game_data.tl_champion_kill__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_champion_kill AND game_data.tl_champion_kill__rmt_new;
RENAME TABLE game_data.tl_champion_kill__rmt_new TO game_data.tl_champion_kill__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_champion_special_kill__rmt_new;
CREATE TABLE game_data.tl_champion_special_kill__rmt_new AS game_data.tl_champion_special_kill
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, killtype, killerid);
INSERT INTO game_data.tl_champion_special_kill__rmt_new SELECT * FROM game_data.tl_champion_special_kill;
OPTIMIZE TABLE game_data.tl_champion_special_kill__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_champion_special_kill AND game_data.tl_champion_special_kill__rmt_new;
RENAME TABLE game_data.tl_champion_special_kill__rmt_new TO game_data.tl_champion_special_kill__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_dragon_soul_given__rmt_new;
CREATE TABLE game_data.tl_dragon_soul_given__rmt_new AS game_data.tl_dragon_soul_given
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, teamid, name);
INSERT INTO game_data.tl_dragon_soul_given__rmt_new SELECT * FROM game_data.tl_dragon_soul_given;
OPTIMIZE TABLE game_data.tl_dragon_soul_given__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_dragon_soul_given AND game_data.tl_dragon_soul_given__rmt_new;
RENAME TABLE game_data.tl_dragon_soul_given__rmt_new TO game_data.tl_dragon_soul_given__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_elite_monster_kill__rmt_new;
CREATE TABLE game_data.tl_elite_monster_kill__rmt_new AS game_data.tl_elite_monster_kill
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, monstertype, killerid);
INSERT INTO game_data.tl_elite_monster_kill__rmt_new SELECT * FROM game_data.tl_elite_monster_kill;
OPTIMIZE TABLE game_data.tl_elite_monster_kill__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_elite_monster_kill AND game_data.tl_elite_monster_kill__rmt_new;
RENAME TABLE game_data.tl_elite_monster_kill__rmt_new TO game_data.tl_elite_monster_kill__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_turret_plate_destroyed__rmt_new;
CREATE TABLE game_data.tl_turret_plate_destroyed__rmt_new AS game_data.tl_turret_plate_destroyed
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, teamid, lanetype, killerid);
INSERT INTO game_data.tl_turret_plate_destroyed__rmt_new SELECT * FROM game_data.tl_turret_plate_destroyed;
OPTIMIZE TABLE game_data.tl_turret_plate_destroyed__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_turret_plate_destroyed AND game_data.tl_turret_plate_destroyed__rmt_new;
RENAME TABLE game_data.tl_turret_plate_destroyed__rmt_new TO game_data.tl_turret_plate_destroyed__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_ck_victim_damage_dealt__rmt_new;
CREATE TABLE game_data.tl_ck_victim_damage_dealt__rmt_new AS game_data.tl_ck_victim_damage_dealt
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, champion_kill_event_id, idx);
INSERT INTO game_data.tl_ck_victim_damage_dealt__rmt_new SELECT * FROM game_data.tl_ck_victim_damage_dealt;
OPTIMIZE TABLE game_data.tl_ck_victim_damage_dealt__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_ck_victim_damage_dealt AND game_data.tl_ck_victim_damage_dealt__rmt_new;
RENAME TABLE game_data.tl_ck_victim_damage_dealt__rmt_new TO game_data.tl_ck_victim_damage_dealt__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_ck_victim_damage_received__rmt_new;
CREATE TABLE game_data.tl_ck_victim_damage_received__rmt_new AS game_data.tl_ck_victim_damage_received
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, champion_kill_event_id, idx);
INSERT INTO game_data.tl_ck_victim_damage_received__rmt_new SELECT * FROM game_data.tl_ck_victim_damage_received;
OPTIMIZE TABLE game_data.tl_ck_victim_damage_received__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_ck_victim_damage_received AND game_data.tl_ck_victim_damage_received__rmt_new;
RENAME TABLE game_data.tl_ck_victim_damage_received__rmt_new TO game_data.tl_ck_victim_damage_received__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_ward_placed__rmt_new;
CREATE TABLE game_data.tl_ward_placed__rmt_new AS game_data.tl_ward_placed
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, creatorid, wardtype);
INSERT INTO game_data.tl_ward_placed__rmt_new SELECT * FROM game_data.tl_ward_placed;
OPTIMIZE TABLE game_data.tl_ward_placed__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_ward_placed AND game_data.tl_ward_placed__rmt_new;
RENAME TABLE game_data.tl_ward_placed__rmt_new TO game_data.tl_ward_placed__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_ward_kill__rmt_new;
CREATE TABLE game_data.tl_ward_kill__rmt_new AS game_data.tl_ward_kill
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, killerid, wardtype);
INSERT INTO game_data.tl_ward_kill__rmt_new SELECT * FROM game_data.tl_ward_kill;
OPTIMIZE TABLE game_data.tl_ward_kill__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_ward_kill AND game_data.tl_ward_kill__rmt_new;
RENAME TABLE game_data.tl_ward_kill__rmt_new TO game_data.tl_ward_kill__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_item_purchased__rmt_new;
CREATE TABLE game_data.tl_item_purchased__rmt_new AS game_data.tl_item_purchased
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid, itemid);
INSERT INTO game_data.tl_item_purchased__rmt_new SELECT * FROM game_data.tl_item_purchased;
OPTIMIZE TABLE game_data.tl_item_purchased__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_item_purchased AND game_data.tl_item_purchased__rmt_new;
RENAME TABLE game_data.tl_item_purchased__rmt_new TO game_data.tl_item_purchased__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_item_sold__rmt_new;
CREATE TABLE game_data.tl_item_sold__rmt_new AS game_data.tl_item_sold
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid, itemid);
INSERT INTO game_data.tl_item_sold__rmt_new SELECT * FROM game_data.tl_item_sold;
OPTIMIZE TABLE game_data.tl_item_sold__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_item_sold AND game_data.tl_item_sold__rmt_new;
RENAME TABLE game_data.tl_item_sold__rmt_new TO game_data.tl_item_sold__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_item_destroyed__rmt_new;
CREATE TABLE game_data.tl_item_destroyed__rmt_new AS game_data.tl_item_destroyed
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid, itemid);
INSERT INTO game_data.tl_item_destroyed__rmt_new SELECT * FROM game_data.tl_item_destroyed;
OPTIMIZE TABLE game_data.tl_item_destroyed__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_item_destroyed AND game_data.tl_item_destroyed__rmt_new;
RENAME TABLE game_data.tl_item_destroyed__rmt_new TO game_data.tl_item_destroyed__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_item_undo__rmt_new;
CREATE TABLE game_data.tl_item_undo__rmt_new AS game_data.tl_item_undo
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid, beforeid, afterid);
INSERT INTO game_data.tl_item_undo__rmt_new SELECT * FROM game_data.tl_item_undo;
OPTIMIZE TABLE game_data.tl_item_undo__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_item_undo AND game_data.tl_item_undo__rmt_new;
RENAME TABLE game_data.tl_item_undo__rmt_new TO game_data.tl_item_undo__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_level_up__rmt_new;
CREATE TABLE game_data.tl_level_up__rmt_new AS game_data.tl_level_up
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid, level);
INSERT INTO game_data.tl_level_up__rmt_new SELECT * FROM game_data.tl_level_up;
OPTIMIZE TABLE game_data.tl_level_up__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_level_up AND game_data.tl_level_up__rmt_new;
RENAME TABLE game_data.tl_level_up__rmt_new TO game_data.tl_level_up__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_skill_level_up__rmt_new;
CREATE TABLE game_data.tl_skill_level_up__rmt_new AS game_data.tl_skill_level_up
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid, skillslot, leveluptype);
INSERT INTO game_data.tl_skill_level_up__rmt_new SELECT * FROM game_data.tl_skill_level_up;
OPTIMIZE TABLE game_data.tl_skill_level_up__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_skill_level_up AND game_data.tl_skill_level_up__rmt_new;
RENAME TABLE game_data.tl_skill_level_up__rmt_new TO game_data.tl_skill_level_up__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_pause_end__rmt_new;
CREATE TABLE game_data.tl_pause_end__rmt_new AS game_data.tl_pause_end
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, realtimestamp);
INSERT INTO game_data.tl_pause_end__rmt_new SELECT * FROM game_data.tl_pause_end;
OPTIMIZE TABLE game_data.tl_pause_end__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_pause_end AND game_data.tl_pause_end__rmt_new;
RENAME TABLE game_data.tl_pause_end__rmt_new TO game_data.tl_pause_end__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_game_end__rmt_new;
CREATE TABLE game_data.tl_game_end__rmt_new AS game_data.tl_game_end
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, winningteam, realtimestamp);
INSERT INTO game_data.tl_game_end__rmt_new SELECT * FROM game_data.tl_game_end;
OPTIMIZE TABLE game_data.tl_game_end__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_game_end AND game_data.tl_game_end__rmt_new;
RENAME TABLE game_data.tl_game_end__rmt_new TO game_data.tl_game_end__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_objective_bounty_prestart__rmt_new;
CREATE TABLE game_data.tl_objective_bounty_prestart__rmt_new AS game_data.tl_objective_bounty_prestart
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, teamid, actualstarttime);
INSERT INTO game_data.tl_objective_bounty_prestart__rmt_new SELECT * FROM game_data.tl_objective_bounty_prestart;
OPTIMIZE TABLE game_data.tl_objective_bounty_prestart__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_objective_bounty_prestart AND game_data.tl_objective_bounty_prestart__rmt_new;
RENAME TABLE game_data.tl_objective_bounty_prestart__rmt_new TO game_data.tl_objective_bounty_prestart__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_objective_bounty_finish__rmt_new;
CREATE TABLE game_data.tl_objective_bounty_finish__rmt_new AS game_data.tl_objective_bounty_finish
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, teamid);
INSERT INTO game_data.tl_objective_bounty_finish__rmt_new SELECT * FROM game_data.tl_objective_bounty_finish;
OPTIMIZE TABLE game_data.tl_objective_bounty_finish__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_objective_bounty_finish AND game_data.tl_objective_bounty_finish__rmt_new;
RENAME TABLE game_data.tl_objective_bounty_finish__rmt_new TO game_data.tl_objective_bounty_finish__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_feat_update__rmt_new;
CREATE TABLE game_data.tl_feat_update__rmt_new AS game_data.tl_feat_update
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, teamid, feattype, featvalue);
INSERT INTO game_data.tl_feat_update__rmt_new SELECT * FROM game_data.tl_feat_update;
OPTIMIZE TABLE game_data.tl_feat_update__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_feat_update AND game_data.tl_feat_update__rmt_new;
RENAME TABLE game_data.tl_feat_update__rmt_new TO game_data.tl_feat_update__pre_rmt;

DROP TABLE IF EXISTS game_data.tl_champion_transform__rmt_new;
CREATE TABLE game_data.tl_champion_transform__rmt_new AS game_data.tl_champion_transform
ENGINE = ReplacingMergeTree
ORDER BY (matchid, frame_timestamp, timestamp, participantid, transformtype);
INSERT INTO game_data.tl_champion_transform__rmt_new SELECT * FROM game_data.tl_champion_transform;
OPTIMIZE TABLE game_data.tl_champion_transform__rmt_new FINAL;
EXCHANGE TABLES game_data.tl_champion_transform AND game_data.tl_champion_transform__rmt_new;
RENAME TABLE game_data.tl_champion_transform__rmt_new TO game_data.tl_champion_transform__pre_rmt;

SELECT
    database,
    name,
    engine,
    sorting_key
FROM system.tables
WHERE database = 'game_data'
  AND name IN (
      'matchdata_matchids',
      'participant_perk_values',
      'participant_perk_ids',
      'tl_participant_stats',
      'metadata',
      'info',
      'bans',
      'feats',
      'objectives',
      'participant_stats',
      'participant_challenges',
      'tl_building_kill',
      'tl_champion_kill',
      'tl_champion_special_kill',
      'tl_dragon_soul_given',
      'tl_elite_monster_kill',
      'tl_turret_plate_destroyed',
      'tl_ck_victim_damage_dealt',
      'tl_ck_victim_damage_received',
      'tl_ward_placed',
      'tl_ward_kill',
      'tl_item_purchased',
      'tl_item_sold',
      'tl_item_destroyed',
      'tl_item_undo',
      'tl_level_up',
      'tl_skill_level_up',
      'tl_pause_end',
      'tl_game_end',
      'tl_objective_bounty_prestart',
      'tl_objective_bounty_finish',
      'tl_feat_update',
      'tl_champion_transform'
  )
ORDER BY name;
