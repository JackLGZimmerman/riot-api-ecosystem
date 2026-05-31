-- noqa: disable=PRS
CREATE DATABASE IF NOT EXISTS game_data_filtered;

-- The filtered database now keeps only tables used by the production ML and
-- classification feature paths. Raw match snapshots, metadata, most timeline
-- event copies, temporal classification artifacts, and retired experiment
-- aggregations are derived or obsolete and should be dropped rather than
-- recreated here.
DROP TABLE IF EXISTS game_data_filtered.metadata;
DROP TABLE IF EXISTS game_data_filtered.info;
DROP TABLE IF EXISTS game_data_filtered.bans;
DROP TABLE IF EXISTS game_data_filtered.feats;
DROP TABLE IF EXISTS game_data_filtered.objectives;
DROP TABLE IF EXISTS game_data_filtered.participant_challenges;
DROP TABLE IF EXISTS game_data_filtered.participant_perk_values;
DROP TABLE IF EXISTS game_data_filtered.participant_perk_ids;
DROP TABLE IF EXISTS game_data_filtered.tl_ward_placed;
DROP TABLE IF EXISTS game_data_filtered.tl_ward_kill;
DROP TABLE IF EXISTS game_data_filtered.tl_item_purchased;
DROP TABLE IF EXISTS game_data_filtered.tl_item_sold;
DROP TABLE IF EXISTS game_data_filtered.tl_item_destroyed;
DROP TABLE IF EXISTS game_data_filtered.tl_item_undo;
DROP TABLE IF EXISTS game_data_filtered.tl_level_up;
DROP TABLE IF EXISTS game_data_filtered.tl_skill_level_up;
DROP TABLE IF EXISTS game_data_filtered.tl_pause_end;
DROP TABLE IF EXISTS game_data_filtered.tl_game_end;
DROP TABLE IF EXISTS game_data_filtered.tl_objective_bounty_prestart;
DROP TABLE IF EXISTS game_data_filtered.tl_objective_bounty_finish;
DROP TABLE IF EXISTS game_data_filtered.tl_feat_update;
DROP TABLE IF EXISTS game_data_filtered.tl_champion_transform;
DROP TABLE IF EXISTS game_data_filtered.tl_building_kill;
DROP TABLE IF EXISTS game_data_filtered.tl_champion_kill;
DROP TABLE IF EXISTS game_data_filtered.tl_champion_special_kill;
DROP TABLE IF EXISTS game_data_filtered.tl_dragon_soul_given;
DROP TABLE IF EXISTS game_data_filtered.tl_elite_monster_kill;
DROP TABLE IF EXISTS game_data_filtered.tl_turret_plate_destroyed;
DROP TABLE IF EXISTS game_data_filtered.tl_ck_victim_damage_dealt;
DROP TABLE IF EXISTS game_data_filtered.tl_ck_victim_damage_received;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_prior_sibling;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_prior_champion_role;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_prior_role_build;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_prior_champion_build;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_prior_build;
DROP TABLE IF EXISTS game_data_filtered.matchup_3v2;
DROP TABLE IF EXISTS game_data_filtered.matchup_3v3;
DROP TABLE IF EXISTS game_data_filtered.synergy_2v2;
DROP TABLE IF EXISTS game_data_filtered.synergy_3v3;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_recent_s16_synergy_1vx_dict;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_recent_s16_matchup_1v1_dict;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_recent_s16_synergy_2vx_dict;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_recent_s16_matchup_1v1_nobuild_dict;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_recent_s16_matchup_1v1_champ_dict;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_recent_s16_synergy_2vx_nobuild_dict;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_recent_s16_synergy_2vx_champ_dict;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_smoke_synergy_1vx_unknown_dict;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_smoke_matchup_1v1_unknown_dict;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_smoke_synergy_2vx_unknown_dict;
DROP DICTIONARY IF EXISTS game_data_filtered.hgnn_synergy_3vx_dict;
DROP TABLE IF EXISTS game_data_filtered.hgnn_recent_s16_ml_game_player_pivot;
DROP TABLE IF EXISTS game_data_filtered.hgnn_recent_s16_synergy_1vx;
DROP TABLE IF EXISTS game_data_filtered.hgnn_recent_s16_matchup_1v1;
DROP TABLE IF EXISTS game_data_filtered.hgnn_recent_s16_synergy_2vx;
DROP TABLE IF EXISTS game_data_filtered.hgnn_recent_s16_matchup_1v1_nobuild;
DROP TABLE IF EXISTS game_data_filtered.hgnn_recent_s16_matchup_1v1_champ;
DROP TABLE IF EXISTS game_data_filtered.hgnn_recent_s16_synergy_2vx_nobuild;
DROP TABLE IF EXISTS game_data_filtered.hgnn_recent_s16_synergy_2vx_champ;
DROP TABLE IF EXISTS game_data_filtered.hgnn_smoke_ml_game_player_pivot_unknown;
DROP TABLE IF EXISTS game_data_filtered.hgnn_smoke_synergy_1vx_unknown;
DROP TABLE IF EXISTS game_data_filtered.hgnn_smoke_matchup_1v1_unknown;
DROP TABLE IF EXISTS game_data_filtered.hgnn_smoke_synergy_2vx_unknown;

-- Persistent ML input table populated by 5003_filtered_tables_build.sql.
DROP TABLE IF EXISTS game_data_filtered.participant_stats;
CREATE TABLE IF NOT EXISTS game_data_filtered.participant_stats
AS game_data.participant_stats
ENGINE = MergeTree
ORDER BY (matchid, participantid, puuid, run_id);

-- Classification embeddings still consume final participant timeline states.
DROP TABLE IF EXISTS game_data_filtered.tl_participant_stats;
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_participant_stats
AS game_data.tl_participant_stats
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, participantid, run_id);

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
