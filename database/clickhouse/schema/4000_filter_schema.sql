-- Filter pipeline: complete table DDL for the 2-stage filter.
--
-- Pre-filter (f14): games with gameduration > 990 s are collected into
--          filter_stg_f14_long_games first. All subsequent stages operate
--          exclusively on this long-game population via SEMI JOIN.
-- Stage 1: cheap per-participant / per-team filters scanned over f14_long_games.
-- Stage 2: build label computation + low-build-value detection
--          (highest_value < 1.0) over stage-1-clean games only.
--
-- Run with clickhouse-client --multiquery before 4000_filter_build.sql.

-- Pre-filter stage: long games only (gameduration > 990 s).
DROP TABLE IF EXISTS game_data.filter_stg_f14_long_games;

CREATE TABLE game_data.filter_stg_f14_long_games
(
    matchid String
)
ENGINE = MergeTree
ORDER BY matchid;

DROP TABLE IF EXISTS game_data.filter_stg_player_winrates;

CREATE TABLE game_data.filter_stg_player_winrates
(
    puuid FixedString (78),
    wins UInt16,
    losses UInt16
)
ENGINE = MergeTree
ORDER BY puuid;

DROP TABLE IF EXISTS game_data.filter_stg_player_role_rates;

CREATE TABLE game_data.filter_stg_player_role_rates
(
    puuid FixedString (78),
    teamposition LowCardinality (String),
    role_games UInt32,
    total_games UInt32
)
ENGINE = MergeTree
ORDER BY (puuid, teamposition);

-- Per-game player_high_winrate flag: precomputed via suffix-WR trim
-- over the games of suspect players (lifetime > 40 games AND WR > 70%).
-- See 4000_filter_build.sql for the trim logic.
DROP TABLE IF EXISTS game_data.filter_stg_player_high_winrate_flags;

CREATE TABLE game_data.filter_stg_player_high_winrate_flags
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    player_high_winrate UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, participantid);

DROP TABLE IF EXISTS game_data.filter_stg_team_flags;

CREATE TABLE game_data.filter_stg_team_flags
(
    matchid String,
    teamid UInt8,
    team_kills UInt16,
    team_damage_to_champions UInt32,
    team_kills_to_deaths UInt8,
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy UInt8,
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, teamid);

-- Stage 1 output: per-participant cheap flags.
-- game_time_lte_16_5 is applied as a base-population pre-filter at the scan
-- level; short games never enter this table, so that column is absent.
DROP TABLE IF EXISTS game_data.filter_stg_participant_flags;

CREATE TABLE game_data.filter_stg_participant_flags
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    player_low_kda UInt8,
    player_gold_spent UInt8,
    player_high_winrate UInt8,
    team_kills_to_deaths UInt8,
    solo_carried UInt8,
    too_little_damage UInt8,
    low_minions_killed UInt8,
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy UInt8,
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, participantid);

-- Tiny helper: matchids that pass every stage-1 filter.
DROP TABLE IF EXISTS game_data.filter_stg_stage1_valid_matchids;

CREATE TABLE game_data.filter_stg_stage1_valid_matchids
(
    matchid String
)
ENGINE = MergeTree
ORDER BY matchid;

-- Stage 2 label staging: precomputes each stage-1-clean participant's
-- build label (greatest value across the 11 archetypes, or 'none' when the
-- participant has no item value).
DROP TABLE IF EXISTS game_data.filter_stg_participant_labels;

CREATE TABLE game_data.filter_stg_participant_labels
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    championid Nullable (Int32),
    teamposition LowCardinality (String),
    highest_value Float32,
    highest_value_label LowCardinality (String),
    low_build_value UInt8
)
ENGINE = MergeTree
ORDER BY (teamposition, matchid, teamid, participantid);

DROP TABLE IF EXISTS game_data.filter_stg_game_flags;

CREATE TABLE game_data.filter_stg_game_flags
(
    matchid String,
    player_low_kda UInt8,
    player_gold_spent UInt8,
    player_high_winrate UInt8,
    team_kills_to_deaths UInt8,
    solo_carried UInt8,
    too_little_damage UInt8,
    low_minions_killed UInt8,
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy UInt8,
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy UInt8,
    low_build_value UInt8,
    any_filter_triggered UInt8
)
ENGINE = MergeTree
ORDER BY matchid;

DROP TABLE IF EXISTS game_data.filter_result;

CREATE TABLE game_data.filter_result
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    player_rule_mask UInt32,
    team_rule_mask UInt32,
    game_rule_mask UInt32,
    rule_mask UInt32,
    is_valid UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, participantid);
