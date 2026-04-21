-- Filter pipeline: complete table DDL for the 3-stage filter.
--
-- Stage 1: 13 cheap per-participant / per-team / per-game filters. Computed
--          in a single scan of participant_stats.
-- Stage 2: rare role detection. Pick counts and percentages are evaluated
--          ONLY over games that passed stage 1, so the "rare" threshold is
--          measured against a cleaned pool rather than the full raw data.
-- Stage 3: rare build detection. Build-label counts are evaluated ONLY over
--          games that passed stage 1 AND stage 2.
--
-- Why 3 stages: rare_role and rare_build are population-dependent filters
-- (they compare per-combo counts against global totals); re-evaluating them
-- against the post-stage-1 pool keeps their "rare" threshold meaningful.
--
-- Run with clickhouse-client --multiquery before 4001_filter_build.sql.

DROP TABLE IF EXISTS game_data.filter_stg_player_winrates;

CREATE TABLE game_data.filter_stg_player_winrates
(
    puuid FixedString (78),
    wins UInt16,
    losses UInt16
)
ENGINE = MergeTree
ORDER BY puuid;

DROP TABLE IF EXISTS game_data.filter_stg_team_flags;

CREATE TABLE game_data.filter_stg_team_flags
(
    matchid String,
    teamid UInt8,
    team_kills UInt16,
    team_damage_to_champions UInt32,
    team_kills_to_deaths UInt8,
    team_non_utility_avg_cs_per_min_gt_2_5_below_enemy UInt8,
    team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, teamid);

-- Stage 1 output: per-participant cheap flags only.
-- Does NOT include has_rare_role or rare_build_label; those are stored in
-- filter_stg_rare_roles / filter_stg_rare_builds and joined at the final
-- rollup.  Keeping this table narrow avoids a second full scan of
-- participant_stats later in the pipeline.
DROP TABLE IF EXISTS game_data.filter_stg_participant_flags;

CREATE TABLE game_data.filter_stg_participant_flags
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    player_low_kda UInt8,
    player_gold_spent UInt8,
    no_contribution_kda UInt8,
    bad_summoner_usage UInt8,
    player_high_winrate UInt8,
    team_kills_to_deaths UInt8,
    solo_carried UInt8,
    too_little_damage UInt8,
    low_minions_killed UInt8,
    team_non_utility_avg_cs_per_min_gt_2_5_below_enemy UInt8,
    team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy UInt8,
    sold_all_items UInt8,
    grief_build UInt8,
    game_time_lte_18 UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, participantid);

-- Tiny helper: matchids that pass every stage-1 filter.  Produced by rolling
-- up filter_stg_participant_flags.  Used as the right-hand side of SEMI JOIN
-- in stage 2 so the rare-role scan only touches the cleaned pool.
DROP TABLE IF EXISTS game_data.filter_stg_stage1_valid_matchids;

CREATE TABLE game_data.filter_stg_stage1_valid_matchids
(
    matchid String
)
ENGINE = MergeTree
ORDER BY matchid;

DROP TABLE IF EXISTS game_data.filter_stg_rare_roles;

CREATE TABLE game_data.filter_stg_rare_roles
(
    matchid String,
    has_rare_role UInt8
)
ENGINE = MergeTree
ORDER BY matchid;

-- Tiny helper: matchids that pass stage 1 AND stage 2.  Used as the SEMI
-- JOIN side for stage 3 so the rare-build scan sees an even smaller pool.
DROP TABLE IF EXISTS game_data.filter_stg_stage2_valid_matchids;

CREATE TABLE game_data.filter_stg_stage2_valid_matchids
(
    matchid String
)
ENGINE = MergeTree
ORDER BY matchid;

-- Stage 3 label staging: precomputes each stage-2-clean participant's
-- build label (greatest value across the 12 archetypes, or 'none' when the
-- participant has no item value).  Persisting this avoids re-scanning
-- participant_stats once per iteration when pruning cascading rare builds.
DROP TABLE IF EXISTS game_data.filter_stg_participant_labels;

CREATE TABLE game_data.filter_stg_participant_labels
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    championid Nullable (Int32),
    teamposition LowCardinality (String),
    highest_value_label LowCardinality (String)
)
ENGINE = MergeTree
ORDER BY (teamposition, matchid, teamid, participantid);

DROP TABLE IF EXISTS game_data.filter_stg_rare_builds;

CREATE TABLE game_data.filter_stg_rare_builds
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    rare_build_label UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, participantid);

DROP TABLE IF EXISTS game_data.filter_stg_game_flags;

CREATE TABLE game_data.filter_stg_game_flags
(
    matchid String,
    player_low_kda UInt8,
    player_gold_spent UInt8,
    no_contribution_kda UInt8,
    bad_summoner_usage UInt8,
    player_high_winrate UInt8,
    team_kills_to_deaths UInt8,
    solo_carried UInt8,
    too_little_damage UInt8,
    low_minions_killed UInt8,
    team_non_utility_avg_cs_per_min_gt_2_5_below_enemy UInt8,
    team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy UInt8,
    sold_all_items UInt8,
    grief_build UInt8,
    game_time_lte_18 UInt8,
    has_rare_role UInt8,
    rare_build_label UInt8,
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
