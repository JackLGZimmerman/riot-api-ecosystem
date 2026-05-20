-- noqa: disable=LT01,LT05,PRS
--
-- Reduced per-(champion, role, build, scaling-bin) historical profile features.
-- Every avg_* column is a per-minute rate (per-game stat / game minutes,
-- averaged over the bin); no _per_min suffix is used. NOT per-minute:
-- log_matchups, win_rate, damage shares, damage_to_taken_ratio, and
-- avg_item_completions. See 6002_1vx_aggregations_build.sql for the
-- definition of every derived and compound metric.

DROP TABLE IF EXISTS game_data_filtered.synergy_1vx;

CREATE TABLE IF NOT EXISTS game_data_filtered.synergy_1vx
(
    split LowCardinality(String),
    championid Int32,
    championname LowCardinality(String),
    teamposition LowCardinality(String),
    build LowCardinality(String),
    bin_idx UInt8,
    bin_label LowCardinality(String),

    log_matchups Float32,
    win_rate Float32,

    avg_gold Float32,
    avg_xp Float32,
    avg_item_completions Float32,
    avg_total_cs Float32,

    avg_kills Float32,
    avg_kills_assists Float32,
    avg_total_damage_dealt Float32,

    physical_damage_share Float32,
    magic_damage_share Float32,
    true_damage_share Float32,

    avg_damage_taken Float32,
    avg_durability Float32,
    damage_to_taken_ratio Float32,

    avg_time_ccing_others Float32,
    avg_protection Float32,

    avg_epic_monster_takedowns Float32,
    avg_turret_takedowns Float32,
    avg_damage_to_objectives Float32,

    avg_vision_score Float32,
    avg_control_wards_bought Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    championid, teamposition, build,
    bin_idx
);
