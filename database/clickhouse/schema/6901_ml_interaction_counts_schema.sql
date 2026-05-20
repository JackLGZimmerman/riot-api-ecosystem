-- noqa: disable=LT01,LT05,PRS

-- Per-game per-token 1vX historical profile features materialised by joining
-- each game's role-pivot players against the reduced `synergy_1vx` table.
-- `synergy_1vx` is keyed by (championid, teamposition, build, bin_idx), so
-- this table preserves one row per player token and scaling bin. Rows are
-- sparse: token/bin slots without a train-split aggregate match do not appear.
--
-- token_idx layout:
--   0..4 = blue TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY
--   5..9 = red  TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY
--
-- bin_idx is the 6002_1vx scaling bin:
--   1 early-mid, 2 mid, 3 mid-late, 4 late

DROP TABLE IF EXISTS game_data_filtered.ml_interaction_counts;

CREATE TABLE IF NOT EXISTS game_data_filtered.ml_interaction_counts
(
    matchid String,
    token_idx UInt16,
    bin_idx UInt8,

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
ORDER BY (matchid, token_idx, bin_idx);
