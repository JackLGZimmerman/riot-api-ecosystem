-- noqa: disable=LT01,LT05,PRS

-- Per-game per-token 1vX build-labeled win-rate priors materialised by
-- joining each game's role-pivot players against `synergy_1vx`.
--
-- token_idx layout:
--   0..4 = blue TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY
--   5..9 = red  TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY

DROP TABLE IF EXISTS game_data_filtered.ml_interaction_counts;

CREATE TABLE IF NOT EXISTS game_data_filtered.ml_interaction_counts
(
    matchid String,
    token_idx UInt16,
    championid Int32,
    teamposition LowCardinality(String),
    build LowCardinality(String),
    matchups UInt32,
    win_rate Float32
)
ENGINE = MergeTree
ORDER BY (matchid, token_idx);
