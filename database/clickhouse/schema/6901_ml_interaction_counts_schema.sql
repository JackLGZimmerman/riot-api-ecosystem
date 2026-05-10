-- noqa: disable=LT01,LT05,PRS

-- Per-game per-token (matchups, primary_wins) materialised by joining each
-- game's role-pivot players against the model-consumed 6xxx aggregate tables.
-- Aggregate joins are filtered to matchups >= 5 to keep right-side cardinality
-- tractable. Rows are sparse:
-- token slots without an aggregate match (or with matchups < 5) simply do
-- not appear. The Python cache builder fills missing tokens with zeros and
-- applies leave-one-out + Bayesian smoothing in numpy.
--
-- token_idx layout (matches app/ml/config.INTERACTION_TYPES order):
--    0..4      1vX synergy blue (5)
--    5..9      1vX synergy red  (5)
--   10..34     1v1 matchups (5 blue x 5 red)
--   35..44     2vX synergy blue (10)
--   45..54     2vX synergy red  (10)
--   55..104    2v1 blue pair vs red single (10 x 5)
--  105..154    2v1 red pair  vs blue single (10 x 5)
--  155..164    3vX synergy blue (10)
--  165..174    3vX synergy red  (10)
--
-- primary_wins is the token's primary side's wins:
--   1v1 cross tokens: blue's wins
--   2v1 tokens: pair's wins (matches matchup_2v1.pair_wins)
--   1vX/2vX/3vX synergy tokens: combo's total wins

DROP TABLE IF EXISTS game_data_filtered.ml_interaction_counts;

CREATE TABLE IF NOT EXISTS game_data_filtered.ml_interaction_counts
(
    matchid String,
    token_idx UInt16,
    matchups UInt64,
    primary_wins UInt64
)
ENGINE = MergeTree
ORDER BY (matchid, token_idx);
