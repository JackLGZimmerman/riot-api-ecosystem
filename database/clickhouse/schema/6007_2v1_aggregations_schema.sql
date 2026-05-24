-- noqa: disable=LT01,LT05,PRS

-- 2v1 matchup priors: a pair from one team versus a single player from the
-- other team. Storage is asymmetric (pair on the left, single on the right);
-- players within the pair are sorted by (championid, teamposition, build).
-- Each match contributes 10 (pair) × 5 (single) = 50 rows per direction
-- and both directions are stored, so 100 rows per match.

DROP TABLE IF EXISTS game_data_filtered.matchup_2v1;

CREATE TABLE IF NOT EXISTS game_data_filtered.matchup_2v1
(
    split LowCardinality(String),
    pair_championid_1 Int32,
    pair_championname_1 LowCardinality(String),
    pair_teamposition_1 LowCardinality(String),
    pair_build_1 LowCardinality(String),
    pair_championid_2 Int32,
    pair_championname_2 LowCardinality(String),
    pair_teamposition_2 LowCardinality(String),
    pair_build_2 LowCardinality(String),
    single_championid Int32,
    single_championname LowCardinality(String),
    single_teamposition LowCardinality(String),
    single_build LowCardinality(String),
    matchups UInt64,
    pair_wins UInt64,
    single_wins UInt64,
    pair_win_rate Float32,
    single_win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    pair_championid_1, pair_teamposition_1, pair_build_1,
    pair_championid_2, pair_teamposition_2, pair_build_2,
    single_championid, single_teamposition, single_build
);
