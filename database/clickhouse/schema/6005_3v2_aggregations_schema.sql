-- noqa: disable=LT01,LT05,PRS

-- 3v2 matchup priors: a trio from one team versus a pair from the other
-- team. Storage is asymmetric (trio on the left, pair on the right);
-- players within the trio and pair are sorted by
-- (championid, teamposition, build). Each match contributes
-- 10 (trio) x 10 (pair) = 100 rows per direction and both directions are
-- stored, so 200 rows per match.

DROP TABLE IF EXISTS game_data_filtered.matchup_3v2;

CREATE TABLE IF NOT EXISTS game_data_filtered.matchup_3v2
(
    split LowCardinality(String),
    trio_championid_1 Int32,
    trio_championname_1 LowCardinality(String),
    trio_teamposition_1 LowCardinality(String),
    trio_build_1 LowCardinality(String),
    trio_championid_2 Int32,
    trio_championname_2 LowCardinality(String),
    trio_teamposition_2 LowCardinality(String),
    trio_build_2 LowCardinality(String),
    trio_championid_3 Int32,
    trio_championname_3 LowCardinality(String),
    trio_teamposition_3 LowCardinality(String),
    trio_build_3 LowCardinality(String),
    pair_championid_1 Int32,
    pair_championname_1 LowCardinality(String),
    pair_teamposition_1 LowCardinality(String),
    pair_build_1 LowCardinality(String),
    pair_championid_2 Int32,
    pair_championname_2 LowCardinality(String),
    pair_teamposition_2 LowCardinality(String),
    pair_build_2 LowCardinality(String),
    matchups UInt64,
    trio_wins UInt64,
    pair_wins UInt64,
    trio_win_rate Float32,
    pair_win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    trio_championid_1, trio_teamposition_1, trio_build_1,
    trio_championid_2, trio_teamposition_2, trio_build_2,
    trio_championid_3, trio_teamposition_3, trio_build_3,
    pair_championid_1, pair_teamposition_1, pair_build_1,
    pair_championid_2, pair_teamposition_2, pair_build_2
);
