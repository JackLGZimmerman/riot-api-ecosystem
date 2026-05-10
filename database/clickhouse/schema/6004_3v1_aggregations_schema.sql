-- noqa: disable=LT01,LT05,PRS

-- 3v1 matchup priors: a trio from one team versus a single player from the
-- other team. Storage is asymmetric (trio on the left, single on the right);
-- players within the trio are sorted by (championid, teamposition, build).
-- Each match contributes 10 (trio) × 5 (single) = 50 rows per direction
-- and both directions are stored, so 100 rows per match.

DROP TABLE IF EXISTS game_data_filtered.matchup_3v1;

CREATE TABLE IF NOT EXISTS game_data_filtered.matchup_3v1
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
    single_championid Int32,
    single_championname LowCardinality(String),
    single_teamposition LowCardinality(String),
    single_build LowCardinality(String),
    matchups UInt64,
    trio_wins UInt64,
    single_wins UInt64,
    trio_win_rate Float32,
    single_win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    trio_championid_1, trio_teamposition_1, trio_build_1,
    trio_championid_2, trio_teamposition_2, trio_build_2,
    trio_championid_3, trio_teamposition_3, trio_build_3,
    single_championid, single_teamposition, single_build
);
