-- noqa: disable=LT01,LT05,PRS
--
-- Build-sibling backoff level for 6004: same-team 2vx synergy priors with
-- item-derived builds collapsed into hand-defined sibling groups. Unlisted
-- builds remain as their own group.

DROP TABLE IF EXISTS game_data_filtered.synergy_2vx_build_group;

CREATE TABLE IF NOT EXISTS game_data_filtered.synergy_2vx_build_group
(
    split LowCardinality(String),
    championid_1 Int32,
    championname_1 LowCardinality(String),
    teamposition_1 LowCardinality(String),
    build_group_1 LowCardinality(String),
    championid_2 Int32,
    championname_2 LowCardinality(String),
    teamposition_2 LowCardinality(String),
    build_group_2 LowCardinality(String),
    matchups UInt64,
    wins UInt64,
    losses UInt64,
    win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    championid_1, teamposition_1, build_group_1,
    championid_2, teamposition_2, build_group_2
);
