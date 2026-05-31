-- noqa: disable=LT01,LT05,PRS
--
-- Backoff level for 6004: same-team 2vx synergy priors with the item-derived
-- build dropped from both members. Keyed on (championid, teamposition) per
-- member, canonicalised smaller-tuple-first (win_rate is own-team, symmetric).

DROP TABLE IF EXISTS game_data_filtered.synergy_2vx_nobuild;

CREATE TABLE IF NOT EXISTS game_data_filtered.synergy_2vx_nobuild
(
    split LowCardinality(String),
    championid_1 Int32,
    teamposition_1 LowCardinality(String),
    championid_2 Int32,
    teamposition_2 LowCardinality(String),
    matchups UInt64,
    win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    championid_1, teamposition_1,
    championid_2, teamposition_2
);
