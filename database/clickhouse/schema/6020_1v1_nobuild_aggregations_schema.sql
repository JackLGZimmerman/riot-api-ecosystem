-- noqa: disable=LT01,LT05,PRS
--
-- Backoff level for 6000: 1v1 matchup priors with the item-derived build
-- dropped from both members. Keyed on (championid, teamposition) per side.
-- Stored DIRECTIONALLY (blue-perspective), not canonicalised, so the cache
-- builder reads blue_win_rate without an orientation flip.

DROP TABLE IF EXISTS game_data_filtered.matchup_1v1_nobuild;

CREATE TABLE IF NOT EXISTS game_data_filtered.matchup_1v1_nobuild
(
    split LowCardinality(String),
    blue_championid Int32,
    blue_teamposition LowCardinality(String),
    red_championid Int32,
    red_teamposition LowCardinality(String),
    matchups UInt64,
    blue_win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    blue_championid, blue_teamposition,
    red_championid, red_teamposition
);
