-- noqa: disable=LT01,LT05,PRS
--
-- Coarsest champion-pair backoff for 6000: 1v1 matchup priors keyed on
-- championid only (build and teamposition dropped, lane-agnostic). Stored
-- DIRECTIONALLY (blue-perspective); the cache builder reads blue_win_rate
-- without an orientation flip.

DROP TABLE IF EXISTS game_data_filtered.matchup_1v1_champ;

CREATE TABLE IF NOT EXISTS game_data_filtered.matchup_1v1_champ
(
    split LowCardinality(String),
    blue_championid Int32,
    red_championid Int32,
    matchups UInt64,
    blue_win_rate Float32
)
ENGINE = MergeTree
ORDER BY (split, blue_championid, red_championid);
