-- noqa: disable=LT01,LT05,PRS
--
-- Simple draft-time champion/role/build priors for the ML model.
--
-- This intentionally excludes the previous time-bin and historical profile
-- feature columns. The model input gets one scalar per player token: the
-- train-split win rate for that exact (championid, teamposition, build).

DROP TABLE IF EXISTS game_data_filtered.synergy_1vx;

CREATE TABLE IF NOT EXISTS game_data_filtered.synergy_1vx
(
    split LowCardinality(String),
    championid Int32,
    championname LowCardinality(String),
    teamposition LowCardinality(String),
    build LowCardinality(String),
    matchups UInt32,
    win_rate Float32
)
ENGINE = MergeTree
ORDER BY (
    split,
    championid,
    teamposition,
    build
);
