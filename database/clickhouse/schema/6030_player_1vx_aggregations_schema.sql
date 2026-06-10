-- noqa: disable=LT01,LT05,PRS
--
-- Draft-time per-player priors for the ML model: each player's train-split
-- game count and win rate across all champions. Train-scoped only, like the
-- champion/role/build priors in synergy_1vx.

DROP TABLE IF EXISTS game_data_filtered.player_1vx;

CREATE TABLE IF NOT EXISTS game_data_filtered.player_1vx
(
    split LowCardinality(String),
    puuid String,
    matchups UInt32,
    win_rate Float32
)
ENGINE = MergeTree
ORDER BY (split, puuid);
