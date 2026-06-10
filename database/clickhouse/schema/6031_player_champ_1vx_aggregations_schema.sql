-- noqa: disable=LT01,LT05,PRS
--
-- Draft-time per-(player, champion) priors for the ML model: train-split
-- game count and win rate of each player on each champion. Captures champion
-- experience/mastery, which matchmaking does not balance. Train-scoped only.

DROP TABLE IF EXISTS game_data_filtered.player_champ_1vx;

CREATE TABLE IF NOT EXISTS game_data_filtered.player_champ_1vx
(
    split LowCardinality(String),
    puuid String,
    championid Int32,
    matchups UInt32,
    win_rate Float32
)
ENGINE = MergeTree
ORDER BY (split, puuid, championid);
