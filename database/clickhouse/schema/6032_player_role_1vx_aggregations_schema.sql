-- noqa: disable=LT01,LT05,PRS
--
-- Draft-time per-(player, role) experience for the ML model: train-split game
-- count of each player on each teamposition. Captures role experience, which
-- matchmaking does not balance (probe2: +0.27pp validation accuracy on top of
-- the player and player-champion priors; the win rate added nothing, so only
-- the count is stored). Train-scoped only.

DROP TABLE IF EXISTS game_data_filtered.player_role_1vx;

CREATE TABLE IF NOT EXISTS game_data_filtered.player_role_1vx
(
    split LowCardinality(String),
    puuid String,
    teamposition LowCardinality(String),
    matchups UInt32
)
ENGINE = MergeTree
ORDER BY (split, puuid, teamposition);
