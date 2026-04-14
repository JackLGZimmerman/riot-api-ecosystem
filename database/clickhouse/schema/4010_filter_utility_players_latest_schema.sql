DROP TABLE IF EXISTS game_data.filter_utility_players_latest;

CREATE TABLE game_data.filter_utility_players_latest
(
    puuid FixedString (78),
    wins UInt16,
    losses UInt16
)
ENGINE = MergeTree
ORDER BY puuid;
