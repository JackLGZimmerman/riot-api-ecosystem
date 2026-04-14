TRUNCATE TABLE game_data.filter_utility_players_latest;

INSERT INTO game_data.filter_utility_players_latest
(
    puuid,
    wins,
    losses
)
SELECT
    puuid,
    argMax(wins, updated_at) AS wins,
    argMax(losses, updated_at) AS losses
FROM game_data.players
GROUP BY puuid;
