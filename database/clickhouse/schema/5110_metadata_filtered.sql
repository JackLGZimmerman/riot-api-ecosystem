CREATE VIEW IF NOT EXISTS game_data_filtered.metadata AS
SELECT t.*
FROM game_data.metadata AS t
ANY INNER JOIN game_data_filtered.valid_game_ids AS v
    ON toUInt64OrNull(arrayElement(splitByChar('_', t.matchid), 2)) = v.gameid;
