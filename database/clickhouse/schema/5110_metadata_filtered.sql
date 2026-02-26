CREATE TABLE IF NOT EXISTS game_data_filtered.metadata
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.metadata AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON toUInt64OrNull(arrayElement(splitByChar('_', t.matchid), 2)) = v.matchid
WHERE 0;
