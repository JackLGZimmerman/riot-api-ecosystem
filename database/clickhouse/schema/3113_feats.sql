CREATE TABLE IF NOT EXISTS game_data.feats
(
    run_id UUID,
    gameId UInt64,
    teamId Enum8('100' = 1, '200' = 2),
    featType Enum('EPIC_MONSTER_KILL' = 1, 'FIRST_BLOOD' = 2, 'FIRST_TURRET' = 3),
    featState Bool
)
ENGINE = MergeTree
ORDER BY (gameId, teamId, featType, run_id);
