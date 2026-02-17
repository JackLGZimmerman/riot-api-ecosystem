CREATE TABLE IF NOT EXISTS game_data.objectives (
    run_id UUID,
    gameId UInt64,
    teamId Enum8('100' = 1, '200' = 2),
    objectiveType Enum(
        'atakhan' = 1,
        'baron' = 2,
        'champion' = 3,
        'dragon' = 4,
        'horde' = 5,
        'inhibitor' = 6,
        'riftHerald' = 7,
        'tower' = 8
    ),
    first UInt8,
    kills UInt8
) ENGINE = MergeTree
ORDER BY (gameId, teamId, objectiveType, run_id);