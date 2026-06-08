CREATE TABLE IF NOT EXISTS game_data.objectives (
    run_id UUID,
    matchid String CODEC (ZSTD(3)),
    teamid UInt8,
    objectivetype Enum8 (
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
ORDER BY (matchid, teamid, objectivetype);
