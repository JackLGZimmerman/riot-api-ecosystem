CREATE TABLE IF NOT EXISTS game_data.bans
(
    run_id UUID,
    gameid UInt64,
    teamid Enum8 ('100' = 1, '200' = 2),
    pickturn Enum8 (
        '1' = 1,
        '2' = 2,
        '3' = 3,
        '4' = 4,
        '5' = 5,
        '6' = 6,
        '7' = 7,
        '8' = 8,
        '9' = 9,
        '10' = 10
    ),
    championid UInt16
)
ENGINE = MergeTree
ORDER BY (gameid, teamid, pickturn, run_id);
