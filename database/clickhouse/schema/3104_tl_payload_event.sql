CREATE TABLE IF NOT EXISTS game_data.tl_payload_event
(
    run_id UUID,
    matchid UInt64,
    frame_timestamp UInt32,
    type Enum8 (
        'WARD_KILL' = 1,
        'WARD_PLACED' = 2,
        'GAME_END' = 3,
        'ITEM_DESTROYED' = 4,
        'ITEM_PURCHASED' = 5,
        'ITEM_SOLD' = 6,
        'ITEM_UNDO' = 7,
        'LEVEL_UP' = 8,
        'PAUSE_END' = 9,
        'SKILL_LEVEL_UP' = 10,
        'OBJECTIVE_BOUNTY_PRESTART' = 11,
        'FEAT_UPDATE' = 12,
        'OBJECTIVE_BOUNTY_FINISH' = 13,
        'CHAMPION_TRANSFORM' = 14
    ),
    timestamp UInt64,
    payload JSON
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, timestamp, type, run_id);
