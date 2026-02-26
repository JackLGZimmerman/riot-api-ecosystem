CREATE TABLE IF NOT EXISTS game_data.feats
(
    run_id UUID,
    matchid UInt64,
    teamid UInt8,
    feattype ENUM8 ('EPIC_MONSTER_KILL' = 1, 'FIRST_BLOOD' = 2, 'FIRST_TURRET' = 3),
    featstate UInt16
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, feattype, run_id);
