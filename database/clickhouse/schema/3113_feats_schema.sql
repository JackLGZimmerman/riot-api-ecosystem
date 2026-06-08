CREATE TABLE IF NOT EXISTS game_data.feats
(
    run_id UUID,
    matchid String CODEC (ZSTD(3)),
    teamid UInt8,
    feattype Enum8 ('EPIC_MONSTER_KILL' = 1, 'FIRST_BLOOD' = 2, 'FIRST_TURRET' = 3),
    featstate UInt16
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, feattype);
