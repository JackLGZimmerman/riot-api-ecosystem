CREATE TABLE IF NOT EXISTS game_data.tl_participant_stats
(
    run_id UUID,
    matchid UInt64,
    frame_timestamp UInt32,
    participantid UInt8,

    abilityhaste UInt16,
    abilitypower UInt16,
    armor Int16,
    attackdamage UInt16,
    attackspeed UInt16,
    ccreduction Int8,
    cooldownreduction UInt16,
    health UInt16,
    healthmax UInt16,
    healthregen UInt16,
    magicresist Int16,
    movementspeed UInt16,
    power UInt16,
    powermax UInt16,
    powerregen UInt16,
    payload Map (LowCardinality (String), UInt32),

    currentgold Int32,

    magicdamagedone UInt32,
    magicdamagedonetochampions UInt32,
    magicdamagetaken UInt32,
    physicaldamagedone UInt32,
    physicaldamagedonetochampions UInt32,
    physicaldamagetaken UInt32,
    totaldamagedone UInt32,
    totaldamagedonetochampions UInt32,
    totaldamagetaken UInt32,
    truedamagedone UInt32,
    truedamagedonetochampions UInt32,
    truedamagetaken UInt32,

    goldpersecond UInt16,
    jungleminionskilled UInt16,
    level UInt8,
    minionskilled UInt16,
    position_x UInt16,
    position_y UInt16,
    timeenemyspentcontrolled UInt32,
    totalgold UInt16,
    xp UInt16
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, participantid, run_id);
