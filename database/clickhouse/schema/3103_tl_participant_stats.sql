CREATE TABLE IF NOT EXISTS game_data.tl_participant_stats
(
    run_id UUID,
    frame_timestamp UInt32,
    participantid UInt8,

    abilityhaste UInt16,
    abilitypower UInt16,
    armor UInt16,
    attackdamage UInt16,
    attackspeed UInt16,
    ccreduction UInt16,
    cooldownreduction UInt16,
    health UInt16,
    healthmax UInt16,
    healthregen UInt8,
    magicresist UInt16,
    movementspeed UInt16,
    power UInt16,
    powermax UInt16,
    powerregen UInt8,
    payload Map (LowCardinality (String), UInt32),

    currentgold UInt16,

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

    goldpersecond UInt8,
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
ORDER BY (frame_timestamp, participantid, run_id);
