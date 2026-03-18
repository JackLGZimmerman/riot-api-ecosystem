CREATE TABLE IF NOT EXISTS game_data.tl_participant_stats
(
    run_id UUID,
    matchid UInt64,
    frame_timestamp UInt32,
    participantid UInt8,

    abilityhaste Int16,
    abilitypower Int16,
    armor Int16,
    attackdamage Int16,
    attackspeed Int16,
    ccreduction Int8,
    cooldownreduction Int16,
    health Int16,
    healthmax Int16,
    healthregen Int16,
    magicresist Int16,
    movementspeed Int16,
    power Int16,
    powermax Int16,
    powerregen Int16,
    payload Map (LowCardinality (String), Int32),

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
    position_x Int16,
    position_y Int16,
    timeenemyspentcontrolled UInt32,
    totalgold UInt16,
    xp UInt16
)
ENGINE = MergeTree
ORDER BY (matchid, frame_timestamp, participantid, run_id);
