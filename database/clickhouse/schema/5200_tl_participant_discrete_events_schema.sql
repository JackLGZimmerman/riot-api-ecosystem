-- noqa: disable=LT05
CREATE TABLE IF NOT EXISTS game_data_filtered.tl_participant_discrete_events
(
    matchid String,
    frame_timestamp UInt32,
    teamid UInt8,
    participantid UInt8,

    kills UInt8,
    deaths UInt8,
    assists UInt8,
    champion_kill_bounty_gold UInt32,
    shutdown_bounty_gold UInt32,

    tower_takedowns UInt8,
    inhibitor_takedowns UInt8,
    building_kill_bounty_gold UInt32,

    elite_monster_takedowns_dragon UInt8,
    elite_monster_takedowns_rift_herald UInt8,
    elite_monster_takedowns_horde UInt8,
    elite_monster_takedowns_baron UInt8,

    wards_killed UInt8,
    wards_placed UInt8,
    turret_plates_top UInt8,
    turret_plates_mid UInt8,
    turret_plates_bot UInt8,

    legendary_item_delta Int16
)
ENGINE = SummingMergeTree(
    (
        kills,
        deaths,
        assists,
        champion_kill_bounty_gold,
        shutdown_bounty_gold,
        tower_takedowns,
        inhibitor_takedowns,
        building_kill_bounty_gold,
        elite_monster_takedowns_dragon,
        elite_monster_takedowns_rift_herald,
        elite_monster_takedowns_horde,
        elite_monster_takedowns_baron,
        wards_killed,
        wards_placed,
        turret_plates_top,
        turret_plates_mid,
        turret_plates_bot,
        legendary_item_delta
    )
)
ORDER BY (matchid, frame_timestamp, teamid, participantid);
