-- noqa: disable=LT05
TRUNCATE TABLE game_data_filtered.tl_participant_discrete_events;

DROP TABLE IF EXISTS game_data_filtered.tl_participant_discrete_events_participant_dim_tmp;

CREATE TABLE game_data_filtered.tl_participant_discrete_events_participant_dim_tmp
(
    matchid String,
    participantid UInt8,
    teamid UInt8
)
ENGINE = Memory;

INSERT INTO game_data_filtered.tl_participant_discrete_events_participant_dim_tmp
SELECT
    matchid,
    participantid,
    any(teamid) AS teamid
FROM game_data_filtered.participant_stats
GROUP BY
    matchid,
    participantid;

-- Keep each event family isolated so this build does not need one wide
-- union-all aggregation. Consumers should still group by the table key unless
-- they explicitly read FINAL, because SummingMergeTree merges asynchronously.

INSERT INTO game_data_filtered.tl_participant_discrete_events
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
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
WITH
CAST([], 'Array(Tuple(UInt8, UInt8, UInt8, UInt8, UInt32, UInt32))') AS empty_actor_rows,

champion_kill_bins AS (
    SELECT
        ckar.matchid,
        ckar.frame_timestamp,
        tupleElement(ckar.actor_row, 1) AS participantid,
        toUInt8(sum(tupleElement(ckar.actor_row, 2))) AS kills,
        toUInt8(sum(tupleElement(ckar.actor_row, 3))) AS deaths,
        toUInt8(sum(tupleElement(ckar.actor_row, 4))) AS assists,
        toUInt32(sum(tupleElement(ckar.actor_row, 5)))
            AS champion_kill_bounty_gold,
        toUInt32(sum(tupleElement(ckar.actor_row, 6))) AS shutdown_bounty_gold
    FROM (
        SELECT
            matchid,
            frame_timestamp,
            actor_row
        FROM game_data_filtered.tl_champion_kill
        ARRAY JOIN
            arrayConcat(
                if(
                    killerid > 0,
                    [
                        (
                            toUInt8(killerid),
                            toUInt8(1),
                            toUInt8(0),
                            toUInt8(0),
                            toUInt32(bounty),
                            toUInt32(shutdownbounty)
                        )
                    ],
                    empty_actor_rows
                ),
                if(
                    victimid > 0,
                    [
                        (
                            toUInt8(victimid),
                            toUInt8(0),
                            toUInt8(1),
                            toUInt8(0),
                            toUInt32(0),
                            toUInt32(0)
                        )
                    ],
                    empty_actor_rows
                ),
                arrayMap(
                    pid -> (
                        pid,
                        toUInt8(0),
                        toUInt8(0),
                        toUInt8(1),
                        toUInt32(0),
                        toUInt32(0)
                    ),
                    assistingparticipantids
                )
            ) AS actor_row
    ) AS ckar
    GROUP BY
        ckar.matchid,
        ckar.frame_timestamp,
        tupleElement(ckar.actor_row, 1)
)

SELECT
    ck.matchid,
    ck.frame_timestamp,
    pd.teamid,
    ck.participantid,
    ck.kills,
    ck.deaths,
    ck.assists,
    ck.champion_kill_bounty_gold,
    ck.shutdown_bounty_gold,
    toUInt8(0) AS tower_takedowns,
    toUInt8(0) AS inhibitor_takedowns,
    toUInt32(0) AS building_kill_bounty_gold,
    toUInt8(0) AS elite_monster_takedowns_dragon,
    toUInt8(0) AS elite_monster_takedowns_rift_herald,
    toUInt8(0) AS elite_monster_takedowns_horde,
    toUInt8(0) AS elite_monster_takedowns_baron,
    toUInt8(0) AS wards_killed,
    toUInt8(0) AS wards_placed,
    toUInt8(0) AS turret_plates_top,
    toUInt8(0) AS turret_plates_mid,
    toUInt8(0) AS turret_plates_bot,
    toInt16(0) AS legendary_item_delta
FROM champion_kill_bins AS ck
ANY LEFT JOIN
    game_data_filtered.tl_participant_discrete_events_participant_dim_tmp AS pd
    ON
        ck.matchid = pd.matchid
        AND ck.participantid = pd.participantid;

INSERT INTO game_data_filtered.tl_participant_discrete_events
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
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
WITH
CAST([], 'Array(Tuple(UInt8, UInt8, UInt8, UInt32))') AS empty_actor_rows,

building_bins AS (
    SELECT
        bkar.matchid,
        bkar.frame_timestamp,
        tupleElement(bkar.actor_row, 1) AS participantid,
        toUInt8(sum(tupleElement(bkar.actor_row, 2))) AS tower_takedowns,
        toUInt8(sum(tupleElement(bkar.actor_row, 3)))
            AS inhibitor_takedowns,
        toUInt32(sum(tupleElement(bkar.actor_row, 4)))
            AS building_kill_bounty_gold
    FROM (
        SELECT
            matchid,
            frame_timestamp,
            actor_row
        FROM game_data_filtered.tl_building_kill
        ARRAY JOIN
            arrayConcat(
                if(
                    killerid > 0,
                    [
                        (
                            toUInt8(killerid),
                            if(buildingtype = 'TOWER_BUILDING', toUInt8(1), toUInt8(0)),
                            if(buildingtype = 'INHIBITOR_BUILDING', toUInt8(1), toUInt8(0)),
                            toUInt32(bounty)
                        )
                    ],
                    empty_actor_rows
                ),
                if(
                    buildingtype IN ('TOWER_BUILDING', 'INHIBITOR_BUILDING'),
                    arrayMap(
                        pid -> (
                            pid,
                            if(buildingtype = 'TOWER_BUILDING', toUInt8(1), toUInt8(0)),
                            if(buildingtype = 'INHIBITOR_BUILDING', toUInt8(1), toUInt8(0)),
                            toUInt32(0)
                        ),
                        assistingparticipantids
                    ),
                    empty_actor_rows
                )
            ) AS actor_row
    ) AS bkar
    GROUP BY
        bkar.matchid,
        bkar.frame_timestamp,
        tupleElement(bkar.actor_row, 1)
)

SELECT
    bk.matchid,
    bk.frame_timestamp,
    pd.teamid,
    bk.participantid,
    toUInt8(0) AS kills,
    toUInt8(0) AS deaths,
    toUInt8(0) AS assists,
    toUInt32(0) AS champion_kill_bounty_gold,
    toUInt32(0) AS shutdown_bounty_gold,
    bk.tower_takedowns,
    bk.inhibitor_takedowns,
    bk.building_kill_bounty_gold,
    toUInt8(0) AS elite_monster_takedowns_dragon,
    toUInt8(0) AS elite_monster_takedowns_rift_herald,
    toUInt8(0) AS elite_monster_takedowns_horde,
    toUInt8(0) AS elite_monster_takedowns_baron,
    toUInt8(0) AS wards_killed,
    toUInt8(0) AS wards_placed,
    toUInt8(0) AS turret_plates_top,
    toUInt8(0) AS turret_plates_mid,
    toUInt8(0) AS turret_plates_bot,
    toInt16(0) AS legendary_item_delta
FROM building_bins AS bk
ANY LEFT JOIN
    game_data_filtered.tl_participant_discrete_events_participant_dim_tmp AS pd
    ON
        bk.matchid = pd.matchid
        AND bk.participantid = pd.participantid;

INSERT INTO game_data_filtered.tl_participant_discrete_events
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
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
WITH
CAST([], 'Array(UInt8)') AS empty_participant_ids,

elite_monster_bins AS (
    SELECT
        emar.matchid,
        emar.frame_timestamp,
        emar.participantid,
        toUInt8(countIf(emar.monstertype = 'DRAGON'))
            AS elite_monster_takedowns_dragon,
        toUInt8(countIf(emar.monstertype = 'RIFTHERALD'))
            AS elite_monster_takedowns_rift_herald,
        toUInt8(countIf(emar.monstertype = 'HORDE'))
            AS elite_monster_takedowns_horde,
        toUInt8(countIf(emar.monstertype = 'BARON_NASHOR'))
            AS elite_monster_takedowns_baron
    FROM (
        SELECT
            matchid,
            frame_timestamp,
            monstertype,
            participantid
        FROM game_data_filtered.tl_elite_monster_kill
        ARRAY JOIN
            arrayConcat(
                if(killerid > 0, [toUInt8(killerid)], empty_participant_ids),
                assistingparticipantids
            ) AS participantid
    ) AS emar
    GROUP BY
        emar.matchid,
        emar.frame_timestamp,
        emar.participantid
)

SELECT
    em.matchid,
    em.frame_timestamp,
    pd.teamid,
    em.participantid,
    toUInt8(0) AS kills,
    toUInt8(0) AS deaths,
    toUInt8(0) AS assists,
    toUInt32(0) AS champion_kill_bounty_gold,
    toUInt32(0) AS shutdown_bounty_gold,
    toUInt8(0) AS tower_takedowns,
    toUInt8(0) AS inhibitor_takedowns,
    toUInt32(0) AS building_kill_bounty_gold,
    em.elite_monster_takedowns_dragon,
    em.elite_monster_takedowns_rift_herald,
    em.elite_monster_takedowns_horde,
    em.elite_monster_takedowns_baron,
    toUInt8(0) AS wards_killed,
    toUInt8(0) AS wards_placed,
    toUInt8(0) AS turret_plates_top,
    toUInt8(0) AS turret_plates_mid,
    toUInt8(0) AS turret_plates_bot,
    toInt16(0) AS legendary_item_delta
FROM elite_monster_bins AS em
ANY LEFT JOIN
    game_data_filtered.tl_participant_discrete_events_participant_dim_tmp AS pd
    ON
        em.matchid = pd.matchid
        AND em.participantid = pd.participantid;

INSERT INTO game_data_filtered.tl_participant_discrete_events
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
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
WITH
ward_kill_bins AS (
    SELECT
        matchid,
        frame_timestamp,
        toUInt8(killerid) AS participantid,
        toUInt8(count()) AS wards_killed
    FROM game_data_filtered.tl_ward_kill
    WHERE killerid > 0
    GROUP BY
        matchid,
        frame_timestamp,
        participantid
)

SELECT
    wk.matchid,
    wk.frame_timestamp,
    pd.teamid,
    wk.participantid,
    toUInt8(0) AS kills,
    toUInt8(0) AS deaths,
    toUInt8(0) AS assists,
    toUInt32(0) AS champion_kill_bounty_gold,
    toUInt32(0) AS shutdown_bounty_gold,
    toUInt8(0) AS tower_takedowns,
    toUInt8(0) AS inhibitor_takedowns,
    toUInt32(0) AS building_kill_bounty_gold,
    toUInt8(0) AS elite_monster_takedowns_dragon,
    toUInt8(0) AS elite_monster_takedowns_rift_herald,
    toUInt8(0) AS elite_monster_takedowns_horde,
    toUInt8(0) AS elite_monster_takedowns_baron,
    wk.wards_killed,
    toUInt8(0) AS wards_placed,
    toUInt8(0) AS turret_plates_top,
    toUInt8(0) AS turret_plates_mid,
    toUInt8(0) AS turret_plates_bot,
    toInt16(0) AS legendary_item_delta
FROM ward_kill_bins AS wk
ANY LEFT JOIN
    game_data_filtered.tl_participant_discrete_events_participant_dim_tmp AS pd
    ON
        wk.matchid = pd.matchid
        AND wk.participantid = pd.participantid;

INSERT INTO game_data_filtered.tl_participant_discrete_events
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
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
WITH
ward_placed_bins AS (
    SELECT
        matchid,
        frame_timestamp,
        creatorid AS participantid,
        toUInt8(count()) AS wards_placed
    FROM game_data_filtered.tl_ward_placed
    WHERE creatorid > 0
    GROUP BY
        matchid,
        frame_timestamp,
        participantid
)

SELECT
    wp.matchid,
    wp.frame_timestamp,
    pd.teamid,
    wp.participantid,
    toUInt8(0) AS kills,
    toUInt8(0) AS deaths,
    toUInt8(0) AS assists,
    toUInt32(0) AS champion_kill_bounty_gold,
    toUInt32(0) AS shutdown_bounty_gold,
    toUInt8(0) AS tower_takedowns,
    toUInt8(0) AS inhibitor_takedowns,
    toUInt32(0) AS building_kill_bounty_gold,
    toUInt8(0) AS elite_monster_takedowns_dragon,
    toUInt8(0) AS elite_monster_takedowns_rift_herald,
    toUInt8(0) AS elite_monster_takedowns_horde,
    toUInt8(0) AS elite_monster_takedowns_baron,
    toUInt8(0) AS wards_killed,
    wp.wards_placed,
    toUInt8(0) AS turret_plates_top,
    toUInt8(0) AS turret_plates_mid,
    toUInt8(0) AS turret_plates_bot,
    toInt16(0) AS legendary_item_delta
FROM ward_placed_bins AS wp
ANY LEFT JOIN
    game_data_filtered.tl_participant_discrete_events_participant_dim_tmp AS pd
    ON
        wp.matchid = pd.matchid
        AND wp.participantid = pd.participantid;

INSERT INTO game_data_filtered.tl_participant_discrete_events
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
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
WITH
turret_plate_bins AS (
    SELECT
        matchid,
        frame_timestamp,
        toUInt8(killerid) AS participantid,
        toUInt8(countIf(lanetype = 'TOP_LANE')) AS turret_plates_top,
        toUInt8(countIf(lanetype = 'MID_LANE')) AS turret_plates_mid,
        toUInt8(countIf(lanetype = 'BOT_LANE')) AS turret_plates_bot
    FROM game_data_filtered.tl_turret_plate_destroyed
    WHERE killerid > 0
    GROUP BY
        matchid,
        frame_timestamp,
        participantid
)

SELECT
    tp.matchid,
    tp.frame_timestamp,
    pd.teamid,
    tp.participantid,
    toUInt8(0) AS kills,
    toUInt8(0) AS deaths,
    toUInt8(0) AS assists,
    toUInt32(0) AS champion_kill_bounty_gold,
    toUInt32(0) AS shutdown_bounty_gold,
    toUInt8(0) AS tower_takedowns,
    toUInt8(0) AS inhibitor_takedowns,
    toUInt32(0) AS building_kill_bounty_gold,
    toUInt8(0) AS elite_monster_takedowns_dragon,
    toUInt8(0) AS elite_monster_takedowns_rift_herald,
    toUInt8(0) AS elite_monster_takedowns_horde,
    toUInt8(0) AS elite_monster_takedowns_baron,
    toUInt8(0) AS wards_killed,
    toUInt8(0) AS wards_placed,
    tp.turret_plates_top,
    tp.turret_plates_mid,
    tp.turret_plates_bot,
    toInt16(0) AS legendary_item_delta
FROM turret_plate_bins AS tp
ANY LEFT JOIN
    game_data_filtered.tl_participant_discrete_events_participant_dim_tmp AS pd
    ON
        tp.matchid = pd.matchid
        AND tp.participantid = pd.participantid;

INSERT INTO game_data_filtered.tl_participant_discrete_events
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
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
WITH
item_purchase_bins AS (
    SELECT
        matchid,
        frame_timestamp,
        participantid,
        toInt16(sum(if(
            dictHas('game_data.item_value_map_dict', (toInt32(0), '', itemid)),
            toInt8(1),
            toInt8(0)
        ))) AS legendary_item_delta
    FROM game_data_filtered.tl_item_purchased
    WHERE participantid > 0
    GROUP BY
        matchid,
        frame_timestamp,
        participantid
)

SELECT
    ip.matchid,
    ip.frame_timestamp,
    pd.teamid,
    ip.participantid,
    toUInt8(0) AS kills,
    toUInt8(0) AS deaths,
    toUInt8(0) AS assists,
    toUInt32(0) AS champion_kill_bounty_gold,
    toUInt32(0) AS shutdown_bounty_gold,
    toUInt8(0) AS tower_takedowns,
    toUInt8(0) AS inhibitor_takedowns,
    toUInt32(0) AS building_kill_bounty_gold,
    toUInt8(0) AS elite_monster_takedowns_dragon,
    toUInt8(0) AS elite_monster_takedowns_rift_herald,
    toUInt8(0) AS elite_monster_takedowns_horde,
    toUInt8(0) AS elite_monster_takedowns_baron,
    toUInt8(0) AS wards_killed,
    toUInt8(0) AS wards_placed,
    toUInt8(0) AS turret_plates_top,
    toUInt8(0) AS turret_plates_mid,
    toUInt8(0) AS turret_plates_bot,
    ip.legendary_item_delta
FROM item_purchase_bins AS ip
ANY LEFT JOIN
    game_data_filtered.tl_participant_discrete_events_participant_dim_tmp AS pd
    ON
        ip.matchid = pd.matchid
        AND ip.participantid = pd.participantid;

INSERT INTO game_data_filtered.tl_participant_discrete_events
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
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
WITH
item_sold_bins AS (
    SELECT
        matchid,
        frame_timestamp,
        participantid,
        toInt16(-sum(if(
            dictHas('game_data.item_value_map_dict', (toInt32(0), '', itemid)),
            toInt8(1),
            toInt8(0)
        ))) AS legendary_item_delta
    FROM game_data_filtered.tl_item_sold
    WHERE participantid > 0
    GROUP BY
        matchid,
        frame_timestamp,
        participantid
)

SELECT
    isb.matchid,
    isb.frame_timestamp,
    pd.teamid,
    isb.participantid,
    toUInt8(0) AS kills,
    toUInt8(0) AS deaths,
    toUInt8(0) AS assists,
    toUInt32(0) AS champion_kill_bounty_gold,
    toUInt32(0) AS shutdown_bounty_gold,
    toUInt8(0) AS tower_takedowns,
    toUInt8(0) AS inhibitor_takedowns,
    toUInt32(0) AS building_kill_bounty_gold,
    toUInt8(0) AS elite_monster_takedowns_dragon,
    toUInt8(0) AS elite_monster_takedowns_rift_herald,
    toUInt8(0) AS elite_monster_takedowns_horde,
    toUInt8(0) AS elite_monster_takedowns_baron,
    toUInt8(0) AS wards_killed,
    toUInt8(0) AS wards_placed,
    toUInt8(0) AS turret_plates_top,
    toUInt8(0) AS turret_plates_mid,
    toUInt8(0) AS turret_plates_bot,
    isb.legendary_item_delta
FROM item_sold_bins AS isb
ANY LEFT JOIN
    game_data_filtered.tl_participant_discrete_events_participant_dim_tmp AS pd
    ON
        isb.matchid = pd.matchid
        AND isb.participantid = pd.participantid;

INSERT INTO game_data_filtered.tl_participant_discrete_events
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
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
WITH
item_destroyed_bins AS (
    SELECT
        matchid,
        frame_timestamp,
        participantid,
        toInt16(-sum(if(
            dictHas('game_data.item_value_map_dict', (toInt32(0), '', itemid)),
            toInt8(1),
            toInt8(0)
        ))) AS legendary_item_delta
    FROM game_data_filtered.tl_item_destroyed
    WHERE participantid > 0
    GROUP BY
        matchid,
        frame_timestamp,
        participantid
)

SELECT
    idb.matchid,
    idb.frame_timestamp,
    pd.teamid,
    idb.participantid,
    toUInt8(0) AS kills,
    toUInt8(0) AS deaths,
    toUInt8(0) AS assists,
    toUInt32(0) AS champion_kill_bounty_gold,
    toUInt32(0) AS shutdown_bounty_gold,
    toUInt8(0) AS tower_takedowns,
    toUInt8(0) AS inhibitor_takedowns,
    toUInt32(0) AS building_kill_bounty_gold,
    toUInt8(0) AS elite_monster_takedowns_dragon,
    toUInt8(0) AS elite_monster_takedowns_rift_herald,
    toUInt8(0) AS elite_monster_takedowns_horde,
    toUInt8(0) AS elite_monster_takedowns_baron,
    toUInt8(0) AS wards_killed,
    toUInt8(0) AS wards_placed,
    toUInt8(0) AS turret_plates_top,
    toUInt8(0) AS turret_plates_mid,
    toUInt8(0) AS turret_plates_bot,
    idb.legendary_item_delta
FROM item_destroyed_bins AS idb
ANY LEFT JOIN
    game_data_filtered.tl_participant_discrete_events_participant_dim_tmp AS pd
    ON
        idb.matchid = pd.matchid
        AND idb.participantid = pd.participantid;

INSERT INTO game_data_filtered.tl_participant_discrete_events
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
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
WITH
item_undo_bins AS (
    SELECT
        matchid,
        frame_timestamp,
        participantid,
        toInt16(sum(
            if(
                dictHas('game_data.item_value_map_dict', (toInt32(0), '', afterid)),
                toInt8(1),
                toInt8(0)
            )
            - if(
                dictHas('game_data.item_value_map_dict', (toInt32(0), '', beforeid)),
                toInt8(1),
                toInt8(0)
            )
        )) AS legendary_item_delta
    FROM game_data_filtered.tl_item_undo
    WHERE participantid > 0
    GROUP BY
        matchid,
        frame_timestamp,
        participantid
)

SELECT
    iu.matchid,
    iu.frame_timestamp,
    pd.teamid,
    iu.participantid,
    toUInt8(0) AS kills,
    toUInt8(0) AS deaths,
    toUInt8(0) AS assists,
    toUInt32(0) AS champion_kill_bounty_gold,
    toUInt32(0) AS shutdown_bounty_gold,
    toUInt8(0) AS tower_takedowns,
    toUInt8(0) AS inhibitor_takedowns,
    toUInt32(0) AS building_kill_bounty_gold,
    toUInt8(0) AS elite_monster_takedowns_dragon,
    toUInt8(0) AS elite_monster_takedowns_rift_herald,
    toUInt8(0) AS elite_monster_takedowns_horde,
    toUInt8(0) AS elite_monster_takedowns_baron,
    toUInt8(0) AS wards_killed,
    toUInt8(0) AS wards_placed,
    toUInt8(0) AS turret_plates_top,
    toUInt8(0) AS turret_plates_mid,
    toUInt8(0) AS turret_plates_bot,
    iu.legendary_item_delta
FROM item_undo_bins AS iu
ANY LEFT JOIN
    game_data_filtered.tl_participant_discrete_events_participant_dim_tmp AS pd
    ON
        iu.matchid = pd.matchid
        AND iu.participantid = pd.participantid;

DROP TABLE IF EXISTS game_data_filtered.tl_participant_discrete_events_participant_dim_tmp;
