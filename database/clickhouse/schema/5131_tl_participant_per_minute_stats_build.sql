TRUNCATE TABLE game_data_filtered.tl_participant_per_minute_stats;

INSERT INTO game_data_filtered.tl_participant_per_minute_stats
(
    matchid,
    frame_timestamp,
    participantid,
    nonchampiondamagedone,
    nonchampionphysicaldamagedone,
    nonchampionmagicdamagedone,
    nonchampiontruedamagedone,
    spentgold,
    totalfarm,
    spentgoldratio,
    championdamageshare,
    nonchampiondamageshare,
    physicalchampiondamageshare,
    physicalnonchampiondamageshare,
    magicchampiondamageshare,
    magicnonchampiondamageshare,
    truechampiondamageshare,
    truenonchampiondamageshare,
    championdamagetodamagetakenratio,
    championdamagepergoldearned,
    physicaldamageshare,
    magicdamageshare,
    truedamageshare,
    physicalchampiondamagetypeshare,
    magicchampiondamagetypeshare,
    truechampiondamagetypeshare,
    physicalnonchampiondamagetypeshare,
    magicnonchampiondamagetypeshare,
    truenonchampiondamagetypeshare,
    physicaldamagetakenshare,
    magicdamagetakenshare,
    truedamagetakenshare,
    lanefarmshare,
    junglefarmshare,
    championkilleventsperminutebin,
    championdeatheventsperminutebin,
    championassisteventsperminutebin,
    wardskilledperminutebin,
    wardsplacedperminutebin,
    itemsdestroyedperminutebin,
    itemspurchasedperminutebin,
    itemundosperminutebin,
    firstbloodeventsperminutebin,
    aceeventsperminutebin,
    multikilleventsperminutebin,
    doublekilltotalkillsperminutesum,
    triplekilltotalkillsperminutesum,
    quadrakilltotalkillsperminutesum,
    pentakilltotalkillsperminutesum,
    toplaneplatesdestroyedperminutesum,
    midlaneplatesdestroyedperminutesum,
    botlaneplatesdestroyedperminutesum,
    kdaperminutebin,
    kdperminutebin,
    kaperminutebin,
    killparticipationeventsperminutebin,
    kdaactivityperminutebin,
    nettakedownmarginperminutebin,
    visionactivityperminutebin,
    visiondenialshareperminutebin,
    itemactivityperminutebin,
    netitempurchaseactionsperminutebin,
    multikillkillshareperminutebin,
    totalplatesdestroyedperminutesum,
    wardplacementtokillratioperminutebin,
    toplaneplateshareperminutebin,
    midlaneplateshareperminutebin,
    botlaneplateshareperminutebin,
    platelaneconcentrationperminutebin,
    championdamageperdeathevent
)
WITH
CAST([], 'Array(Tuple(UInt8, UInt8, UInt8, UInt8))') AS empty_champion_kill_actor_rows,
base_p_stats AS (
    SELECT
        p.matchid,
        p.frame_timestamp,
        p.participantid,
        intDiv(p.frame_timestamp, 60000) AS minute_bin,
        p.currentgold,
        p.magicdamagedone,
        p.magicdamagedonetochampions,
        p.magicdamagetaken,
        p.physicaldamagedone,
        p.physicaldamagedonetochampions,
        p.physicaldamagetaken,
        p.totaldamagedone,
        p.totaldamagedonetochampions,
        p.totaldamagetaken,
        p.truedamagedone,
        p.truedamagedonetochampions,
        p.truedamagetaken,
        p.jungleminionskilled,
        p.minionskilled,
        p.totalgold
    FROM game_data_filtered.tl_participant_stats AS p
),

p_stats_base AS (
    SELECT
        p.matchid,
        p.frame_timestamp,
        p.participantid,
        p.minute_bin,
        p.minionskilled,
        p.jungleminionskilled,
        p.physicaldamagedone,
        p.magicdamagedone,
        p.truedamagedone,
        p.physicaldamagetaken,
        p.magicdamagetaken,
        p.truedamagetaken,
        p.totaldamagedonetochampions,
        p.physicaldamagedonetochampions,
        p.magicdamagedonetochampions,
        p.truedamagedonetochampions,
        p.totaldamagedone - p.totaldamagedonetochampions AS nonchampiondamagedone,
        p.physicaldamagedone
        - p.physicaldamagedonetochampions AS nonchampionphysicaldamagedone,
        p.magicdamagedone - p.magicdamagedonetochampions AS nonchampionmagicdamagedone,
        p.truedamagedone - p.truedamagedonetochampions AS nonchampiontruedamagedone,
        p.totalgold - p.currentgold AS spentgold,
        p.minionskilled + p.jungleminionskilled AS totalfarm,
        p.totaldamagedone AS totaldamagedone_denom,
        p.physicaldamagedone AS physicaldamagedone_denom,
        p.magicdamagedone AS magicdamagedone_denom,
        p.truedamagedone AS truedamagedone_denom,
        p.totaldamagedonetochampions AS championdamagedone_denom,
        p.totaldamagedone - p.totaldamagedonetochampions AS nonchampiondamagedone_denom,
        p.totaldamagetaken AS totaldamagetaken_denom,
        p.totalgold AS totalgold_denom,
        p.minionskilled + p.jungleminionskilled AS totalfarm_denom
    FROM base_p_stats AS p
),

p_stats_derived AS (
    SELECT
        p.matchid,
        p.frame_timestamp,
        p.participantid,
        p.minute_bin,
        p.totaldamagedonetochampions,
        p.nonchampiondamagedone,
        p.nonchampionphysicaldamagedone,
        p.nonchampionmagicdamagedone,
        p.nonchampiontruedamagedone,
        p.spentgold,
        p.totalfarm,
        p.spentgold / p.totalgold_denom AS spentgoldratio,
        p.totaldamagedonetochampions / p.totaldamagedone_denom AS championdamageshare,
        p.nonchampiondamagedone / p.totaldamagedone_denom AS nonchampiondamageshare,
        p.physicaldamagedonetochampions
        / p.physicaldamagedone_denom AS physicalchampiondamageshare,
        p.nonchampionphysicaldamagedone
        / p.physicaldamagedone_denom AS physicalnonchampiondamageshare,
        p.magicdamagedonetochampions
        / p.magicdamagedone_denom AS magicchampiondamageshare,
        p.nonchampionmagicdamagedone
        / p.magicdamagedone_denom AS magicnonchampiondamageshare,
        p.truedamagedonetochampions / p.truedamagedone_denom AS truechampiondamageshare,
        p.nonchampiontruedamagedone
        / p.truedamagedone_denom AS truenonchampiondamageshare,
        p.totaldamagedonetochampions
        / p.totaldamagetaken_denom AS championdamagetodamagetakenratio,
        p.totaldamagedonetochampions / p.totalgold_denom AS championdamagepergoldearned,
        p.physicaldamagedone / p.totaldamagedone_denom AS physicaldamageshare,
        p.magicdamagedone / p.totaldamagedone_denom AS magicdamageshare,
        p.truedamagedone / p.totaldamagedone_denom AS truedamageshare,
        p.physicaldamagedonetochampions
        / p.championdamagedone_denom AS physicalchampiondamagetypeshare,
        p.magicdamagedonetochampions
        / p.championdamagedone_denom AS magicchampiondamagetypeshare,
        p.truedamagedonetochampions
        / p.championdamagedone_denom AS truechampiondamagetypeshare,
        p.nonchampionphysicaldamagedone
        / p.nonchampiondamagedone_denom AS physicalnonchampiondamagetypeshare,
        p.nonchampionmagicdamagedone
        / p.nonchampiondamagedone_denom AS magicnonchampiondamagetypeshare,
        p.nonchampiontruedamagedone
        / p.nonchampiondamagedone_denom AS truenonchampiondamagetypeshare,
        p.physicaldamagetaken / p.totaldamagetaken_denom AS physicaldamagetakenshare,
        p.magicdamagetaken / p.totaldamagetaken_denom AS magicdamagetakenshare,
        p.truedamagetaken / p.totaldamagetaken_denom AS truedamagetakenshare,
        p.minionskilled / p.totalfarm_denom AS lanefarmshare,
        p.jungleminionskilled / p.totalfarm_denom AS junglefarmshare
    FROM p_stats_base AS p
),

champion_kill_actor_rows AS (
    SELECT
        matchid,
        intDiv(timestamp, 60000) AS minute_bin,
        tupleElement(actor_row, 1) AS participantid,
        tupleElement(actor_row, 2) AS kill_events,
        tupleElement(actor_row, 3) AS death_events,
        tupleElement(actor_row, 4) AS assist_events
    FROM game_data_filtered.tl_champion_kill
    ARRAY JOIN arrayConcat(
        if(
            killerid > 0,
            [(killerid, 1, 0, 0)],
            empty_champion_kill_actor_rows
        ),
        if(
            victimid > 0,
            [(victimid, 0, 1, 0)],
            empty_champion_kill_actor_rows
        ),
        arrayMap(
            participant_id -> (participant_id, 0, 0, 1),
            assistingparticipantids
        )
    ) AS actor_row
),

champion_kill_bins AS (
    SELECT
        ckar.matchid,
        ckar.minute_bin,
        ckar.participantid,
        sum(ckar.kill_events) AS championkilleventsperminutebin,
        sum(ckar.death_events) AS championdeatheventsperminutebin,
        sum(ckar.assist_events) AS championassisteventsperminutebin
    FROM champion_kill_actor_rows AS ckar
    GROUP BY
        ckar.matchid,
        ckar.minute_bin,
        ckar.participantid
),

payload_actor_rows AS (
    SELECT
        wk.matchid,
        intDiv(wk.timestamp, 60000) AS minute_bin,
        wk.killerid AS participantid,
        'WARD_KILL' AS type
    FROM game_data_filtered.tl_ward_kill AS wk
    WHERE wk.killerid > 0
    UNION ALL
    SELECT
        wp.matchid,
        intDiv(wp.timestamp, 60000) AS minute_bin,
        wp.creatorid AS participantid,
        'WARD_PLACED' AS type
    FROM game_data_filtered.tl_ward_placed AS wp
    WHERE wp.creatorid > 0
    UNION ALL
    SELECT
        id.matchid,
        intDiv(id.timestamp, 60000) AS minute_bin,
        id.participantid,
        'ITEM_DESTROYED' AS type
    FROM game_data_filtered.tl_item_destroyed AS id
    WHERE id.participantid > 0
    UNION ALL
    SELECT
        ip.matchid,
        intDiv(ip.timestamp, 60000) AS minute_bin,
        ip.participantid,
        'ITEM_PURCHASED' AS type
    FROM game_data_filtered.tl_item_purchased AS ip
    WHERE ip.participantid > 0
    UNION ALL
    SELECT
        iu.matchid,
        intDiv(iu.timestamp, 60000) AS minute_bin,
        iu.participantid,
        'ITEM_UNDO' AS type
    FROM game_data_filtered.tl_item_undo AS iu
    WHERE iu.participantid > 0
),

payload_bins AS (
    SELECT
        par.matchid,
        par.minute_bin,
        par.participantid,
        countIf(par.type = 'WARD_KILL') AS wardskilledperminutebin,
        countIf(par.type = 'WARD_PLACED') AS wardsplacedperminutebin,
        countIf(par.type = 'ITEM_DESTROYED') AS itemsdestroyedperminutebin,
        countIf(par.type = 'ITEM_PURCHASED') AS itemspurchasedperminutebin,
        countIf(par.type = 'ITEM_UNDO') AS itemundosperminutebin
    FROM payload_actor_rows AS par
    GROUP BY
        par.matchid,
        par.minute_bin,
        par.participantid
),

champion_special_kill_bins AS (
    SELECT
        csk.matchid,
        intDiv(csk.timestamp, 60000) AS minute_bin,
        csk.killerid AS participantid,
        countIf(csk.killtype = 'KILL_FIRST_BLOOD') AS firstbloodeventsperminutebin,
        countIf(csk.killtype = 'KILL_ACE') AS aceeventsperminutebin,
        countIf(csk.killtype = 'KILL_MULTI') AS multikilleventsperminutebin,
        sumIf(
            csk.multikilllength, csk.killtype = 'KILL_MULTI' AND csk.multikilllength = 2
        ) AS doublekilltotalkillsperminutesum,
        sumIf(
            csk.multikilllength, csk.killtype = 'KILL_MULTI' AND csk.multikilllength = 3
        ) AS triplekilltotalkillsperminutesum,
        sumIf(
            csk.multikilllength, csk.killtype = 'KILL_MULTI' AND csk.multikilllength = 4
        ) AS quadrakilltotalkillsperminutesum,
        sumIf(
            csk.multikilllength, csk.killtype = 'KILL_MULTI' AND csk.multikilllength = 5
        ) AS pentakilltotalkillsperminutesum
    FROM game_data_filtered.tl_champion_special_kill AS csk
    WHERE csk.killerid > 0
    GROUP BY
        csk.matchid,
        minute_bin,
        participantid
),

turret_plate_bins AS (
    SELECT
        tpd.matchid,
        intDiv(tpd.timestamp, 60000) AS minute_bin,
        tpd.killerid AS participantid,
        countIf(tpd.lanetype = 'TOP_LANE') AS toplaneplatesdestroyedperminutesum,
        countIf(tpd.lanetype = 'MID_LANE') AS midlaneplatesdestroyedperminutesum,
        countIf(tpd.lanetype = 'BOT_LANE') AS botlaneplatesdestroyedperminutesum
    FROM game_data_filtered.tl_turret_plate_destroyed AS tpd
    WHERE tpd.killerid > 0
    GROUP BY
        tpd.matchid,
        minute_bin,
        participantid
),

frame_metrics AS (
    SELECT
        -- Keep plain names after ClickHouse expands joined CTE columns.
        -- noqa: disable=AL09
        p.matchid AS matchid,
        p.frame_timestamp AS frame_timestamp,
        p.participantid AS participantid,
        p.minute_bin AS minute_bin,
        p.totaldamagedonetochampions AS totaldamagedonetochampions,
        p.nonchampiondamagedone AS nonchampiondamagedone,
        p.nonchampionphysicaldamagedone AS nonchampionphysicaldamagedone,
        p.nonchampionmagicdamagedone AS nonchampionmagicdamagedone,
        p.nonchampiontruedamagedone AS nonchampiontruedamagedone,
        p.spentgold AS spentgold,
        p.totalfarm AS totalfarm,
        p.spentgoldratio AS spentgoldratio,
        p.championdamageshare AS championdamageshare,
        p.nonchampiondamageshare AS nonchampiondamageshare,
        p.physicalchampiondamageshare AS physicalchampiondamageshare,
        p.physicalnonchampiondamageshare AS physicalnonchampiondamageshare,
        p.magicchampiondamageshare AS magicchampiondamageshare,
        p.magicnonchampiondamageshare AS magicnonchampiondamageshare,
        p.truechampiondamageshare AS truechampiondamageshare,
        p.truenonchampiondamageshare AS truenonchampiondamageshare,
        p.championdamagetodamagetakenratio AS championdamagetodamagetakenratio,
        p.championdamagepergoldearned AS championdamagepergoldearned,
        p.physicaldamageshare AS physicaldamageshare,
        p.magicdamageshare AS magicdamageshare,
        p.truedamageshare AS truedamageshare,
        p.physicalchampiondamagetypeshare AS physicalchampiondamagetypeshare,
        p.magicchampiondamagetypeshare AS magicchampiondamagetypeshare,
        p.truechampiondamagetypeshare AS truechampiondamagetypeshare,
        p.physicalnonchampiondamagetypeshare AS physicalnonchampiondamagetypeshare,
        p.magicnonchampiondamagetypeshare AS magicnonchampiondamagetypeshare,
        p.truenonchampiondamagetypeshare AS truenonchampiondamagetypeshare,
        p.physicaldamagetakenshare AS physicaldamagetakenshare,
        p.magicdamagetakenshare AS magicdamagetakenshare,
        p.truedamagetakenshare AS truedamagetakenshare,
        p.lanefarmshare AS lanefarmshare,
        p.junglefarmshare AS junglefarmshare,
        -- noqa: enable=AL09
        coalesce(
            ck.championkilleventsperminutebin,
            0
        ) AS championkilleventsperminutebin,
        coalesce(
            ck.championdeatheventsperminutebin,
            0
        ) AS championdeatheventsperminutebin,
        coalesce(
            ck.championassisteventsperminutebin,
            0
        ) AS championassisteventsperminutebin,
        coalesce(pb.wardskilledperminutebin, 0) AS wardskilledperminutebin,
        coalesce(pb.wardsplacedperminutebin, 0) AS wardsplacedperminutebin,
        coalesce(pb.itemsdestroyedperminutebin, 0) AS itemsdestroyedperminutebin,
        coalesce(pb.itemspurchasedperminutebin, 0) AS itemspurchasedperminutebin,
        coalesce(pb.itemundosperminutebin, 0) AS itemundosperminutebin,
        coalesce(csk.firstbloodeventsperminutebin, 0) AS firstbloodeventsperminutebin,
        coalesce(csk.aceeventsperminutebin, 0) AS aceeventsperminutebin,
        coalesce(csk.multikilleventsperminutebin, 0) AS multikilleventsperminutebin,
        coalesce(
            csk.doublekilltotalkillsperminutesum,
            0
        ) AS doublekilltotalkillsperminutesum,
        coalesce(
            csk.triplekilltotalkillsperminutesum,
            0
        ) AS triplekilltotalkillsperminutesum,
        coalesce(
            csk.quadrakilltotalkillsperminutesum,
            0
        ) AS quadrakilltotalkillsperminutesum,
        coalesce(
            csk.pentakilltotalkillsperminutesum,
            0
        ) AS pentakilltotalkillsperminutesum,
        coalesce(
            tpb.toplaneplatesdestroyedperminutesum,
            0
        ) AS toplaneplatesdestroyedperminutesum,
        coalesce(
            tpb.midlaneplatesdestroyedperminutesum,
            0
        ) AS midlaneplatesdestroyedperminutesum,
        coalesce(
            tpb.botlaneplatesdestroyedperminutesum,
            0
        ) AS botlaneplatesdestroyedperminutesum,
        coalesce(ck.championkilleventsperminutebin, 0)
        + coalesce(
            ck.championassisteventsperminutebin,
            0
        ) AS killparticipationeventsperminutebin,
        coalesce(ck.championkilleventsperminutebin, 0)
        + coalesce(ck.championdeatheventsperminutebin, 0)
        + coalesce(ck.championassisteventsperminutebin, 0)
            AS kdaactivityperminutebin,
        coalesce(ck.championkilleventsperminutebin, 0)
        + coalesce(ck.championassisteventsperminutebin, 0)
        - coalesce(ck.championdeatheventsperminutebin, 0)
            AS nettakedownmarginperminutebin,
        coalesce(pb.wardskilledperminutebin, 0)
        + coalesce(pb.wardsplacedperminutebin, 0) AS visionactivityperminutebin,
        coalesce(pb.itemsdestroyedperminutebin, 0)
        + coalesce(pb.itemspurchasedperminutebin, 0)
        + coalesce(pb.itemundosperminutebin, 0) AS itemactivityperminutebin,
        coalesce(pb.itemspurchasedperminutebin, 0)
        - coalesce(pb.itemundosperminutebin, 0)
            AS netitempurchaseactionsperminutebin,
        coalesce(tpb.toplaneplatesdestroyedperminutesum, 0)
        + coalesce(tpb.midlaneplatesdestroyedperminutesum, 0)
        + coalesce(tpb.botlaneplatesdestroyedperminutesum, 0)
            AS totalplatesdestroyedperminutesum
    FROM p_stats_derived AS p
    LEFT JOIN champion_kill_bins AS ck
        ON
            p.matchid = ck.matchid
            AND p.participantid = ck.participantid
            AND p.minute_bin = ck.minute_bin
    LEFT JOIN payload_bins AS pb
        ON
            p.matchid = pb.matchid
            AND p.participantid = pb.participantid
            AND p.minute_bin = pb.minute_bin
    LEFT JOIN champion_special_kill_bins AS csk
        ON
            p.matchid = csk.matchid
            AND p.participantid = csk.participantid
            AND p.minute_bin = csk.minute_bin
    LEFT JOIN turret_plate_bins AS tpb
        ON
            p.matchid = tpb.matchid
            AND p.participantid = tpb.participantid
            AND p.minute_bin = tpb.minute_bin
),

final_input AS (
    SELECT
        *,
        championdeatheventsperminutebin AS championdeath_denom,
        championassisteventsperminutebin AS championassist_denom,
        championkilleventsperminutebin AS championkill_denom,
        visionactivityperminutebin AS visionactivity_denom,
        totalplatesdestroyedperminutesum AS totalplates_denom,
        (
            doublekilltotalkillsperminutesum
            + triplekilltotalkillsperminutesum
            + quadrakilltotalkillsperminutesum
            + pentakilltotalkillsperminutesum
        ) AS multikilltotalkillsperminutesum
    FROM frame_metrics
)

SELECT
    fi.matchid,
    fi.frame_timestamp,
    fi.participantid,
    fi.nonchampiondamagedone,
    fi.nonchampionphysicaldamagedone,
    fi.nonchampionmagicdamagedone,
    fi.nonchampiontruedamagedone,
    fi.spentgold,
    fi.totalfarm,
    fi.spentgoldratio,
    fi.championdamageshare,
    fi.nonchampiondamageshare,
    fi.physicalchampiondamageshare,
    fi.physicalnonchampiondamageshare,
    fi.magicchampiondamageshare,
    fi.magicnonchampiondamageshare,
    fi.truechampiondamageshare,
    fi.truenonchampiondamageshare,
    fi.championdamagetodamagetakenratio,
    fi.championdamagepergoldearned,
    fi.physicaldamageshare,
    fi.magicdamageshare,
    fi.truedamageshare,
    fi.physicalchampiondamagetypeshare,
    fi.magicchampiondamagetypeshare,
    fi.truechampiondamagetypeshare,
    fi.physicalnonchampiondamagetypeshare,
    fi.magicnonchampiondamagetypeshare,
    fi.truenonchampiondamagetypeshare,
    fi.physicaldamagetakenshare,
    fi.magicdamagetakenshare,
    fi.truedamagetakenshare,
    fi.lanefarmshare,
    fi.junglefarmshare,
    fi.championkilleventsperminutebin,
    fi.championdeatheventsperminutebin,
    fi.championassisteventsperminutebin,
    fi.wardskilledperminutebin,
    fi.wardsplacedperminutebin,
    fi.itemsdestroyedperminutebin,
    fi.itemspurchasedperminutebin,
    fi.itemundosperminutebin,
    fi.firstbloodeventsperminutebin,
    fi.aceeventsperminutebin,
    fi.multikilleventsperminutebin,
    fi.doublekilltotalkillsperminutesum,
    fi.triplekilltotalkillsperminutesum,
    fi.quadrakilltotalkillsperminutesum,
    fi.pentakilltotalkillsperminutesum,
    fi.toplaneplatesdestroyedperminutesum,
    fi.midlaneplatesdestroyedperminutesum,
    fi.botlaneplatesdestroyedperminutesum,
    (fi.championkilleventsperminutebin + fi.championassisteventsperminutebin)
    / fi.championdeath_denom AS kdaperminutebin,
    fi.championkilleventsperminutebin / fi.championdeath_denom AS kdperminutebin,
    fi.championkilleventsperminutebin / fi.championassist_denom AS kaperminutebin,
    fi.killparticipationeventsperminutebin,
    fi.kdaactivityperminutebin,
    fi.nettakedownmarginperminutebin,
    fi.visionactivityperminutebin,
    fi.wardskilledperminutebin
    / fi.visionactivity_denom AS visiondenialshareperminutebin,
    fi.itemactivityperminutebin,
    fi.netitempurchaseactionsperminutebin,
    fi.multikilltotalkillsperminutesum
    / fi.championkill_denom AS multikillkillshareperminutebin,
    fi.totalplatesdestroyedperminutesum,
    fi.wardsplacedperminutebin
    / fi.wardskilledperminutebin AS wardplacementtokillratioperminutebin,
    fi.toplaneplatesdestroyedperminutesum
    / fi.totalplates_denom AS toplaneplateshareperminutebin,
    fi.midlaneplatesdestroyedperminutesum
    / fi.totalplates_denom AS midlaneplateshareperminutebin,
    fi.botlaneplatesdestroyedperminutesum
    / fi.totalplates_denom AS botlaneplateshareperminutebin,
    (
        pow(fi.toplaneplatesdestroyedperminutesum / fi.totalplates_denom, 2)
        + pow(fi.midlaneplatesdestroyedperminutesum / fi.totalplates_denom, 2)
        + pow(fi.botlaneplatesdestroyedperminutesum / fi.totalplates_denom, 2)
    ) AS platelaneconcentrationperminutebin,
    fi.totaldamagedonetochampions
    / fi.championdeath_denom AS championdamageperdeathevent
FROM final_input AS fi;
