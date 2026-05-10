-- noqa: disable=PRS,AL02,CP02,ST11
-- Build game_data.participant_stats_corrected by removing the contribution of
-- "stat-padding" events from kill- and gold-derived columns.
--
-- Padding window: an event is treated as padding when its timeline timestamp
-- is within 15000 ms of the match's tl_game_end timestamp.
--
-- CHAMPION_KILL adjustments (per-participant deltas, clamped at 0):
--   d_kills    = #padding events where killerid = P (P > 0)
--   d_deaths   = #padding events where victimid = P
--   d_assists  = #padding events where P in assistingparticipantids
--   d_*damagedealttochampions(P) =
--       sum(received rows where source        = P)        -- P -> victim
--     + sum(dealt    rows where event.victimid = P)        -- P (as victim) -> others
--   d_*damagetaken(P) =
--       sum(received rows where event.victimid = P)        -- victim P took damage
--     + sum(dealt    rows where target         = P)        -- P took damage from victim
--
-- ITEM_* adjustments (goldspent, clamped at 0):
--   ITEM_PURCHASED: d_goldspent += item price (purchase inflated goldspent)
--   ITEM_SOLD:      d_goldspent -= round(price * 0.7) (sold item recovers 70% sell value)
--   ITEM_UNDO:      d_goldspent -= goldgain (reverses the prior transaction's gold delta)
--
-- Caveats / NOT adjusted (CHAMPION_KILL-derived but harder to reconstruct):
--   - doublekills / triplekills / quadrakills / pentakills / unrealkills
--   - killingsprees / largestkillingspree / largestmultikill
--   - firstbloodkill / firstbloodassist (padding window is end-of-game)
--   - bounty / shutdownbounty -> goldearned (no exposed gold mapping)
--   - totaldamagedealt / *damagedealt (includes non-champion damage; the
--     kill-event damage tables only describe champion-to-champion exchanges)
--   - damageselfmitigated, totalheal*, timeccingothers, totaltimeccdealt
--     (not present on CHAMPION_KILL events)
--
-- Run after the raw game_data.* schemas and before 4000_filter_build.sql so
-- the filter applies to corrected stats from game_data.

SET max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768,
    join_algorithm = 'partial_merge';

TRUNCATE TABLE game_data.participant_stats_corrected;

INSERT INTO game_data.participant_stats_corrected
WITH
long_game_ids AS (
    SELECT matchid
    FROM game_data.info
    WHERE gameduration > 1080
),

game_end_ts AS (
    SELECT
        matchid,
        max(timestamp) AS end_ts
    FROM game_data.tl_game_end
    SEMI JOIN long_game_ids USING (matchid)
    GROUP BY matchid
),

-- CHAMPION_KILL events in the final 15 s, deduplicated across run_id.
padding_events AS (
    SELECT
        ck.matchid,
        ck.champion_kill_event_id,
        any(ck.killerid) AS killerid,
        any(ck.victimid) AS victimid,
        any(ck.assistingparticipantids) AS assistingparticipantids
    FROM game_data.tl_champion_kill AS ck
    INNER JOIN game_end_ts AS g USING (matchid)
    WHERE ck.timestamp >= g.end_ts - 15000
    GROUP BY ck.matchid, ck.champion_kill_event_id
),

-- ITEM_PURCHASED events in the final 15 s, deduplicated across run_id.
padding_purchased AS (
    SELECT
        ip.matchid,
        toUInt8(ip.participantid) AS participantid,
        dictGetOrDefault('game_data.item_info_dict', 'price', toUInt32(ip.itemid), toUInt32(0)) AS price
    FROM game_data.tl_item_purchased AS ip
    INNER JOIN game_end_ts AS g USING (matchid)
    WHERE ip.timestamp >= g.end_ts - 15000
    GROUP BY ip.matchid, ip.frame_timestamp, ip.timestamp, ip.participantid, ip.itemid
),

-- ITEM_SOLD events in the final 15 s, deduplicated across run_id.
-- Sold items recover 70% of their price (sell value).
padding_sold AS (
    SELECT
        is_.matchid,
        toUInt8(is_.participantid) AS participantid,
        dictGetOrDefault('game_data.item_info_dict', 'price', toUInt32(is_.itemid), toUInt32(0)) AS price
    FROM game_data.tl_item_sold AS is_
    INNER JOIN game_end_ts AS g USING (matchid)
    WHERE is_.timestamp >= g.end_ts - 15000
    GROUP BY is_.matchid, is_.frame_timestamp, is_.timestamp, is_.participantid, is_.itemid
),

-- ITEM_UNDO events in the final 15 s, deduplicated across run_id.
-- goldgain is the gold delta from the undo (positive = purchase reversed).
padding_undo AS (
    SELECT
        iu.matchid,
        toUInt8(iu.participantid) AS participantid,
        any(iu.goldgain) AS goldgain
    FROM game_data.tl_item_undo AS iu
    INNER JOIN game_end_ts AS g USING (matchid)
    WHERE iu.timestamp >= g.end_ts - 15000
    GROUP BY iu.matchid, iu.frame_timestamp, iu.timestamp, iu.participantid
),

-- Per-(match, participant) deltas. Each UNION branch emits a single
-- contribution; the outer GROUP BY sums them once.
deltas AS (
    SELECT
        matchid,
        participantid,
        sum(d_kills) AS d_kills,
        sum(d_deaths) AS d_deaths,
        sum(d_assists) AS d_assists,
        sum(d_phys_dealt) AS d_phys_dealt,
        sum(d_magic_dealt) AS d_magic_dealt,
        sum(d_true_dealt) AS d_true_dealt,
        sum(d_phys_taken) AS d_phys_taken,
        sum(d_magic_taken) AS d_magic_taken,
        sum(d_true_taken) AS d_true_taken,
        sum(d_goldspent) AS d_goldspent
    FROM (
        -- Killer credit.
        SELECT
            matchid,
            toUInt8(killerid) AS participantid,
            toUInt8(1) AS d_kills,
            toUInt8(0) AS d_deaths,
            toUInt16(0) AS d_assists,
            toUInt16(0) AS d_phys_dealt,
            toUInt16(0) AS d_magic_dealt,
            toUInt16(0) AS d_true_dealt,
            toUInt16(0) AS d_phys_taken,
            toUInt16(0) AS d_magic_taken,
            toUInt16(0) AS d_true_taken,
            toInt32(0) AS d_goldspent
        FROM padding_events
        WHERE killerid > 0

        UNION ALL

        -- Victim debit (death).
        SELECT
            matchid,
            toUInt8(victimid) AS participantid,
            toUInt8(0) AS d_kills,
            toUInt8(1) AS d_deaths,
            toUInt16(0) AS d_assists,
            toUInt16(0) AS d_phys_dealt,
            toUInt16(0) AS d_magic_dealt,
            toUInt16(0) AS d_true_dealt,
            toUInt16(0) AS d_phys_taken,
            toUInt16(0) AS d_magic_taken,
            toUInt16(0) AS d_true_taken,
            toInt32(0) AS d_goldspent
        FROM padding_events
        WHERE victimid > 0

        UNION ALL

        -- Each assisting participant.
        SELECT
            matchid,
            toUInt8(arrayJoin(assistingparticipantids)) AS participantid,
            toUInt8(0) AS d_kills,
            toUInt8(0) AS d_deaths,
            toUInt16(1) AS d_assists,
            toUInt16(0) AS d_phys_dealt,
            toUInt16(0) AS d_magic_dealt,
            toUInt16(0) AS d_true_dealt,
            toUInt16(0) AS d_phys_taken,
            toUInt16(0) AS d_magic_taken,
            toUInt16(0) AS d_true_taken,
            toInt32(0) AS d_goldspent
        FROM padding_events

        UNION ALL

        -- Attacker -> victim damage: credit attacker.damageDealtToChampions.
        SELECT
            r.matchid,
            r.participantid,
            toUInt8(0) AS d_kills,
            toUInt8(0) AS d_deaths,
            toUInt16(0) AS d_assists,
            toUInt16(r.physicaldamage) AS d_phys_dealt,
            toUInt16(r.magicdamage) AS d_magic_dealt,
            toUInt16(r.truedamage) AS d_true_dealt,
            toUInt16(0) AS d_phys_taken,
            toUInt16(0) AS d_magic_taken,
            toUInt16(0) AS d_true_taken,
            toInt32(0) AS d_goldspent
        FROM game_data.tl_ck_victim_damage_received AS r
        INNER JOIN padding_events USING (matchid, champion_kill_event_id)
        WHERE r.participantid > 0

        UNION ALL

        -- Same rows: debit victim.damageTaken.
        SELECT
            p.matchid,
            toUInt8(p.victimid) AS participantid,
            toUInt8(0) AS d_kills,
            toUInt8(0) AS d_deaths,
            toUInt16(0) AS d_assists,
            toUInt16(0) AS d_phys_dealt,
            toUInt16(0) AS d_magic_dealt,
            toUInt16(0) AS d_true_dealt,
            toUInt16(r.physicaldamage) AS d_phys_taken,
            toUInt16(r.magicdamage) AS d_magic_taken,
            toUInt16(r.truedamage) AS d_true_taken,
            toInt32(0) AS d_goldspent
        FROM game_data.tl_ck_victim_damage_received AS r
        INNER JOIN padding_events AS p USING (matchid, champion_kill_event_id)
        WHERE p.victimid > 0

        UNION ALL

        -- Victim's return damage: credit victim.damageDealtToChampions.
        SELECT
            p.matchid,
            toUInt8(p.victimid) AS participantid,
            toUInt8(0) AS d_kills,
            toUInt8(0) AS d_deaths,
            toUInt16(0) AS d_assists,
            toUInt16(d.physicaldamage) AS d_phys_dealt,
            toUInt16(d.magicdamage) AS d_magic_dealt,
            toUInt16(d.truedamage) AS d_true_dealt,
            toUInt16(0) AS d_phys_taken,
            toUInt16(0) AS d_magic_taken,
            toUInt16(0) AS d_true_taken,
            toInt32(0) AS d_goldspent
        FROM game_data.tl_ck_victim_damage_dealt AS d
        INNER JOIN padding_events AS p USING (matchid, champion_kill_event_id)
        WHERE p.victimid > 0

        UNION ALL

        -- Same rows: debit recipient.damageTaken.
        SELECT
            d.matchid,
            d.participantid,
            toUInt8(0) AS d_kills,
            toUInt8(0) AS d_deaths,
            toUInt16(0) AS d_assists,
            toUInt16(0) AS d_phys_dealt,
            toUInt16(0) AS d_magic_dealt,
            toUInt16(0) AS d_true_dealt,
            toUInt16(d.physicaldamage) AS d_phys_taken,
            toUInt16(d.magicdamage) AS d_magic_taken,
            toUInt16(d.truedamage) AS d_true_taken,
            toInt32(0) AS d_goldspent
        FROM game_data.tl_ck_victim_damage_dealt AS d
        INNER JOIN padding_events USING (matchid, champion_kill_event_id)
        WHERE d.participantid > 0

        UNION ALL

        -- ITEM_PURCHASED: full price subtracted from goldspent.
        SELECT
            matchid,
            participantid,
            toUInt8(0) AS d_kills,
            toUInt8(0) AS d_deaths,
            toUInt16(0) AS d_assists,
            toUInt16(0) AS d_phys_dealt,
            toUInt16(0) AS d_magic_dealt,
            toUInt16(0) AS d_true_dealt,
            toUInt16(0) AS d_phys_taken,
            toUInt16(0) AS d_magic_taken,
            toUInt16(0) AS d_true_taken,
            toInt32(price) AS d_goldspent
        FROM padding_purchased

        UNION ALL

        -- ITEM_SOLD: sell value (70%) credited back against goldspent.
        SELECT
            matchid,
            participantid,
            toUInt8(0) AS d_kills,
            toUInt8(0) AS d_deaths,
            toUInt16(0) AS d_assists,
            toUInt16(0) AS d_phys_dealt,
            toUInt16(0) AS d_magic_dealt,
            toUInt16(0) AS d_true_dealt,
            toUInt16(0) AS d_phys_taken,
            toUInt16(0) AS d_magic_taken,
            toUInt16(0) AS d_true_taken,
            -toInt32(round(price * 0.7)) AS d_goldspent
        FROM padding_sold

        UNION ALL

        -- ITEM_UNDO: goldgain is the gold delta from reversing the prior transaction.
        SELECT
            matchid,
            participantid,
            toUInt8(0) AS d_kills,
            toUInt8(0) AS d_deaths,
            toUInt16(0) AS d_assists,
            toUInt16(0) AS d_phys_dealt,
            toUInt16(0) AS d_magic_dealt,
            toUInt16(0) AS d_true_dealt,
            toUInt16(0) AS d_phys_taken,
            toUInt16(0) AS d_magic_taken,
            toUInt16(0) AS d_true_taken,
            -toInt32(goldgain) AS d_goldspent
        FROM padding_undo
    )
    GROUP BY matchid, participantid
)

-- Pass every column of participant_stats through unchanged except the twelve
-- replaced below. ps.* REPLACE keeps this build resilient to schema additions.
SELECT ps.* REPLACE (
    toUInt8(greatest(toInt32(ps.kills) - toInt32(d.d_kills), 0)) AS kills,
    toUInt8(greatest(toInt32(ps.deaths) - toInt32(d.d_deaths), 0)) AS deaths,
    toUInt8(greatest(toInt32(ps.assists) - toInt32(d.d_assists), 0)) AS assists,
    toUInt32(greatest(
        toInt64(ps.totaldamagedealttochampions)
        - toInt64(d.d_phys_dealt + d.d_magic_dealt + d.d_true_dealt),
        0
    )) AS totaldamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.physicaldamagedealttochampions) - toInt64(d.d_phys_dealt), 0
    )) AS physicaldamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.magicdamagedealttochampions) - toInt64(d.d_magic_dealt), 0
    )) AS magicdamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.truedamagedealttochampions) - toInt64(d.d_true_dealt), 0
    )) AS truedamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.totaldamagetaken)
        - toInt64(d.d_phys_taken + d.d_magic_taken + d.d_true_taken),
        0
    )) AS totaldamagetaken,
    toUInt32(greatest(
        toInt64(ps.physicaldamagetaken) - toInt64(d.d_phys_taken), 0
    )) AS physicaldamagetaken,
    toUInt32(greatest(
        toInt64(ps.magicdamagetaken) - toInt64(d.d_magic_taken), 0
    )) AS magicdamagetaken,
    toUInt32(greatest(
        toInt64(ps.truedamagetaken) - toInt64(d.d_true_taken), 0
    )) AS truedamagetaken,
    toUInt32(greatest(
        toInt64(ps.goldspent) - toInt64(d.d_goldspent), 0
    )) AS goldspent
)
FROM game_data.participant_stats AS ps
SEMI JOIN long_game_ids USING (matchid)
LEFT JOIN deltas AS d
    ON ps.matchid = d.matchid AND ps.participantid = d.participantid;
