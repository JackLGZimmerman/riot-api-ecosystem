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
--   ITEM_SOLD: d_goldspent -= round(price * 0.7) (sold item recovers 70% sell value)
--   ITEM_UNDO: d_goldspent -= goldgain (reverses the prior transaction's gold delta)
--
-- Run after the raw game_data.* schemas and before 4000_filter_build.sql so
-- the filter applies to corrected stats from the latest-season long-game
-- population in game_data.

SET max_threads = 1, max_block_size = 8192, max_insert_block_size = 32768,
    join_algorithm = 'partial_merge',
    max_bytes_before_external_group_by = 1000000000,
    max_bytes_before_external_sort = 1000000000;

DROP TABLE IF EXISTS game_data.participant_stats_corrected_long_game_ids;
DROP TABLE IF EXISTS game_data.participant_stats_corrected_game_end_ts;
DROP TABLE IF EXISTS game_data.participant_stats_corrected_padding_events;
DROP TABLE IF EXISTS game_data.participant_stats_corrected_padding_purchased;
DROP TABLE IF EXISTS game_data.participant_stats_corrected_padding_sold;
DROP TABLE IF EXISTS game_data.participant_stats_corrected_padding_undo;
DROP TABLE IF EXISTS game_data.participant_stats_corrected_padding_damage_received;
DROP TABLE IF EXISTS game_data.participant_stats_corrected_padding_damage_dealt;
DROP TABLE IF EXISTS game_data.participant_stats_corrected_delta_totals;
DROP TABLE IF EXISTS game_data.participant_stats_corrected_rows;

CREATE TABLE game_data.participant_stats_corrected_long_game_ids
(
    `matchid` String
)
ENGINE = MergeTree
ORDER BY matchid;

INSERT INTO game_data.participant_stats_corrected_long_game_ids
SELECT i.matchid
FROM game_data.info AS i
WHERE
    i.season = (SELECT max(latest_i.season) FROM game_data.info AS latest_i)
    AND i.gameduration > 990;

CREATE TABLE game_data.participant_stats_corrected_game_end_ts
(
    `matchid` String,
    `end_ts` UInt64
)
ENGINE = MergeTree
ORDER BY matchid;

INSERT INTO game_data.participant_stats_corrected_game_end_ts
SELECT
    matchid,
    max(timestamp) AS end_ts
FROM game_data.tl_game_end
SEMI JOIN game_data.participant_stats_corrected_long_game_ids USING (matchid)
GROUP BY matchid;

-- CHAMPION_KILL events in the final 15 s.
CREATE TABLE game_data.participant_stats_corrected_padding_events
(
    `matchid` String,
    `champion_kill_event_id` String,
    `killerid` Int8,
    `victimid` Int8,
    `assistingparticipantids` Array(UInt8)
)
ENGINE = MergeTree
ORDER BY (matchid, champion_kill_event_id);

INSERT INTO game_data.participant_stats_corrected_padding_events
SELECT
    ck.matchid,
    ck.champion_kill_event_id,
    any(ck.killerid) AS killerid,
    any(ck.victimid) AS victimid,
    any(ck.assistingparticipantids) AS assistingparticipantids
FROM game_data.tl_champion_kill AS ck
INNER JOIN game_data.participant_stats_corrected_game_end_ts AS g USING (matchid)
WHERE ck.timestamp >= g.end_ts - 15000
GROUP BY ck.matchid, ck.champion_kill_event_id;

-- ITEM_PURCHASED events in the final 15 s.
CREATE TABLE game_data.participant_stats_corrected_padding_purchased
(
    `matchid` String,
    `participantid` UInt8,
    `price` UInt32
)
ENGINE = MergeTree
ORDER BY (matchid, participantid);

INSERT INTO game_data.participant_stats_corrected_padding_purchased
SELECT
    ip.matchid,
    ip.participantid,
    dictGetOrDefault('game_data.item_info_dict', 'price', toUInt64(ip.itemid), toUInt32(0)) AS price
FROM game_data.tl_item_purchased AS ip
INNER JOIN game_data.participant_stats_corrected_game_end_ts AS g USING (matchid)
WHERE ip.timestamp >= g.end_ts - 15000
GROUP BY ip.matchid, ip.frame_timestamp, ip.timestamp, ip.participantid, ip.itemid;

-- ITEM_SOLD events in the final 15 s. Sold items recover 70% of their price.
CREATE TABLE game_data.participant_stats_corrected_padding_sold
(
    `matchid` String,
    `participantid` UInt8,
    `price` UInt32
)
ENGINE = MergeTree
ORDER BY (matchid, participantid);

INSERT INTO game_data.participant_stats_corrected_padding_sold
SELECT
    is_.matchid,
    is_.participantid,
    dictGetOrDefault('game_data.item_info_dict', 'price', toUInt64(is_.itemid), toUInt32(0)) AS price
FROM game_data.tl_item_sold AS is_
INNER JOIN game_data.participant_stats_corrected_game_end_ts AS g USING (matchid)
WHERE is_.timestamp >= g.end_ts - 15000
GROUP BY is_.matchid, is_.frame_timestamp, is_.timestamp, is_.participantid, is_.itemid;

-- ITEM_UNDO events in the final 15 s. goldgain is the gold delta from the
-- undo (positive = purchase reversed).
CREATE TABLE game_data.participant_stats_corrected_padding_undo
(
    `matchid` String,
    `participantid` UInt8,
    `goldgain` Int32
)
ENGINE = MergeTree
ORDER BY (matchid, participantid);

INSERT INTO game_data.participant_stats_corrected_padding_undo
SELECT
    iu.matchid,
    iu.participantid,
    any(iu.goldgain) AS goldgain
FROM game_data.tl_item_undo AS iu
INNER JOIN game_data.participant_stats_corrected_game_end_ts AS g USING (matchid)
WHERE iu.timestamp >= g.end_ts - 15000
GROUP BY iu.matchid, iu.frame_timestamp, iu.timestamp, iu.participantid;

-- Materialize the JOIN of each damage-direction table with padding_events
-- once. Each underlying full-table scan (23.7 GiB and 13.5 GiB) is paid once,
-- not twice, and downstream delta sums read from a small table.
CREATE TABLE game_data.participant_stats_corrected_padding_damage_received
(
    `matchid` String,
    `attacker_id` UInt8,
    `victim_id` Int8,
    `physicaldamage` UInt16,
    `magicdamage` UInt16,
    `truedamage` UInt32
)
ENGINE = MergeTree
ORDER BY matchid;

INSERT INTO game_data.participant_stats_corrected_padding_damage_received
SELECT
    r.matchid,
    r.participantid AS attacker_id,
    p.victimid AS victim_id,
    r.physicaldamage,
    r.magicdamage,
    r.truedamage
FROM game_data.tl_ck_victim_damage_received AS r
INNER JOIN game_data.participant_stats_corrected_padding_events AS p USING (matchid, champion_kill_event_id);

CREATE TABLE game_data.participant_stats_corrected_padding_damage_dealt
(
    `matchid` String,
    `attacker_id` UInt8,
    `victim_id` Int8,
    `physicaldamage` UInt16,
    `magicdamage` UInt16,
    `truedamage` UInt32
)
ENGINE = MergeTree
ORDER BY matchid;

INSERT INTO game_data.participant_stats_corrected_padding_damage_dealt
SELECT
    d.matchid,
    d.participantid AS attacker_id,
    p.victimid AS victim_id,
    d.physicaldamage,
    d.magicdamage,
    d.truedamage
FROM game_data.tl_ck_victim_damage_dealt AS d
INNER JOIN game_data.participant_stats_corrected_padding_events AS p USING (matchid, champion_kill_event_id);

CREATE TABLE game_data.participant_stats_corrected_delta_totals
(
    `matchid` String,
    `participantid` UInt8,
    `d_kills` Int64,
    `d_deaths` Int64,
    `d_assists` Int64,
    `d_phys_dealt` Int64,
    `d_magic_dealt` Int64,
    `d_true_dealt` Int64,
    `d_phys_taken` Int64,
    `d_magic_taken` Int64,
    `d_true_taken` Int64,
    `d_goldspent` Int64
)
ENGINE = MergeTree
ORDER BY (matchid, participantid);

-- Single aggregated INSERT: every per-source partial delta is unioned and
-- summed in one pass. Replaces the previous 9-INSERT staging + re-aggregate.
INSERT INTO game_data.participant_stats_corrected_delta_totals
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
FROM
(
    -- Killer credit.
    SELECT
        matchid, toUInt8(killerid) AS participantid,
        1 AS d_kills, 0 AS d_deaths, 0 AS d_assists,
        0 AS d_phys_dealt, 0 AS d_magic_dealt, 0 AS d_true_dealt,
        0 AS d_phys_taken, 0 AS d_magic_taken, 0 AS d_true_taken,
        0 AS d_goldspent
    FROM game_data.participant_stats_corrected_padding_events WHERE killerid > 0
    UNION ALL
    -- Victim debit (death).
    SELECT
        matchid, toUInt8(victimid),
        0, 1, 0,
        0, 0, 0,
        0, 0, 0,
        0
    FROM game_data.participant_stats_corrected_padding_events WHERE victimid > 0
    UNION ALL
    -- Each assisting participant.
    SELECT
        matchid, toUInt8(arrayJoin(assistingparticipantids)),
        0, 0, 1,
        0, 0, 0,
        0, 0, 0,
        0
    FROM game_data.participant_stats_corrected_padding_events
    UNION ALL
    -- Attacker -> victim damage: credit attacker.damageDealtToChampions.
    SELECT
        matchid, attacker_id,
        0, 0, 0,
        physicaldamage, magicdamage, truedamage,
        0, 0, 0,
        0
    FROM game_data.participant_stats_corrected_padding_damage_received WHERE attacker_id > 0
    UNION ALL
    -- Same rows: debit victim.damageTaken.
    SELECT
        matchid, toUInt8(victim_id),
        0, 0, 0,
        0, 0, 0,
        physicaldamage, magicdamage, truedamage,
        0
    FROM game_data.participant_stats_corrected_padding_damage_received WHERE victim_id > 0
    UNION ALL
    -- Victim's return damage: credit victim.damageDealtToChampions.
    SELECT
        matchid, toUInt8(victim_id),
        0, 0, 0,
        physicaldamage, magicdamage, truedamage,
        0, 0, 0,
        0
    FROM game_data.participant_stats_corrected_padding_damage_dealt WHERE victim_id > 0
    UNION ALL
    -- Same rows: debit recipient.damageTaken.
    SELECT
        matchid, attacker_id,
        0, 0, 0,
        0, 0, 0,
        physicaldamage, magicdamage, truedamage,
        0
    FROM game_data.participant_stats_corrected_padding_damage_dealt WHERE attacker_id > 0
    UNION ALL
    -- ITEM_PURCHASED: full price subtracted from goldspent.
    SELECT
        matchid, participantid,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        price
    FROM game_data.participant_stats_corrected_padding_purchased
    UNION ALL
    -- ITEM_SOLD: sell value (70%) credited back against goldspent.
    SELECT
        matchid, participantid,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        -toInt64(round(price * 0.7))
    FROM game_data.participant_stats_corrected_padding_sold
    UNION ALL
    -- ITEM_UNDO: goldgain is the gold delta from reversing the prior transaction.
    SELECT
        matchid, participantid,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        -toInt64(goldgain)
    FROM game_data.participant_stats_corrected_padding_undo
)
GROUP BY matchid, participantid;

-- Materialize the corrected rows once. Pays the participant_stats scan +
-- delta_totals join + REPLACE arithmetic exactly once, instead of 16 times via
-- a re-evaluated view. The final INSERT below is then a cheap streaming copy.
CREATE TABLE game_data.participant_stats_corrected_rows AS game_data.participant_stats
ENGINE = MergeTree
ORDER BY (matchid, participantid, puuid);

INSERT INTO game_data.participant_stats_corrected_rows
SELECT ps.* REPLACE (
    toUInt8(greatest(toInt32(ps.kills) - toInt32(ifNull(d.d_kills, 0)), 0)) AS kills,
    toUInt8(greatest(toInt32(ps.deaths) - toInt32(ifNull(d.d_deaths, 0)), 0)) AS deaths,
    toUInt8(greatest(toInt32(ps.assists) - toInt32(ifNull(d.d_assists, 0)), 0)) AS assists,
    toUInt32(greatest(
        toInt64(ps.totaldamagedealttochampions)
        - (ifNull(d.d_phys_dealt, 0) + ifNull(d.d_magic_dealt, 0) + ifNull(d.d_true_dealt, 0)),
        0
    )) AS totaldamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.physicaldamagedealttochampions) - ifNull(d.d_phys_dealt, 0), 0
    )) AS physicaldamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.magicdamagedealttochampions) - ifNull(d.d_magic_dealt, 0), 0
    )) AS magicdamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.truedamagedealttochampions) - ifNull(d.d_true_dealt, 0), 0
    )) AS truedamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.totaldamagetaken)
        - (ifNull(d.d_phys_taken, 0) + ifNull(d.d_magic_taken, 0) + ifNull(d.d_true_taken, 0)),
        0
    )) AS totaldamagetaken,
    toUInt32(greatest(
        toInt64(ps.physicaldamagetaken) - ifNull(d.d_phys_taken, 0), 0
    )) AS physicaldamagetaken,
    toUInt32(greatest(
        toInt64(ps.magicdamagetaken) - ifNull(d.d_magic_taken, 0), 0
    )) AS magicdamagetaken,
    toUInt32(greatest(
        toInt64(ps.truedamagetaken) - ifNull(d.d_true_taken, 0), 0
    )) AS truedamagetaken,
    toUInt32(greatest(
        toInt64(ps.goldspent) - ifNull(d.d_goldspent, 0), 0
    )) AS goldspent
)
FROM
(
    SELECT * FROM game_data.participant_stats
) AS ps
SEMI JOIN game_data.participant_stats_corrected_long_game_ids USING (matchid)
LEFT JOIN game_data.participant_stats_corrected_delta_totals AS d
    ON ps.matchid = d.matchid AND ps.participantid = d.participantid;

TRUNCATE TABLE game_data.participant_stats_corrected;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows;

DROP TABLE game_data.participant_stats_corrected_rows;
DROP TABLE game_data.participant_stats_corrected_long_game_ids;
DROP TABLE game_data.participant_stats_corrected_game_end_ts;
DROP TABLE game_data.participant_stats_corrected_padding_events;
DROP TABLE game_data.participant_stats_corrected_padding_purchased;
DROP TABLE game_data.participant_stats_corrected_padding_sold;
DROP TABLE game_data.participant_stats_corrected_padding_undo;
DROP TABLE game_data.participant_stats_corrected_padding_damage_received;
DROP TABLE game_data.participant_stats_corrected_padding_damage_dealt;
DROP TABLE game_data.participant_stats_corrected_delta_totals;
