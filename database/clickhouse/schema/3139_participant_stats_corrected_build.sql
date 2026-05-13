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
-- the filter applies to corrected stats from game_data.

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
DROP TABLE IF EXISTS game_data.participant_stats_corrected_deltas;
DROP TABLE IF EXISTS game_data.participant_stats_corrected_delta_totals;
DROP VIEW IF EXISTS game_data.participant_stats_corrected_rows;

CREATE TABLE game_data.participant_stats_corrected_long_game_ids
(
    `matchid` String
)
ENGINE = MergeTree
ORDER BY matchid;

INSERT INTO game_data.participant_stats_corrected_long_game_ids
SELECT matchid
FROM game_data.info FINAL
WHERE gameduration > 1080;

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
FROM game_data.tl_game_end FINAL
SEMI JOIN game_data.participant_stats_corrected_long_game_ids USING (matchid)
GROUP BY matchid;

-- CHAMPION_KILL events in the final 15 s, deduplicated across run_id.
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
FROM game_data.tl_champion_kill AS ck FINAL
INNER JOIN game_data.participant_stats_corrected_game_end_ts AS g USING (matchid)
WHERE ck.timestamp >= g.end_ts - 15000
GROUP BY ck.matchid, ck.champion_kill_event_id;

-- ITEM_PURCHASED events in the final 15 s, deduplicated across run_id.
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
    toUInt8(ip.participantid) AS participantid,
    dictGetOrDefault('game_data.item_info_dict', 'price', toUInt32(ip.itemid), toUInt32(0)) AS price
FROM game_data.tl_item_purchased AS ip FINAL
INNER JOIN game_data.participant_stats_corrected_game_end_ts AS g USING (matchid)
WHERE ip.timestamp >= g.end_ts - 15000
GROUP BY ip.matchid, ip.frame_timestamp, ip.timestamp, ip.participantid, ip.itemid;

-- ITEM_SOLD events in the final 15 s, deduplicated across run_id.
-- Sold items recover 70% of their price (sell value).
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
    toUInt8(is_.participantid) AS participantid,
    dictGetOrDefault('game_data.item_info_dict', 'price', toUInt32(is_.itemid), toUInt32(0)) AS price
FROM game_data.tl_item_sold AS is_ FINAL
INNER JOIN game_data.participant_stats_corrected_game_end_ts AS g USING (matchid)
WHERE is_.timestamp >= g.end_ts - 15000
GROUP BY is_.matchid, is_.frame_timestamp, is_.timestamp, is_.participantid, is_.itemid;

-- ITEM_UNDO events in the final 15 s, deduplicated across run_id.
-- goldgain is the gold delta from the undo (positive = purchase reversed).
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
    toUInt8(iu.participantid) AS participantid,
    any(iu.goldgain) AS goldgain
FROM game_data.tl_item_undo AS iu FINAL
INNER JOIN game_data.participant_stats_corrected_game_end_ts AS g USING (matchid)
WHERE iu.timestamp >= g.end_ts - 15000
GROUP BY iu.matchid, iu.frame_timestamp, iu.timestamp, iu.participantid;

CREATE TABLE game_data.participant_stats_corrected_deltas
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

-- Killer credit.
INSERT INTO game_data.participant_stats_corrected_deltas
SELECT
    matchid,
    toUInt8(killerid) AS participantid,
    1 AS d_kills,
    0 AS d_deaths,
    0 AS d_assists,
    0 AS d_phys_dealt,
    0 AS d_magic_dealt,
    0 AS d_true_dealt,
    0 AS d_phys_taken,
    0 AS d_magic_taken,
    0 AS d_true_taken,
    0 AS d_goldspent
FROM game_data.participant_stats_corrected_padding_events
WHERE killerid > 0;

-- Victim debit (death).
INSERT INTO game_data.participant_stats_corrected_deltas
SELECT
    matchid,
    toUInt8(victimid) AS participantid,
    0 AS d_kills,
    1 AS d_deaths,
    0 AS d_assists,
    0 AS d_phys_dealt,
    0 AS d_magic_dealt,
    0 AS d_true_dealt,
    0 AS d_phys_taken,
    0 AS d_magic_taken,
    0 AS d_true_taken,
    0 AS d_goldspent
FROM game_data.participant_stats_corrected_padding_events
WHERE victimid > 0;

-- Each assisting participant.
INSERT INTO game_data.participant_stats_corrected_deltas
SELECT
    matchid,
    toUInt8(arrayJoin(assistingparticipantids)) AS participantid,
    0 AS d_kills,
    0 AS d_deaths,
    1 AS d_assists,
    0 AS d_phys_dealt,
    0 AS d_magic_dealt,
    0 AS d_true_dealt,
    0 AS d_phys_taken,
    0 AS d_magic_taken,
    0 AS d_true_taken,
    0 AS d_goldspent
FROM game_data.participant_stats_corrected_padding_events;

-- Attacker -> victim damage: credit attacker.damageDealtToChampions.
INSERT INTO game_data.participant_stats_corrected_deltas
SELECT
    r.matchid,
    r.participantid,
    0 AS d_kills,
    0 AS d_deaths,
    0 AS d_assists,
    toInt64(r.physicaldamage) AS d_phys_dealt,
    toInt64(r.magicdamage) AS d_magic_dealt,
    toInt64(r.truedamage) AS d_true_dealt,
    0 AS d_phys_taken,
    0 AS d_magic_taken,
    0 AS d_true_taken,
    0 AS d_goldspent
FROM game_data.tl_ck_victim_damage_received AS r FINAL
INNER JOIN game_data.participant_stats_corrected_padding_events USING (matchid, champion_kill_event_id)
WHERE r.participantid > 0;

-- Same rows: debit victim.damageTaken.
INSERT INTO game_data.participant_stats_corrected_deltas
SELECT
    p.matchid,
    toUInt8(p.victimid) AS participantid,
    0 AS d_kills,
    0 AS d_deaths,
    0 AS d_assists,
    0 AS d_phys_dealt,
    0 AS d_magic_dealt,
    0 AS d_true_dealt,
    toInt64(r.physicaldamage) AS d_phys_taken,
    toInt64(r.magicdamage) AS d_magic_taken,
    toInt64(r.truedamage) AS d_true_taken,
    0 AS d_goldspent
FROM game_data.tl_ck_victim_damage_received AS r FINAL
INNER JOIN game_data.participant_stats_corrected_padding_events AS p USING (matchid, champion_kill_event_id)
WHERE p.victimid > 0;

-- Victim's return damage: credit victim.damageDealtToChampions.
INSERT INTO game_data.participant_stats_corrected_deltas
SELECT
    p.matchid,
    toUInt8(p.victimid) AS participantid,
    0 AS d_kills,
    0 AS d_deaths,
    0 AS d_assists,
    toInt64(d.physicaldamage) AS d_phys_dealt,
    toInt64(d.magicdamage) AS d_magic_dealt,
    toInt64(d.truedamage) AS d_true_dealt,
    0 AS d_phys_taken,
    0 AS d_magic_taken,
    0 AS d_true_taken,
    0 AS d_goldspent
FROM game_data.tl_ck_victim_damage_dealt AS d FINAL
INNER JOIN game_data.participant_stats_corrected_padding_events AS p USING (matchid, champion_kill_event_id)
WHERE p.victimid > 0;

-- Same rows: debit recipient.damageTaken.
INSERT INTO game_data.participant_stats_corrected_deltas
SELECT
    d.matchid,
    d.participantid,
    0 AS d_kills,
    0 AS d_deaths,
    0 AS d_assists,
    0 AS d_phys_dealt,
    0 AS d_magic_dealt,
    0 AS d_true_dealt,
    toInt64(d.physicaldamage) AS d_phys_taken,
    toInt64(d.magicdamage) AS d_magic_taken,
    toInt64(d.truedamage) AS d_true_taken,
    0 AS d_goldspent
FROM game_data.tl_ck_victim_damage_dealt AS d FINAL
INNER JOIN game_data.participant_stats_corrected_padding_events USING (matchid, champion_kill_event_id)
WHERE d.participantid > 0;

-- ITEM_PURCHASED: full price subtracted from goldspent.
INSERT INTO game_data.participant_stats_corrected_deltas
SELECT
    matchid,
    participantid,
    0 AS d_kills,
    0 AS d_deaths,
    0 AS d_assists,
    0 AS d_phys_dealt,
    0 AS d_magic_dealt,
    0 AS d_true_dealt,
    0 AS d_phys_taken,
    0 AS d_magic_taken,
    0 AS d_true_taken,
    toInt64(price) AS d_goldspent
FROM game_data.participant_stats_corrected_padding_purchased;

-- ITEM_SOLD: sell value (70%) credited back against goldspent.
INSERT INTO game_data.participant_stats_corrected_deltas
SELECT
    matchid,
    participantid,
    0 AS d_kills,
    0 AS d_deaths,
    0 AS d_assists,
    0 AS d_phys_dealt,
    0 AS d_magic_dealt,
    0 AS d_true_dealt,
    0 AS d_phys_taken,
    0 AS d_magic_taken,
    0 AS d_true_taken,
    -toInt64(round(price * 0.7)) AS d_goldspent
FROM game_data.participant_stats_corrected_padding_sold;

-- ITEM_UNDO: goldgain is the gold delta from reversing the prior transaction.
INSERT INTO game_data.participant_stats_corrected_deltas
SELECT
    matchid,
    participantid,
    0 AS d_kills,
    0 AS d_deaths,
    0 AS d_assists,
    0 AS d_phys_dealt,
    0 AS d_magic_dealt,
    0 AS d_true_dealt,
    0 AS d_phys_taken,
    0 AS d_magic_taken,
    0 AS d_true_taken,
    -toInt64(goldgain) AS d_goldspent
FROM game_data.participant_stats_corrected_padding_undo;

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
FROM game_data.participant_stats_corrected_deltas
GROUP BY matchid, participantid;

-- Pass every column of participant_stats through unchanged except the twelve
-- replaced below. ps.* REPLACE keeps this build resilient to schema additions.
-- Read FINAL so duplicate run_ids collapse before this clean snapshot is built.
CREATE VIEW game_data.participant_stats_corrected_rows AS
SELECT ps.* REPLACE (
    toUInt8(greatest(toInt32(ps.kills) - toInt32(ifNull(d.d_kills, 0)), 0)) AS kills,
    toUInt8(greatest(toInt32(ps.deaths) - toInt32(ifNull(d.d_deaths, 0)), 0)) AS deaths,
    toUInt8(greatest(toInt32(ps.assists) - toInt32(ifNull(d.d_assists, 0)), 0)) AS assists,
    toUInt32(greatest(
        toInt64(ps.totaldamagedealttochampions)
        - toInt64(ifNull(d.d_phys_dealt, 0) + ifNull(d.d_magic_dealt, 0) + ifNull(d.d_true_dealt, 0)),
        0
    )) AS totaldamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.physicaldamagedealttochampions) - toInt64(ifNull(d.d_phys_dealt, 0)), 0
    )) AS physicaldamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.magicdamagedealttochampions) - toInt64(ifNull(d.d_magic_dealt, 0)), 0
    )) AS magicdamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.truedamagedealttochampions) - toInt64(ifNull(d.d_true_dealt, 0)), 0
    )) AS truedamagedealttochampions,
    toUInt32(greatest(
        toInt64(ps.totaldamagetaken)
        - toInt64(ifNull(d.d_phys_taken, 0) + ifNull(d.d_magic_taken, 0) + ifNull(d.d_true_taken, 0)),
        0
    )) AS totaldamagetaken,
    toUInt32(greatest(
        toInt64(ps.physicaldamagetaken) - toInt64(ifNull(d.d_phys_taken, 0)), 0
    )) AS physicaldamagetaken,
    toUInt32(greatest(
        toInt64(ps.magicdamagetaken) - toInt64(ifNull(d.d_magic_taken, 0)), 0
    )) AS magicdamagetaken,
    toUInt32(greatest(
        toInt64(ps.truedamagetaken) - toInt64(ifNull(d.d_true_taken, 0)), 0
    )) AS truedamagetaken,
    toUInt32(greatest(
        toInt64(ps.goldspent) - toInt64(ifNull(d.d_goldspent, 0)), 0
    )) AS goldspent
)
FROM (
    SELECT * FROM game_data.participant_stats FINAL
) AS ps
SEMI JOIN game_data.participant_stats_corrected_long_game_ids USING (matchid)
LEFT JOIN game_data.participant_stats_corrected_delta_totals AS d
    ON ps.matchid = d.matchid AND ps.participantid = d.participantid;

TRUNCATE TABLE game_data.participant_stats_corrected;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 0;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 1;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 2;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 3;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 4;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 5;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 6;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 7;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 8;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 9;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 10;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 11;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 12;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 13;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 14;

INSERT INTO game_data.participant_stats_corrected
SELECT * FROM game_data.participant_stats_corrected_rows
WHERE cityHash64(`ps.matchid`) % 16 = 15;

DROP VIEW game_data.participant_stats_corrected_rows;
DROP TABLE game_data.participant_stats_corrected_long_game_ids;
DROP TABLE game_data.participant_stats_corrected_game_end_ts;
DROP TABLE game_data.participant_stats_corrected_padding_events;
DROP TABLE game_data.participant_stats_corrected_padding_purchased;
DROP TABLE game_data.participant_stats_corrected_padding_sold;
DROP TABLE game_data.participant_stats_corrected_padding_undo;
DROP TABLE game_data.participant_stats_corrected_deltas;
DROP TABLE game_data.participant_stats_corrected_delta_totals;
