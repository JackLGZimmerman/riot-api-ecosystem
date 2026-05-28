-- noqa: disable=AL05,LT01,LT02,LT05,RF02,PRS
--
-- Build phase-bucketed (champion, role, build, phase) priors for the
-- classification / embedding pipeline. Each source participant row is assigned
-- to its strongest temporal phase via participant_scaling_weights.max_value_bin.
--
-- Formulas:
--   sum(x) / count()                          — rates and largest* averages
--   60 * sum(x) / sum(timeplayed)             — per-minute volume metrics
--   avg(argMax(timeline_x, frame_timestamp))  — final timeline snapshots

SET enable_analyzer = 1;
SET max_threads = 2;
SET max_block_size = 8192;
SET max_insert_block_size = 32768;
SET max_bytes_before_external_group_by = 4294967296;
SET join_algorithm = 'full_sorting_merge';
SET join_use_nulls = 1;

DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_context;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_timeline_final;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_base;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_timeline_group;

CREATE TABLE game_data_filtered.synergy_1vx_temporal_context
(
    matchid String,
    participantid UInt8,
    championid Int32,
    teamposition LowCardinality(String),
    build LowCardinality(String),
    phase LowCardinality(String)
)
ENGINE = MergeTree
ORDER BY (matchid, participantid);

INSERT INTO game_data_filtered.synergy_1vx_temporal_context
SELECT
    psw.matchid,
    psw.participantid,
    assumeNotNull(psw.championid) AS championid,
    toString(psw.teamposition) AS teamposition,
    toString(ivt.highest_value_label) AS build,
    toString(psw.max_value_bin) AS phase
FROM game_data_filtered.participant_scaling_weights AS psw
INNER JOIN game_data_filtered.participant_item_value_totals AS ivt
    ON
        psw.matchid = ivt.matchid
        AND psw.participantid = ivt.participantid
WHERE psw.matchid IN (
    SELECT matchid
    FROM game_data_filtered.ml_game_split
    WHERE split = 'train'
);

CREATE TABLE game_data_filtered.synergy_1vx_temporal_timeline_final
(
    matchid String,
    participantid UInt8,
    healthmax Float32,
    lifesteal Float32,
    movementspeed Float32,
    omnivamp Float32,
    physicalvamp Float32,
    spellvamp Float32,
    armor Float32,
    magicresist Float32,
    abilitypower Float32,
    attackdamage Float32,
    attackspeed Float32
)
ENGINE = MergeTree
ORDER BY (matchid, participantid);

INSERT INTO game_data_filtered.synergy_1vx_temporal_timeline_final
SELECT
    matchid,
    participantid,
    toFloat32(tupleElement(final_stats, 1)) AS healthmax,
    toFloat32(tupleElement(final_stats, 2)) AS lifesteal,
    toFloat32(tupleElement(final_stats, 3)) AS movementspeed,
    toFloat32(tupleElement(final_stats, 4)) AS omnivamp,
    toFloat32(tupleElement(final_stats, 5)) AS physicalvamp,
    toFloat32(tupleElement(final_stats, 6)) AS spellvamp,
    toFloat32(tupleElement(final_stats, 7)) AS armor,
    toFloat32(tupleElement(final_stats, 8)) AS magicresist,
    toFloat32(tupleElement(final_stats, 9)) AS abilitypower,
    toFloat32(tupleElement(final_stats, 10)) AS attackdamage,
    toFloat32(tupleElement(final_stats, 11)) AS attackspeed
FROM (
    SELECT
        matchid,
        participantid,
        argMax(
            tuple(
                healthmax,
                lifesteal,
                movementspeed,
                omnivamp,
                physicalvamp,
                spellvamp,
                armor,
                magicresist,
                abilitypower,
                attackdamage,
                attackspeed
            ),
            frame_timestamp
        ) AS final_stats
    FROM game_data_filtered.tl_participant_stats
    WHERE matchid IN (
        SELECT matchid
        FROM game_data_filtered.ml_game_split
        WHERE split = 'train'
    )
    GROUP BY
        matchid,
        participantid
);

CREATE TABLE game_data_filtered.synergy_1vx_temporal_base
AS game_data_filtered.synergy_1vx_temporal
ENGINE = MergeTree
ORDER BY (split, championid, teamposition, build, phase);

INSERT INTO game_data_filtered.synergy_1vx_temporal_base
SELECT
    'train' AS split,
    ctx.championid AS championid,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(ctx.championid),
        ''
    ) AS championname,
    ctx.teamposition AS teamposition,
    ctx.build AS build,
    ctx.phase AS phase,
    toUInt32(count()) AS matchups,
    toFloat64(sum(ps.timeplayed)) AS sum_w_timeplayed,

    -- rates
    toFloat32(sum(ps.win) / count()) AS win,
    toFloat32(sum(ps.firstbloodkill) / count()) AS firstbloodkill,
    toFloat32(sum(ps.firstbloodassist) / count()) AS firstbloodassist,
    toFloat32(sum(ps.firsttowerkill) / count()) AS firsttowerkill,
    toFloat32(sum(ps.firsttowerassist) / count()) AS firsttowerassist,

    -- progression (per-minute)
    toFloat32(60 * sum(ps.champexperience) / sum(ps.timeplayed)) AS champexperience,

    -- combat (per-minute)
    toFloat32(60 * sum(ps.kills) / sum(ps.timeplayed)) AS kills,
    toFloat32(60 * sum(ps.deaths) / sum(ps.timeplayed)) AS deaths,
    toFloat32(60 * sum(ps.assists) / sum(ps.timeplayed)) AS assists,
    toFloat32(60 * sum(ps.doublekills) / sum(ps.timeplayed)) AS doublekills,
    toFloat32(60 * sum(ps.triplekills) / sum(ps.timeplayed)) AS triplekills,
    toFloat32(60 * sum(ps.killingsprees) / sum(ps.timeplayed)) AS killingsprees,

    -- combat: per-game maxima (weighted average, not per-minute)
    toFloat32(sum(ps.largestkillingspree) / count()) AS largestkillingspree,
    toFloat32(sum(ps.largestmultikill) / count()) AS largestmultikill,
    toFloat32(sum(ps.largestcriticalstrike) / count()) AS largestcriticalstrike,

    -- economy (per-minute)
    toFloat32(60 * sum(ps.goldearned) / sum(ps.timeplayed)) AS goldearned,

    -- damage dealt (per-minute)
    toFloat32(60 * sum(ps.totaldamagedealt) / sum(ps.timeplayed)) AS totaldamagedealt,
    toFloat32(60 * sum(ps.totaldamagedealttochampions) / sum(ps.timeplayed)) AS totaldamagedealttochampions,
    toFloat32(60 * sum(ps.physicaldamagedealt) / sum(ps.timeplayed)) AS physicaldamagedealt,
    toFloat32(60 * sum(ps.physicaldamagedealttochampions) / sum(ps.timeplayed)) AS physicaldamagedealttochampions,
    toFloat32(60 * sum(ps.magicdamagedealt) / sum(ps.timeplayed)) AS magicdamagedealt,
    toFloat32(60 * sum(ps.magicdamagedealttochampions) / sum(ps.timeplayed)) AS magicdamagedealttochampions,
    toFloat32(60 * sum(ps.truedamagedealt) / sum(ps.timeplayed)) AS truedamagedealt,
    toFloat32(60 * sum(ps.truedamagedealttochampions) / sum(ps.timeplayed)) AS truedamagedealttochampions,
    toFloat32(60 * sum(ps.damagedealttobuildings) / sum(ps.timeplayed)) AS damagedealttobuildings,
    toFloat32(60 * sum(ps.damagedealttoturrets) / sum(ps.timeplayed)) AS damagedealttoturrets,
    toFloat32(60 * sum(ps.damagedealttoobjectives) / sum(ps.timeplayed)) AS damagedealttoobjectives,
    toFloat32(60 * sum(coalesce(ps.damagedealttoepicmonsters, 0)) / sum(ps.timeplayed)) AS damagedealttoepicmonsters,

    -- damage taken / mitigated / healing / shielding (per-minute)
    toFloat32(60 * sum(ps.totaldamagetaken) / sum(ps.timeplayed)) AS totaldamagetaken,
    toFloat32(60 * sum(ps.physicaldamagetaken) / sum(ps.timeplayed)) AS physicaldamagetaken,
    toFloat32(60 * sum(ps.magicdamagetaken) / sum(ps.timeplayed)) AS magicdamagetaken,
    toFloat32(60 * sum(ps.truedamagetaken) / sum(ps.timeplayed)) AS truedamagetaken,
    toFloat32(60 * sum(ps.damageselfmitigated) / sum(ps.timeplayed)) AS damageselfmitigated,
    toFloat32(60 * sum(ps.totalheal) / sum(ps.timeplayed)) AS totalheal,
    toFloat32(60 * sum(ps.totalhealsonteammates) / sum(ps.timeplayed)) AS totalhealsonteammates,
    toFloat32(60 * sum(ps.totaldamageshieldedonteammates) / sum(ps.timeplayed)) AS totaldamageshieldedonteammates,

    -- final timeline snapshots (per-game averages)
    toFloat32(0) AS healthmax,
    toFloat32(0) AS lifesteal,
    toFloat32(0) AS movementspeed,
    toFloat32(0) AS omnivamp,
    toFloat32(0) AS physicalvamp,
    toFloat32(0) AS spellvamp,
    toFloat32(0) AS armor,
    toFloat32(0) AS magicresist,
    toFloat32(0) AS abilitypower,
    toFloat32(0) AS attackdamage,
    toFloat32(0) AS attackspeed,

    -- crowd control (per-minute)
    toFloat32(60 * sum(ps.timeccingothers) / sum(ps.timeplayed)) AS timeccingothers,
    toFloat32(60 * sum(ps.totaltimeccdealt) / sum(ps.timeplayed)) AS totaltimeccdealt,

    -- minions / monsters (per-minute)
    toFloat32(60 * sum(ps.totalminionskilled) / sum(ps.timeplayed)) AS totalminionskilled,
    toFloat32(60 * sum(ps.neutralminionskilled) / sum(ps.timeplayed)) AS neutralminionskilled,
    toFloat32(60 * sum(ps.totalallyjungleminionskilled) / sum(ps.timeplayed)) AS totalallyjungleminionskilled,
    toFloat32(60 * sum(ps.totalenemyjungleminionskilled) / sum(ps.timeplayed)) AS totalenemyjungleminionskilled,

    -- objectives / structures (per-minute)
    toFloat32(60 * sum(ps.baronkills) / sum(ps.timeplayed)) AS baronkills,
    toFloat32(60 * sum(ps.dragonkills) / sum(ps.timeplayed)) AS dragonkills,
    toFloat32(60 * sum(ps.inhibitorkills) / sum(ps.timeplayed)) AS inhibitorkills,
    toFloat32(60 * sum(ps.inhibitortakedowns) / sum(ps.timeplayed)) AS inhibitortakedowns,
    toFloat32(60 * sum(ps.inhibitorslost) / sum(ps.timeplayed)) AS inhibitorslost,
    toFloat32(60 * sum(ps.turretkills) / sum(ps.timeplayed)) AS turretkills,
    toFloat32(60 * sum(ps.turrettakedowns) / sum(ps.timeplayed)) AS turrettakedowns,
    toFloat32(60 * sum(ps.turretslost) / sum(ps.timeplayed)) AS turretslost,

    -- vision (per-minute)
    toFloat32(60 * sum(ps.visionscore) / sum(ps.timeplayed)) AS visionscore,
    toFloat32(60 * sum(ps.wardsplaced) / sum(ps.timeplayed)) AS wardsplaced,
    toFloat32(60 * sum(ps.wardskilled) / sum(ps.timeplayed)) AS wardskilled,
    toFloat32(60 * sum(ps.detectorwardsplaced) / sum(ps.timeplayed)) AS detectorwardsplaced,
    toFloat32(60 * sum(ps.visionwardsboughtingame) / sum(ps.timeplayed)) AS visionwardsboughtingame
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.synergy_1vx_temporal_context AS ctx
    ON
        ps.matchid = ctx.matchid
        AND ps.participantid = ctx.participantid
GROUP BY
    championid,
    teamposition,
    build,
    phase;

CREATE TABLE game_data_filtered.synergy_1vx_temporal_timeline_group
(
    split LowCardinality(String),
    championid Int32,
    teamposition LowCardinality(String),
    build LowCardinality(String),
    phase LowCardinality(String),
    healthmax Float32,
    lifesteal Float32,
    movementspeed Float32,
    omnivamp Float32,
    physicalvamp Float32,
    spellvamp Float32,
    armor Float32,
    magicresist Float32,
    abilitypower Float32,
    attackdamage Float32,
    attackspeed Float32
)
ENGINE = MergeTree
ORDER BY (split, championid, teamposition, build, phase);

INSERT INTO game_data_filtered.synergy_1vx_temporal_timeline_group
SELECT
    'train' AS split,
    ctx.championid AS championid,
    ctx.teamposition AS teamposition,
    ctx.build AS build,
    ctx.phase AS phase,
    toFloat32(coalesce(avg(tps.healthmax), 0)) AS healthmax,
    toFloat32(coalesce(avg(tps.lifesteal), 0)) AS lifesteal,
    toFloat32(coalesce(avg(tps.movementspeed), 0)) AS movementspeed,
    toFloat32(coalesce(avg(tps.omnivamp), 0)) AS omnivamp,
    toFloat32(coalesce(avg(tps.physicalvamp), 0)) AS physicalvamp,
    toFloat32(coalesce(avg(tps.spellvamp), 0)) AS spellvamp,
    toFloat32(coalesce(avg(tps.armor), 0)) AS armor,
    toFloat32(coalesce(avg(tps.magicresist), 0)) AS magicresist,
    toFloat32(coalesce(avg(tps.abilitypower), 0)) AS abilitypower,
    toFloat32(coalesce(avg(tps.attackdamage), 0)) AS attackdamage,
    toFloat32(coalesce(avg(tps.attackspeed), 0)) AS attackspeed
FROM game_data_filtered.synergy_1vx_temporal_context AS ctx
LEFT JOIN game_data_filtered.synergy_1vx_temporal_timeline_final AS tps
    ON
        ctx.matchid = tps.matchid
        AND ctx.participantid = tps.participantid
GROUP BY
    championid,
    teamposition,
    build,
    phase;

TRUNCATE TABLE game_data_filtered.synergy_1vx_temporal;

INSERT INTO game_data_filtered.synergy_1vx_temporal
SELECT b.* REPLACE (
    toFloat32(coalesce(tg.healthmax, b.healthmax)) AS healthmax,
    toFloat32(coalesce(tg.lifesteal, b.lifesteal)) AS lifesteal,
    toFloat32(coalesce(tg.movementspeed, b.movementspeed)) AS movementspeed,
    toFloat32(coalesce(tg.omnivamp, b.omnivamp)) AS omnivamp,
    toFloat32(coalesce(tg.physicalvamp, b.physicalvamp)) AS physicalvamp,
    toFloat32(coalesce(tg.spellvamp, b.spellvamp)) AS spellvamp,
    toFloat32(coalesce(tg.armor, b.armor)) AS armor,
    toFloat32(coalesce(tg.magicresist, b.magicresist)) AS magicresist,
    toFloat32(coalesce(tg.abilitypower, b.abilitypower)) AS abilitypower,
    toFloat32(coalesce(tg.attackdamage, b.attackdamage)) AS attackdamage,
    toFloat32(coalesce(tg.attackspeed, b.attackspeed)) AS attackspeed
)
FROM game_data_filtered.synergy_1vx_temporal_base AS b
LEFT JOIN game_data_filtered.synergy_1vx_temporal_timeline_group AS tg
    USING (split, championid, teamposition, build, phase);

DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_context;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_timeline_final;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_base;
DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_timeline_group;
