-- noqa: disable=AL05,LT01,LT02,LT05,RF02,PRS
--
-- Build phase-bucketed (champion, role, build, phase) priors for the
-- classification / embedding pipeline. Each source participant row is assigned
-- to its strongest temporal phase via participant_scaling_weights.max_value_bin.
--
-- Formulas:
--   sum(x) / count()                          — rates and largest* averages
--   60 * sum(x) / sum(timeplayed)             — per-minute volume metrics

TRUNCATE TABLE game_data_filtered.synergy_1vx_temporal;

INSERT INTO game_data_filtered.synergy_1vx_temporal
SELECT
    'train' AS split,
    assumeNotNull(ps.championid) AS championid,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(assumeNotNull(ps.championid)),
        ''
    ) AS championname,
    toString(ps.teamposition) AS teamposition,
    toString(ivt.highest_value_label) AS build,
    toString(psw.max_value_bin) AS phase,
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
INNER JOIN game_data_filtered.ml_game_split AS s
    ON ps.matchid = s.matchid
INNER JOIN game_data_filtered.participant_item_value_totals AS ivt
    ON
        ps.matchid = ivt.matchid
        AND ps.participantid = ivt.participantid
INNER JOIN game_data_filtered.participant_scaling_weights AS psw
    ON
        ps.matchid = psw.matchid
        AND ps.participantid = psw.participantid
WHERE s.split = 'train'
GROUP BY
    championid,
    teamposition,
    build,
    phase;
