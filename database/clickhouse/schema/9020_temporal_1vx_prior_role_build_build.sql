-- noqa: disable=AL05,LT01,LT02,LT05,RF02,PRS
--
-- PRIOR 3 (role + build_group): aggregate 6010 across all champions for
-- each (teamposition, build_group, phase). build_group collapses sibling
-- pairs (ap, ad, tank, utility); unpaired builds remain their own group.

TRUNCATE TABLE game_data_filtered.synergy_1vx_temporal_prior_role_build;

INSERT INTO game_data_filtered.synergy_1vx_temporal_prior_role_build
SELECT
    'train' AS split,
    t.teamposition,
    multiIf(
        t.build IN ('ability_power', 'ap_off_tank'),           'ap',
        t.build IN ('attack_damage', 'ad_off_tank'),           'ad',
        t.build IN ('ar_tank', 'mr_tank'),                     'tank',
        t.build IN ('utility_enchanter', 'utility_protection'),'utility',
        t.build
    ) AS build_group,
    t.phase,
    toUInt32(sum(t.matchups)) AS matchups,

    -- rates: sum(value * matchups) / sum(matchups)
    toFloat32(sum(t.win * t.matchups) / sum(t.matchups)) AS win,
    toFloat32(sum(t.firstbloodkill * t.matchups) / sum(t.matchups)) AS firstbloodkill,
    toFloat32(sum(t.firstbloodassist * t.matchups) / sum(t.matchups)) AS firstbloodassist,
    toFloat32(sum(t.firsttowerkill * t.matchups) / sum(t.matchups)) AS firsttowerkill,
    toFloat32(sum(t.firsttowerassist * t.matchups) / sum(t.matchups)) AS firsttowerassist,

    -- per-minute: sum(value * sum_w_timeplayed) / sum(sum_w_timeplayed)
    toFloat32(sum(t.champexperience * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS champexperience,
    toFloat32(sum(t.kills * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS kills,
    toFloat32(sum(t.deaths * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS deaths,
    toFloat32(sum(t.assists * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS assists,
    toFloat32(sum(t.doublekills * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS doublekills,
    toFloat32(sum(t.triplekills * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS triplekills,
    toFloat32(sum(t.killingsprees * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS killingsprees,

    -- largest_avg: sum(value * matchups) / sum(matchups)
    toFloat32(sum(t.largestkillingspree * t.matchups) / sum(t.matchups)) AS largestkillingspree,
    toFloat32(sum(t.largestmultikill * t.matchups) / sum(t.matchups)) AS largestmultikill,
    toFloat32(sum(t.largestcriticalstrike * t.matchups) / sum(t.matchups)) AS largestcriticalstrike,

    toFloat32(sum(t.goldearned * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS goldearned,
    toFloat32(sum(t.totaldamagedealt * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS totaldamagedealt,
    toFloat32(sum(t.totaldamagedealttochampions * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS totaldamagedealttochampions,
    toFloat32(sum(t.physicaldamagedealt * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS physicaldamagedealt,
    toFloat32(sum(t.physicaldamagedealttochampions * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS physicaldamagedealttochampions,
    toFloat32(sum(t.magicdamagedealt * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS magicdamagedealt,
    toFloat32(sum(t.magicdamagedealttochampions * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS magicdamagedealttochampions,
    toFloat32(sum(t.truedamagedealt * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS truedamagedealt,
    toFloat32(sum(t.truedamagedealttochampions * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS truedamagedealttochampions,
    toFloat32(sum(t.damagedealttobuildings * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS damagedealttobuildings,
    toFloat32(sum(t.damagedealttoturrets * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS damagedealttoturrets,
    toFloat32(sum(t.damagedealttoobjectives * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS damagedealttoobjectives,
    toFloat32(sum(t.damagedealttoepicmonsters * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS damagedealttoepicmonsters,
    toFloat32(sum(t.totaldamagetaken * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS totaldamagetaken,
    toFloat32(sum(t.physicaldamagetaken * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS physicaldamagetaken,
    toFloat32(sum(t.magicdamagetaken * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS magicdamagetaken,
    toFloat32(sum(t.truedamagetaken * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS truedamagetaken,
    toFloat32(sum(t.damageselfmitigated * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS damageselfmitigated,
    toFloat32(sum(t.totalheal * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS totalheal,
    toFloat32(sum(t.totalhealsonteammates * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS totalhealsonteammates,
    toFloat32(sum(t.totaldamageshieldedonteammates * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS totaldamageshieldedonteammates,
    -- final timeline snapshots: sum(value * matchups) / sum(matchups)
    toFloat32(sum(t.healthmax * t.matchups) / sum(t.matchups)) AS healthmax,
    toFloat32(sum(t.lifesteal * t.matchups) / sum(t.matchups)) AS lifesteal,
    toFloat32(sum(t.movementspeed * t.matchups) / sum(t.matchups)) AS movementspeed,
    toFloat32(sum(t.omnivamp * t.matchups) / sum(t.matchups)) AS omnivamp,
    toFloat32(sum(t.physicalvamp * t.matchups) / sum(t.matchups)) AS physicalvamp,
    toFloat32(sum(t.spellvamp * t.matchups) / sum(t.matchups)) AS spellvamp,
    toFloat32(sum(t.armor * t.matchups) / sum(t.matchups)) AS armor,
    toFloat32(sum(t.magicresist * t.matchups) / sum(t.matchups)) AS magicresist,
    toFloat32(sum(t.abilitypower * t.matchups) / sum(t.matchups)) AS abilitypower,
    toFloat32(sum(t.attackdamage * t.matchups) / sum(t.matchups)) AS attackdamage,
    toFloat32(sum(t.attackspeed * t.matchups) / sum(t.matchups)) AS attackspeed,
    toFloat32(sum(t.timeccingothers * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS timeccingothers,
    toFloat32(sum(t.totaltimeccdealt * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS totaltimeccdealt,
    toFloat32(sum(t.totalminionskilled * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS totalminionskilled,
    toFloat32(sum(t.neutralminionskilled * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS neutralminionskilled,
    toFloat32(sum(t.totalallyjungleminionskilled * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS totalallyjungleminionskilled,
    toFloat32(sum(t.totalenemyjungleminionskilled * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS totalenemyjungleminionskilled,
    toFloat32(sum(t.baronkills * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS baronkills,
    toFloat32(sum(t.dragonkills * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS dragonkills,
    toFloat32(sum(t.inhibitorkills * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS inhibitorkills,
    toFloat32(sum(t.inhibitortakedowns * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS inhibitortakedowns,
    toFloat32(sum(t.inhibitorslost * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS inhibitorslost,
    toFloat32(sum(t.turretkills * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS turretkills,
    toFloat32(sum(t.turrettakedowns * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS turrettakedowns,
    toFloat32(sum(t.turretslost * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS turretslost,
    toFloat32(sum(t.visionscore * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS visionscore,
    toFloat32(sum(t.wardsplaced * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS wardsplaced,
    toFloat32(sum(t.wardskilled * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS wardskilled,
    toFloat32(sum(t.detectorwardsplaced * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS detectorwardsplaced,
    toFloat32(sum(t.visionwardsboughtingame * t.sum_w_timeplayed) / sum(t.sum_w_timeplayed)) AS visionwardsboughtingame
FROM game_data_filtered.synergy_1vx_temporal AS t
WHERE t.split = 'train'
GROUP BY
    t.teamposition,
    build_group,
    t.phase;
