-- noqa: disable=LT05
-- Removed columns:
--   gameendedinearlysurrender Float32
--   gameendedinignbsurrender Float32
--   gameendedinsurrender Float32
--   teamearlysurrendered Float32
--   teamignbsurrendered Float32
--   largestcriticalstrike_per_minute Float32
--   consumablespurchased_per_minute Float32
--   itemspurchased_per_minute Float32
--   attack_damage_per_minute Float32
--   ability_power_per_minute Float32
--   lethality_per_minute Float32
--   on_hit_per_minute Float32
--   crit_per_minute Float32
--   utility_enchanter_per_minute Float32
--   utility_protection_per_minute Float32
--   ar_tank_per_minute Float32
--   mr_tank_per_minute Float32
--   ad_off_tank_per_minute Float32
--   ap_off_tank_per_minute Float32
--   highest_value_per_minute Float32
--   summoner1casts Float32
--   summoner2casts Float32
--   spell1casts Float32
--   spell2casts Float32
--   spell3casts Float32
--   spell4casts Float32
--   sightwardsboughtingame Float32
--   visionclearedpings Float32
--   unrealkills Float32
--   quadrakills Float32
--   pentakills Float32
--   totalunitshealed Float32
--   bountylevel Float32
TRUNCATE TABLE game_data_filtered.participant_build_minute_averages;

INSERT INTO game_data_filtered.participant_build_minute_averages
(
    championid,
    championname,
    teamposition,
    build,
    participant_count,
    win,
    firstbloodkill,
    firstbloodassist,
    firsttowerkill,
    firsttowerassist,
    champlevel,
    champexperience,
    kills,
    deaths,
    assists,
    doublekills,
    triplekills,
    killingsprees,
    largestkillingspree,
    largestmultikill,
    goldearned,
    goldspent,
    totaldamagedealt,
    totaldamagedealttochampions,
    physicaldamagedealt,
    physicaldamagedealttochampions,
    magicdamagedealt,
    magicdamagedealttochampions,
    truedamagedealt,
    truedamagedealttochampions,
    damagedealttobuildings,
    damagedealttoturrets,
    damagedealttoobjectives,
    damagedealttoepicmonsters,
    totaldamagetaken,
    physicaldamagetaken,
    magicdamagetaken,
    truedamagetaken,
    damageselfmitigated,
    totalheal,
    totalhealsonteammates,
    totaldamageshieldedonteammates,
    timeccingothers,
    totaltimeccdealt,
    totalminionskilled,
    neutralminionskilled,
    totalallyjungleminionskilled,
    totalenemyjungleminionskilled,
    baronkills,
    dragonkills,
    inhibitorkills,
    inhibitortakedowns,
    inhibitorslost,
    turretkills,
    turrettakedowns,
    turretslost,
    objectivesstolen,
    objectivesstolenassists,
    visionscore,
    wardsplaced,
    wardskilled,
    detectorwardsplaced,
    visionwardsboughtingame,
    totaltimespentdead,
    longesttimespentliving,
    pings,

    kda,
    ka,
    firstblood_participation,
    totalprotectiononteammates,
    expected_frontline_index,
    expected_effective_durability,
    expected_vision_denial_ratio,
    expected_vision_action_score,
    expected_epic_objective_score,
    expected_structure_score,
    expected_snowball_score,
    expected_damage_per_gold,
    expected_physical_damage_share,
    expected_magic_damage_share,
    expected_true_damage_share,
    damage_to_taken_ratio,
    totalcs
)
WITH source AS (
    SELECT
        assumeNotNull(ps.championid) AS championid,
        dictGetOrDefault(
            'game_data.championid_name_map_dict',
            'name',
            toString(assumeNotNull(ps.championid)),
            ''
        ) AS championname,
        toString(ps.teamposition) AS teamposition,
        ivt.highest_value_label AS build,
        toFloat32(ps.timeplayed) AS timeplayed,

        ps.win,
        ps.firstbloodkill,
        ps.firstbloodassist,
        ps.firsttowerkill,
        ps.firsttowerassist,

        ps.champlevel,
        ps.champexperience,
        ps.kills,
        ps.deaths,
        ps.assists,
        ps.doublekills,
        ps.triplekills,
        ps.killingsprees,
        ps.largestkillingspree,
        ps.largestmultikill,
        ps.goldearned,
        ps.goldspent,
        ps.totaldamagedealt,
        ps.totaldamagedealttochampions,
        ps.physicaldamagedealt,
        ps.physicaldamagedealttochampions,
        ps.magicdamagedealt,
        ps.magicdamagedealttochampions,
        ps.truedamagedealt,
        ps.truedamagedealttochampions,
        ps.damagedealttobuildings,
        ps.damagedealttoturrets,
        ps.damagedealttoobjectives,
        ps.damagedealttoepicmonsters,
        ps.totaldamagetaken,
        ps.physicaldamagetaken,
        ps.magicdamagetaken,
        ps.truedamagetaken,
        ps.damageselfmitigated,
        ps.totalheal,
        ps.totalhealsonteammates,
        ps.totaldamageshieldedonteammates,
        ps.timeccingothers,
        ps.totaltimeccdealt,
        ps.totalminionskilled,
        ps.neutralminionskilled,
        ps.totalallyjungleminionskilled,
        ps.totalenemyjungleminionskilled,
        ps.baronkills,
        ps.dragonkills,
        ps.inhibitorkills,
        ps.inhibitortakedowns,
        ps.inhibitorslost,
        ps.turretkills,
        ps.turrettakedowns,
        ps.turretslost,
        ps.objectivesstolen,
        ps.objectivesstolenassists,
        ps.visionscore,
        ps.wardsplaced,
        ps.wardskilled,
        ps.detectorwardsplaced,
        ps.visionwardsboughtingame,
        ps.totaltimespentdead,
        ps.longesttimespentliving,
        (
            toUInt32(ps.allinpings)
            + toUInt32(ps.assistmepings)
            + toUInt32(ps.basicpings)
            + toUInt32(ps.commandpings)
            + toUInt32(ps.dangerpings)
            + toUInt32(ps.enemymissingpings)
            + toUInt32(ps.enemyvisionpings)
            + toUInt32(ps.getbackpings)
            + toUInt32(ps.holdpings)
            + toUInt32(ps.needvisionpings)
            + toUInt32(ps.onmywaypings)
            + toUInt32(ps.pushpings)
            + toUInt32(coalesce(ps.retreatpings, 0))
        ) AS pings
    FROM game_data_filtered.participant_stats AS ps
    ANY INNER JOIN game_data_filtered.participant_item_value_totals AS ivt
        ON
            ps.matchid = ivt.matchid
            AND ps.participantid = ivt.participantid
    WHERE
        ps.championid IS NOT NULL
        AND ps.timeplayed > 0
),

aggregated AS (
    SELECT
        championid,
        championname,
        teamposition,
        build,
        count() AS participant_count,
        avg(toFloat32(win)) AS win,
        avg(toFloat32(firstbloodkill)) AS firstbloodkill,
        avg(toFloat32(firstbloodassist)) AS firstbloodassist,
        avg(toFloat32(firsttowerkill)) AS firsttowerkill,
        avg(toFloat32(firsttowerassist)) AS firsttowerassist,

        avg(toFloat32(champlevel)) AS champlevel,
        avg(toFloat32(champexperience) * toFloat32(60) / toFloat32(timeplayed))
            AS champexperience,
        avg(toFloat32(kills) * toFloat32(60) / toFloat32(timeplayed)) AS kills,
        avg(toFloat32(deaths) * toFloat32(60) / toFloat32(timeplayed)) AS deaths,
        avg(toFloat32(assists) * toFloat32(60) / toFloat32(timeplayed)) AS assists,
        avg(toFloat32(doublekills) * toFloat32(60) / toFloat32(timeplayed)) AS doublekills,
        avg(toFloat32(triplekills) * toFloat32(60) / toFloat32(timeplayed)) AS triplekills,
        avg(toFloat32(killingsprees) * toFloat32(60) / toFloat32(timeplayed)) AS killingsprees,
        avg(toFloat32(largestkillingspree)) AS largestkillingspree,
        avg(toFloat32(largestmultikill)) AS largestmultikill,
        avg(toFloat32(goldearned) * toFloat32(60) / toFloat32(timeplayed)) AS goldearned,
        avg(toFloat32(goldspent) * toFloat32(60) / toFloat32(timeplayed)) AS goldspent,
        avg(toFloat32(totaldamagedealt) * toFloat32(60) / toFloat32(timeplayed))
            AS totaldamagedealt,
        avg(toFloat32(totaldamagedealttochampions) * toFloat32(60) / toFloat32(timeplayed))
            AS totaldamagedealttochampions,
        avg(toFloat32(physicaldamagedealt) * toFloat32(60) / toFloat32(timeplayed))
            AS physicaldamagedealt,
        avg(toFloat32(physicaldamagedealttochampions) * toFloat32(60) / toFloat32(timeplayed))
            AS physicaldamagedealttochampions,
        avg(toFloat32(magicdamagedealt) * toFloat32(60) / toFloat32(timeplayed))
            AS magicdamagedealt,
        avg(toFloat32(magicdamagedealttochampions) * toFloat32(60) / toFloat32(timeplayed))
            AS magicdamagedealttochampions,
        avg(toFloat32(truedamagedealt) * toFloat32(60) / toFloat32(timeplayed))
            AS truedamagedealt,
        avg(toFloat32(truedamagedealttochampions) * toFloat32(60) / toFloat32(timeplayed))
            AS truedamagedealttochampions,
        avg(toFloat32(damagedealttobuildings) * toFloat32(60) / toFloat32(timeplayed))
            AS damagedealttobuildings,
        avg(toFloat32(damagedealttoturrets) * toFloat32(60) / toFloat32(timeplayed))
            AS damagedealttoturrets,
        avg(toFloat32(damagedealttoobjectives) * toFloat32(60) / toFloat32(timeplayed))
            AS damagedealttoobjectives,
        avg(toFloat32(coalesce(damagedealttoepicmonsters, 0)) * toFloat32(60) / toFloat32(timeplayed))
            AS damagedealttoepicmonsters,
        avg(toFloat32(totaldamagetaken) * toFloat32(60) / toFloat32(timeplayed))
            AS totaldamagetaken,
        avg(toFloat32(physicaldamagetaken) * toFloat32(60) / toFloat32(timeplayed))
            AS physicaldamagetaken,
        avg(toFloat32(magicdamagetaken) * toFloat32(60) / toFloat32(timeplayed))
            AS magicdamagetaken,
        avg(toFloat32(truedamagetaken) * toFloat32(60) / toFloat32(timeplayed))
            AS truedamagetaken,
        avg(toFloat32(damageselfmitigated) * toFloat32(60) / toFloat32(timeplayed))
            AS damageselfmitigated,
        avg(toFloat32(totalheal) * toFloat32(60) / toFloat32(timeplayed)) AS totalheal,
        avg(toFloat32(totalhealsonteammates) * toFloat32(60) / toFloat32(timeplayed))
            AS totalhealsonteammates,
        avg(toFloat32(totaldamageshieldedonteammates) * toFloat32(60) / toFloat32(timeplayed))
            AS totaldamageshieldedonteammates,
        avg(toFloat32(timeccingothers) * toFloat32(60) / toFloat32(timeplayed))
            AS timeccingothers,
        avg(toFloat32(totaltimeccdealt) * toFloat32(60) / toFloat32(timeplayed))
            AS totaltimeccdealt,
        avg(toFloat32(totalminionskilled) * toFloat32(60) / toFloat32(timeplayed))
            AS totalminionskilled,
        avg(toFloat32(neutralminionskilled) * toFloat32(60) / toFloat32(timeplayed))
            AS neutralminionskilled,
        avg(toFloat32(totalallyjungleminionskilled) * toFloat32(60) / toFloat32(timeplayed))
            AS totalallyjungleminionskilled,
        avg(toFloat32(totalenemyjungleminionskilled) * toFloat32(60) / toFloat32(timeplayed))
            AS totalenemyjungleminionskilled,
        avg(toFloat32(baronkills) * toFloat32(60) / toFloat32(timeplayed)) AS baronkills,
        avg(toFloat32(dragonkills) * toFloat32(60) / toFloat32(timeplayed)) AS dragonkills,
        avg(toFloat32(inhibitorkills)) AS inhibitorkills,
        avg(toFloat32(inhibitortakedowns)) AS inhibitortakedowns,
        avg(toFloat32(inhibitorslost)) AS inhibitorslost,
        avg(toFloat32(turretkills)) AS turretkills,
        avg(toFloat32(turrettakedowns)) AS turrettakedowns,
        avg(toFloat32(turretslost)) AS turretslost,
        avg(toFloat32(objectivesstolen) * toFloat32(60) / toFloat32(timeplayed))
            AS objectivesstolen,
        avg(toFloat32(objectivesstolenassists) * toFloat32(60) / toFloat32(timeplayed))
            AS objectivesstolenassists,
        avg(toFloat32(visionscore) * toFloat32(60) / toFloat32(timeplayed)) AS visionscore,
        avg(toFloat32(wardsplaced) * toFloat32(60) / toFloat32(timeplayed)) AS wardsplaced,
        avg(toFloat32(wardskilled) * toFloat32(60) / toFloat32(timeplayed)) AS wardskilled,
        avg(toFloat32(detectorwardsplaced) * toFloat32(60) / toFloat32(timeplayed))
            AS detectorwardsplaced,
        avg(toFloat32(visionwardsboughtingame) * toFloat32(60) / toFloat32(timeplayed))
            AS visionwardsboughtingame,
        avg(toFloat32(totaltimespentdead) * toFloat32(60) / toFloat32(timeplayed))
            AS totaltimespentdead,
        avg(toFloat32(longesttimespentliving)) AS longesttimespentliving,
        avg(toFloat32(pings) * toFloat32(60) / toFloat32(timeplayed)) AS pings
    FROM source
    GROUP BY
        championid,
        championname,
        teamposition,
        build
)

SELECT
    *,
    (kills + assists) / greatest(deaths, toFloat32(0.001)) AS kda,
    kills + assists AS ka,
    firstbloodkill + firstbloodassist AS firstblood_participation,
    totalhealsonteammates + totaldamageshieldedonteammates AS totalprotectiononteammates,
    (totaldamagetaken + damageselfmitigated)
    / greatest(totaldamagedealttochampions, toFloat32(1)) AS expected_frontline_index,
    totaldamagetaken + damageselfmitigated + totalheal AS expected_effective_durability,
    wardskilled
    / greatest(wardsplaced + wardskilled, toFloat32(0.001)) AS expected_vision_denial_ratio,
    wardsplaced
    + toFloat32(1.5) * wardskilled
    + toFloat32(2) * detectorwardsplaced
    + toFloat32(2) * visionwardsboughtingame AS expected_vision_action_score,
    dragonkills
    + toFloat32(2) * baronkills
    + toFloat32(2) * objectivesstolen
    + objectivesstolenassists AS expected_epic_objective_score,
    turretkills
    + turrettakedowns
    + toFloat32(2) * inhibitorkills
    + toFloat32(2) * inhibitortakedowns AS expected_structure_score,
    doublekills
    + toFloat32(2) * triplekills
    + killingsprees
    + toFloat32(0.5) * largestkillingspree AS expected_snowball_score,
    totaldamagedealttochampions
    / greatest(goldearned, toFloat32(1)) AS expected_damage_per_gold,
    physicaldamagedealttochampions
    / greatest(totaldamagedealttochampions, toFloat32(1)) AS expected_physical_damage_share,
    magicdamagedealttochampions
    / greatest(totaldamagedealttochampions, toFloat32(1)) AS expected_magic_damage_share,
    truedamagedealttochampions
    / greatest(totaldamagedealttochampions, toFloat32(1)) AS expected_true_damage_share,
    totaldamagedealttochampions
    / greatest(totaldamagetaken, toFloat32(1)) AS damage_to_taken_ratio,
    totalminionskilled + neutralminionskilled AS totalcs
FROM aggregated;
