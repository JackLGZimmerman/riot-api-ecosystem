-- noqa: disable=LT01,LT05,PRS
--
-- PRIOR 3 (role + build_group): any champion, same role, similar build.
--
-- Keyed by (split, teamposition, build_group, phase). Aggregates the source
-- across all champions for a given role/build_group. Captures lane-economy
-- shape (CS, gold, XP, vision) which dominates per-minute volume metrics.
--
-- build_group values:
--   paired: 'ap', 'ad', 'tank', 'utility'
--           (ap        = ability_power + ap_off_tank,
--            ad        = attack_damage + ad_off_tank,
--            tank      = ar_tank + mr_tank,
--            utility   = utility_enchanter + utility_protection)
--   singleton: any build with no sibling uses its own build name
--
-- Utility isolation: teamposition is a key, so 'utility' build_group only
-- aggregates from Utility-role rows.

DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_prior_role_build;

CREATE TABLE IF NOT EXISTS game_data_filtered.synergy_1vx_temporal_prior_role_build
(
    split LowCardinality(String),
    teamposition LowCardinality(String),
    build_group LowCardinality(String),
    phase LowCardinality(String),
    matchups UInt32,

    -- rates
    win Float32,
    firstbloodkill Float32,
    firstbloodassist Float32,
    firsttowerkill Float32,
    firsttowerassist Float32,

    -- progression (per-minute)
    champexperience Float32,

    -- combat
    kills Float32,
    deaths Float32,
    assists Float32,
    doublekills Float32,
    triplekills Float32,
    killingsprees Float32,
    largestkillingspree Float32,
    largestmultikill Float32,
    largestcriticalstrike Float32,

    -- economy (per-minute)
    goldearned Float32,

    -- damage dealt (per-minute)
    totaldamagedealt Float32,
    totaldamagedealttochampions Float32,
    physicaldamagedealt Float32,
    physicaldamagedealttochampions Float32,
    magicdamagedealt Float32,
    magicdamagedealttochampions Float32,
    truedamagedealt Float32,
    truedamagedealttochampions Float32,
    damagedealttobuildings Float32,
    damagedealttoturrets Float32,
    damagedealttoobjectives Float32,
    damagedealttoepicmonsters Float32,

    -- damage taken / mitigated / healing / shielding (per-minute)
    totaldamagetaken Float32,
    physicaldamagetaken Float32,
    magicdamagetaken Float32,
    truedamagetaken Float32,
    damageselfmitigated Float32,
    totalheal Float32,
    totalhealsonteammates Float32,
    totaldamageshieldedonteammates Float32,

    -- final timeline snapshots (per-game averages)
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
    attackspeed Float32,

    -- crowd control (per-minute)
    timeccingothers Float32,
    totaltimeccdealt Float32,

    -- minions / monsters (per-minute)
    totalminionskilled Float32,
    neutralminionskilled Float32,
    totalallyjungleminionskilled Float32,
    totalenemyjungleminionskilled Float32,

    -- objectives / structures (per-minute)
    baronkills Float32,
    dragonkills Float32,
    inhibitorkills Float32,
    inhibitortakedowns Float32,
    inhibitorslost Float32,
    turretkills Float32,
    turrettakedowns Float32,
    turretslost Float32,

    -- vision (per-minute)
    visionscore Float32,
    wardsplaced Float32,
    wardskilled Float32,
    detectorwardsplaced Float32,
    visionwardsboughtingame Float32
)
ENGINE = MergeTree
ORDER BY (split, teamposition, build_group, phase);
