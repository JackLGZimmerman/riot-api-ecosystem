-- noqa: disable=LT01,LT05,PRS
--
-- PRIOR 4 (champion + build_group): same champion, any role, similar build.
--
-- Keyed by (split, championid, build_group, phase). Aggregates the source
-- across all roles for a given champion/build_group. Preserves champion
-- identity (kit, scalings, base stats) when role-level signal is unavailable.
--
-- build_group values: see 9020_temporal_1vx_prior_role_build_schema.sql.
--
-- Utility isolation: utility_* builds only appear in Utility-role source
-- rows, so a 'utility' build_group naturally only aggregates Utility data.
-- Non-utility build_groups never include Utility-role rows because those
-- build labels do not appear there.

DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal_prior_champion_build;

CREATE TABLE IF NOT EXISTS game_data_filtered.synergy_1vx_temporal_prior_champion_build
(
    split LowCardinality(String),
    championid Int32,
    championname LowCardinality(String),
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
ORDER BY (split, championid, build_group, phase);
