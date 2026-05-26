-- noqa: disable=LT01,LT05,PRS
--
-- Phase-bucketed champion/role/build embedding inputs.
--
-- One row per (championid, teamposition, build, phase). `matchups` is the
-- raw participant count for the phase selected by
-- participant_scaling_weights.max_value_bin.
--
-- Two aggregate formulas used:
--   * sum(x) / count()                         — rates ([0,1] booleans)
--                                                and per-game maxima
--                                                (largest*)
--   * 60 * sum(x) / sum(timeplayed)            — per-minute volume metrics
--
-- Columns removed for >95% zero rate in source data:
--   quadrakills (98.81%), pentakills (99.82%), unrealkills (99.9999%),
--   objectivesstolen (97.69%), objectivesstolenassists (98.97%),
--   sightwardsboughtingame (100%), visionclearedpings (100%)
--
-- Still-sparse columns retained (under threshold):
--   triplekills (~92% zero), baronkills (~90% zero)

DROP TABLE IF EXISTS game_data_filtered.synergy_1vx_temporal;

CREATE TABLE IF NOT EXISTS game_data_filtered.synergy_1vx_temporal
(
    split LowCardinality(String),
    championid Int32,
    championname LowCardinality(String),
    teamposition LowCardinality(String),
    build LowCardinality(String),
    phase LowCardinality(String),
    matchups UInt32,

    -- sum(timeplayed). Stored as Float64 to preserve precision when
    -- downstream priors re-aggregate per-minute metrics exactly via:
    --   prior_per_min = sum(value_c * sum_w_timeplayed_c) / sum(sum_w_timeplayed_c)
    sum_w_timeplayed Float64,

    -- rates (per-game event probabilities, [0, 1])
    win Float32,
    firstbloodkill Float32,
    firstbloodassist Float32,
    firsttowerkill Float32,
    firsttowerassist Float32,

    -- progression (per-minute)
    champexperience Float32,

    -- combat: counts per-minute; largest* as weighted averages
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
ORDER BY (split, championid, teamposition, build, phase);
