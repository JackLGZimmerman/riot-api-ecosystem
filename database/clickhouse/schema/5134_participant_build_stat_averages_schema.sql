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
DROP TABLE IF EXISTS game_data_filtered.participant_build_stat_averages;
DROP TABLE IF EXISTS game_data_filtered.participant_build_minute_averages;

CREATE TABLE game_data_filtered.participant_build_minute_averages
(
    championid Int32,
    championname LowCardinality (String),
    teamposition LowCardinality (String),
    build LowCardinality (String),
    participant_count UInt64,

    win Float32,
    firstbloodkill Float32,
    firstbloodassist Float32,
    firsttowerkill Float32,
    firsttowerassist Float32,

    champlevel Float32,
    champexperience Float32,
    kills Float32,
    deaths Float32,
    assists Float32,
    doublekills Float32,
    triplekills Float32,
    killingsprees Float32,
    largestkillingspree Float32,
    largestmultikill Float32,
    goldearned Float32,
    goldspent Float32,
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
    totaldamagetaken Float32,
    physicaldamagetaken Float32,
    magicdamagetaken Float32,
    truedamagetaken Float32,
    damageselfmitigated Float32,
    totalheal Float32,
    totalhealsonteammates Float32,
    totaldamageshieldedonteammates Float32,
    timeccingothers Float32,
    totaltimeccdealt Float32,
    totalminionskilled Float32,
    neutralminionskilled Float32,
    totalallyjungleminionskilled Float32,
    totalenemyjungleminionskilled Float32,
    baronkills Float32,
    dragonkills Float32,
    inhibitorkills Float32,
    inhibitortakedowns Float32,
    inhibitorslost Float32,
    turretkills Float32,
    turrettakedowns Float32,
    turretslost Float32,
    objectivesstolen Float32,
    objectivesstolenassists Float32,
    visionscore Float32,
    wardsplaced Float32,
    wardskilled Float32,
    detectorwardsplaced Float32,
    visionwardsboughtingame Float32,
    totaltimespentdead Float32,
    longesttimespentliving Float32,
    pings Float32,

    kda Float32,
    ka Float32,
    firstblood_participation Float32,
    totalprotectiononteammates Float32,
    expected_frontline_index Float32,
    expected_effective_durability Float32,
    expected_vision_denial_ratio Float32,
    expected_vision_action_score Float32,
    expected_epic_objective_score Float32,
    expected_structure_score Float32,
    expected_snowball_score Float32,
    expected_damage_per_gold Float32,
    expected_physical_damage_share Float32,
    expected_magic_damage_share Float32,
    expected_true_damage_share Float32,
    damage_to_taken_ratio Float32,
    totalcs Float32
)
ENGINE = MergeTree
ORDER BY (championid, teamposition, build);
