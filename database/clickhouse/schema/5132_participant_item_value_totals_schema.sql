DROP TABLE IF EXISTS game_data_filtered.participant_item_value_totals;

CREATE TABLE game_data_filtered.participant_item_value_totals
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    puuid FixedString (78),
    championid Nullable (Int32),
    teamposition LowCardinality (String),

    attack_damage Float32,
    ability_power Float32,
    lethality Float32,
    on_hit Float32,
    crit Float32,
    utility_enchanter Float32,
    utility_protection Float32,
    ar_tank Float32,
    mr_tank Float32,
    ad_off_tank Float32,
    ap_off_tank Float32,

    highest_value Float32,
    highest_value_label LowCardinality (String)
)
ENGINE = MergeTree
ORDER BY (matchid, participantid, puuid);
