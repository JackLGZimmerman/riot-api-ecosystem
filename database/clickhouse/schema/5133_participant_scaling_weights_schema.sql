DROP TABLE IF EXISTS game_data_filtered.participant_scaling_weights;

CREATE TABLE game_data_filtered.participant_scaling_weights
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    puuid FixedString (78),
    championid Nullable (Int32),
    teamposition LowCardinality (String),
    gameduration UInt32,

    early_mid Float32,
    mid Float32,
    mid_late Float32,
    late Float32,

    max_value_bin LowCardinality (String)
)
ENGINE = MergeTree
ORDER BY (matchid, participantid, puuid);
