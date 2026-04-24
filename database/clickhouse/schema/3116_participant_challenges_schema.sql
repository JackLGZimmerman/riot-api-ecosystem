CREATE TABLE IF NOT EXISTS game_data.participant_challenges
(
    run_id UUID,
    matchid String,
    teamid UInt8,
    puuid FixedString (78),
    payload Map (String, Float32)
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, puuid, run_id);
