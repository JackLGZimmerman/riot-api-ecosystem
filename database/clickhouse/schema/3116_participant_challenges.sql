CREATE TABLE IF NOT EXISTS game_data.participant_challenges
(
    run_id UUID,
    matchid UInt64,
    teamid UInt8,
    puuid FixedString (78),
    payload JSON
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, puuid, run_id);
