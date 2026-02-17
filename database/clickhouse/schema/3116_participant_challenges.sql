CREATE TABLE IF NOT EXISTS game_data.participant_challenges
(
    run_id UUID,
    gameid UInt64,
    teamid UInt8,
    puuid FixedString (78)
)
ENGINE = MergeTree
ORDER BY (gameid, teamid, puuid, run_id);
