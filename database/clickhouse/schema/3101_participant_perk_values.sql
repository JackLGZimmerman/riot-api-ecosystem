CREATE TABLE IF NOT EXISTS game_data.participant_perk_values
(
    run_id UUID,
    gameid UInt64,
    teamid UInt8,
    puuid FixedString (78),

    primary_var1_1 UInt16,
    primary_var2_1 UInt16,
    primary_var3_1 UInt16,
    primary_var1_2 UInt16,
    primary_var2_2 UInt16,
    primary_var3_2 UInt16,
    primary_var1_3 UInt16,
    primary_var2_3 UInt16,
    primary_var3_3 UInt16,
    primary_var1_4 UInt16,
    primary_var2_4 UInt16,
    primary_var3_4 UInt16,

    sub_var1_1 UInt16,
    sub_var2_1 UInt16,
    sub_var3_1 UInt16,
    sub_var1_2 UInt16,
    sub_var2_2 UInt16,
    sub_var3_2 UInt16
)
ENGINE = MergeTree
ORDER BY (gameid, teamid, puuid, run_id);
