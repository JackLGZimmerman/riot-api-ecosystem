CREATE TABLE IF NOT EXISTS game_data.participant_perk_values
(
    run_id UUID,
    matchid UInt64,
    teamid UInt8,
    puuid FixedString (78),

    primary_var1_1 Int32,
    primary_var2_1 Int32,
    primary_var3_1 Int32,
    primary_var1_2 Int32,
    primary_var2_2 Int32,
    primary_var3_2 Int32,
    primary_var1_3 Int32,
    primary_var2_3 Int32,
    primary_var3_3 Int32,
    primary_var1_4 Int32,
    primary_var2_4 Int32,
    primary_var3_4 Int32,

    sub_var1_1 Int32,
    sub_var2_1 Int32,
    sub_var3_1 Int32,
    sub_var1_2 Int32,
    sub_var2_2 Int32,
    sub_var3_2 Int32
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, puuid, run_id);
