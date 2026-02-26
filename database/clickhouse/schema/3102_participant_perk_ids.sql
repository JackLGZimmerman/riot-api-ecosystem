CREATE TABLE IF NOT EXISTS game_data.participant_perk_ids
(
    run_id UUID,
    matchid UInt64,
    teamid UInt8,
    puuid FixedString (78),

    stat_defense UInt16,
    stat_flex UInt16,
    stat_offense UInt16,

    primary_style UInt16,
    sub_style UInt16,

    primary_perk_1 UInt16,
    primary_perk_2 UInt16,
    primary_perk_3 UInt16,
    primary_perk_4 UInt16,

    sub_perk_1 UInt16,
    sub_perk_2 UInt16,
    perk_combo_key UInt128
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, puuid, run_id);
