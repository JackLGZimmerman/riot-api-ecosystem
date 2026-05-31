-- noqa: disable=LT01,LT05,PRS
--
-- In-memory hash dictionary for the champion-pair 1v1 matchup backoff (6021).
-- Stored directionally (blue-perspective): probe (blue, red) and read
-- blue_win_rate straight, no inversion.
--
-- LIFETIME(0): rebuild matchup_1v1_champ first, then run the build file.
-- Auth: see ch_internal named collection (commands.md).

DROP DICTIONARY IF EXISTS game_data_filtered.matchup_1v1_champ_dict;

CREATE DICTIONARY IF NOT EXISTS game_data_filtered.matchup_1v1_champ_dict
(
    blue_championid Int32,
    red_championid Int32,
    matchups UInt64,
    blue_win_rate Float32
)
PRIMARY KEY blue_championid, red_championid
SOURCE(CLICKHOUSE(
    NAME 'ch_internal'
    QUERY 'SELECT blue_championid, red_championid, matchups, blue_win_rate FROM game_data_filtered.matchup_1v1_champ WHERE split = ''train'''
))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0);
