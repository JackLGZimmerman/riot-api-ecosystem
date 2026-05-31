-- noqa: disable=LT01,LT05,PRS
--
-- In-memory hash dictionary for the no-build 1v1 matchup backoff (6020).
-- Stored directionally (blue-perspective): probe (blue, red) and read
-- blue_win_rate straight, no inversion.
--
-- LIFETIME(0): rebuild matchup_1v1_nobuild first, then run the build file.
-- Auth: see ch_internal named collection (commands.md).

DROP DICTIONARY IF EXISTS game_data_filtered.matchup_1v1_nobuild_dict;

CREATE DICTIONARY IF NOT EXISTS game_data_filtered.matchup_1v1_nobuild_dict
(
    blue_championid Int32,
    blue_teamposition String,
    red_championid Int32,
    red_teamposition String,
    matchups UInt64,
    blue_win_rate Float32
)
PRIMARY KEY blue_championid, blue_teamposition, red_championid, red_teamposition
SOURCE(CLICKHOUSE(
    NAME 'ch_internal'
    QUERY 'SELECT blue_championid, blue_teamposition, red_championid, red_teamposition, matchups, blue_win_rate FROM game_data_filtered.matchup_1v1_nobuild WHERE split = ''train'''
))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0);
