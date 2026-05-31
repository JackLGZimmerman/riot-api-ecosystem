-- noqa: disable=LT01,LT05,PRS
--
-- In-memory hash dictionary for the no-build 2vx synergy backoff (6022).
-- Source rows are canonicalised (slot 1 <= slot 2); win_rate is symmetric, so
-- callers can probe either ordering without inverting.
--
-- LIFETIME(0): rebuild synergy_2vx_nobuild first, then run the build file.
-- Auth: see ch_internal named collection (commands.md).

DROP DICTIONARY IF EXISTS game_data_filtered.synergy_2vx_nobuild_dict;

CREATE DICTIONARY IF NOT EXISTS game_data_filtered.synergy_2vx_nobuild_dict
(
    championid_1 Int32,
    teamposition_1 String,
    championid_2 Int32,
    teamposition_2 String,
    matchups UInt64,
    win_rate Float32
)
PRIMARY KEY championid_1, teamposition_1, championid_2, teamposition_2
SOURCE(CLICKHOUSE(
    NAME 'ch_internal'
    QUERY 'SELECT championid_1, teamposition_1, championid_2, teamposition_2, matchups, win_rate FROM game_data_filtered.synergy_2vx_nobuild WHERE split = ''train'''
))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0);
