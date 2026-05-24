-- noqa: disable=LT01,LT05,PRS
--
-- In-memory hash dictionary for canonical same-team 2-player synergy priors.
-- Source rows are canonicalised (slot 1 <= slot 2); win_rate is symmetric, so
-- callers can probe either ordering without inverting.
--
-- LIFETIME(0): rebuild synergy_2vx first, then run the build file to reload.
-- Auth: see ch_internal named collection (commands.md).

DROP DICTIONARY IF EXISTS game_data_filtered.synergy_2vx_dict;

CREATE DICTIONARY IF NOT EXISTS game_data_filtered.synergy_2vx_dict
(
    championid_1   Int32,
    teamposition_1 String,
    build_1        String,
    championid_2   Int32,
    teamposition_2 String,
    build_2        String,
    matchups       UInt64,
    win_rate       Float32
)
PRIMARY KEY championid_1, teamposition_1, build_1, championid_2, teamposition_2, build_2
SOURCE(CLICKHOUSE(
    NAME 'ch_internal'
    QUERY 'SELECT championid_1, teamposition_1, build_1, championid_2, teamposition_2, build_2, matchups, win_rate FROM game_data_filtered.synergy_2vx WHERE split = ''train'''
))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0);
