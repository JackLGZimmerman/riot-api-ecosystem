-- noqa: disable=LT01,LT05,PRS
--
-- In-memory hash dictionary for canonical 1v1 cross-team matchup priors.
-- Source rows are stored canonicalised (left <= right); callers that want a
-- blue-perspective value must look up (b, r) when b <= r and otherwise look
-- up (r, b) and use (1 - left_win_rate).
--
-- LIFETIME(0): rebuild matchup_1v1 first, then run the build file to reload.
-- Auth: see ch_internal named collection (commands.md).

DROP DICTIONARY IF EXISTS game_data_filtered.matchup_1v1_dict;

CREATE DICTIONARY IF NOT EXISTS game_data_filtered.matchup_1v1_dict
(
    left_championid   Int32,
    left_teamposition String,
    left_build        String,
    right_championid  Int32,
    right_teamposition String,
    right_build       String,
    matchups          UInt64,
    left_win_rate     Float32
)
PRIMARY KEY left_championid, left_teamposition, left_build, right_championid, right_teamposition, right_build
SOURCE(CLICKHOUSE(
    NAME 'ch_internal'
    QUERY 'SELECT left_championid, left_teamposition, left_build, right_championid, right_teamposition, right_build, matchups, left_win_rate FROM game_data_filtered.matchup_1v1 WHERE split = ''train'''
))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0);
