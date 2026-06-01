-- noqa: disable=LT01,LT05,PRS
--
-- In-memory hash dictionary for the build-sibling 2vx synergy backoff (6024).
-- Source rows are canonicalised (slot 1 <= slot 2); win_rate is symmetric.

DROP DICTIONARY IF EXISTS game_data_filtered.synergy_2vx_build_group_dict;

CREATE DICTIONARY IF NOT EXISTS game_data_filtered.synergy_2vx_build_group_dict
(
    championid_1   Int32,
    teamposition_1 String,
    build_group_1  String,
    championid_2   Int32,
    teamposition_2 String,
    build_group_2  String,
    matchups       UInt64,
    win_rate       Float32
)
PRIMARY KEY championid_1, teamposition_1, build_group_1, championid_2, teamposition_2, build_group_2
SOURCE(CLICKHOUSE(
    NAME 'ch_internal'
    QUERY 'SELECT championid_1, teamposition_1, build_group_1, championid_2, teamposition_2, build_group_2, matchups, win_rate FROM game_data_filtered.synergy_2vx_build_group WHERE split = ''train'''
))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0);
