-- noqa: disable=LT01,LT05,PRS
--
-- In-memory hash dictionary for the champion-pair 2vx synergy backoff (6023).
-- Source rows are canonicalised (id 1 <= id 2); win_rate is symmetric.
--
-- LIFETIME(0): rebuild synergy_2vx_champ first, then run the build file.
-- Auth: see ch_internal named collection (commands.md).

DROP DICTIONARY IF EXISTS game_data_filtered.synergy_2vx_champ_dict;

CREATE DICTIONARY IF NOT EXISTS game_data_filtered.synergy_2vx_champ_dict
(
    championid_1 Int32,
    championid_2 Int32,
    matchups UInt64,
    win_rate Float32
)
PRIMARY KEY championid_1, championid_2
SOURCE(CLICKHOUSE(
    NAME 'ch_internal'
    QUERY 'SELECT championid_1, championid_2, matchups, win_rate FROM game_data_filtered.synergy_2vx_champ WHERE split = ''train'''
))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0);
