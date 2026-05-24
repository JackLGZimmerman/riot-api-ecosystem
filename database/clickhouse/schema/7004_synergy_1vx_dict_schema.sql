-- noqa: disable=LT01,LT05,PRS
--
-- In-memory hash dictionary for draft-time champion/role/build priors.
--
-- Keyed on (championid, teamposition, build) so dictGetOrDefault can look up
-- each player tuple directly inside arrayMap, eliminating the ARRAY JOIN fan-out,
-- ANY LEFT JOIN, GROUP BY, and arraySort that a table join requires.
--
-- LIFETIME(0): data is fixed at create/reload time; rebuild synergy_1vx first,
-- then run the build file to reload.
--
-- Auth: SOURCE(CLICKHOUSE(...)) references the `ch_internal` named collection
-- (see commands.md "Named collection for dictionary reloads"); the `default`
-- user no longer exists, so credentials must come from the collection.

DROP DICTIONARY IF EXISTS game_data_filtered.synergy_1vx_dict;

CREATE DICTIONARY IF NOT EXISTS game_data_filtered.synergy_1vx_dict
(
    championid  Int32,
    teamposition String,
    build        String,
    matchups     UInt32,
    win_rate     Float32
)
PRIMARY KEY championid, teamposition, build
SOURCE(CLICKHOUSE(
    NAME 'ch_internal'
    QUERY 'SELECT championid, teamposition, build, matchups, win_rate FROM game_data_filtered.synergy_1vx WHERE split = ''train'''
))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0);
