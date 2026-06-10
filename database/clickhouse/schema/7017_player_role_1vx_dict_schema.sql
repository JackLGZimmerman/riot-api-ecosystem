-- noqa: disable=LT01,LT05,PRS
--
-- In-memory hash dictionary for draft-time per-(player, role) experience,
-- keyed on (puuid, teamposition) so dictGetOrDefault can resolve each pivot
-- slot directly inside arrayMap.
--
-- LIFETIME(0): data is fixed at create/reload time; rebuild player_role_1vx
-- first, then run the build file to reload.
--
-- Auth: SOURCE(CLICKHOUSE(...)) references the `ch_internal` named collection
-- (see commands.md "Named collection for dictionary reloads").

DROP DICTIONARY IF EXISTS game_data_filtered.player_role_1vx_dict;

CREATE DICTIONARY IF NOT EXISTS game_data_filtered.player_role_1vx_dict
(
    puuid        String,
    teamposition String,
    matchups     UInt32
)
PRIMARY KEY puuid, teamposition
SOURCE(CLICKHOUSE(
    NAME 'ch_internal'
    QUERY 'SELECT puuid, teamposition, matchups FROM game_data_filtered.player_role_1vx WHERE split = ''train'''
))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0);
