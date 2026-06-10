-- noqa: disable=LT01,LT05,PRS
--
-- In-memory hash dictionary for draft-time per-(player, champion) priors,
-- keyed on (puuid, championid) so dictGetOrDefault can resolve each pivot
-- slot directly inside arrayMap.
--
-- LIFETIME(0): data is fixed at create/reload time; rebuild player_champ_1vx
-- first, then run the build file to reload.
--
-- Auth: SOURCE(CLICKHOUSE(...)) references the `ch_internal` named collection
-- (see commands.md "Named collection for dictionary reloads").

DROP DICTIONARY IF EXISTS game_data_filtered.player_champ_1vx_dict;

CREATE DICTIONARY IF NOT EXISTS game_data_filtered.player_champ_1vx_dict
(
    puuid      String,
    championid Int32,
    matchups   UInt32,
    win_rate   Float32
)
PRIMARY KEY puuid, championid
SOURCE(CLICKHOUSE(
    NAME 'ch_internal'
    QUERY 'SELECT puuid, championid, matchups, win_rate FROM game_data_filtered.player_champ_1vx WHERE split = ''train'''
))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0);
