-- sqlfluff: disable=PRS
-- item_value_map.jsonl now carries (championid, teamposition, itemid) per row.
-- Rows with NULL championid / NULL teamposition are fallbacks that apply to
-- every champion/position combination; rows with populated championid /
-- teamposition apply only to that specific pair.
--
-- The file is loaded with JSONEachRow + input_format_null_as_default = 1, so
-- NULLs collapse to the default sentinel values (championid = 0,
-- teamposition = '').  Callers must first look up the specific key
-- (championid, teamposition, itemid); if that lookup misses, they fall back
-- to the sentinel key (0, '', itemid).
CREATE OR REPLACE DICTIONARY game_data.item_value_map_dict
(
    championid Int32 DEFAULT 0,
    teamposition String DEFAULT '',
    itemid UInt32,
    attack_damage Float32 DEFAULT 0,
    ability_power Float32 DEFAULT 0,
    lethality Float32 DEFAULT 0,
    on_hit Float32 DEFAULT 0,
    crit Float32 DEFAULT 0,
    utility_enchanter Float32 DEFAULT 0,
    utility_protection Float32 DEFAULT 0,
    ar_tank Float32 DEFAULT 0,
    mr_tank Float32 DEFAULT 0,
    ad_off_tank Float32 DEFAULT 0,
    ap_off_tank Float32 DEFAULT 0
)
PRIMARY KEY championid, teamposition, itemid
SOURCE(
    FILE(
        path '/var/lib/clickhouse/user_files/clickhouse_support/item_value_map.jsonl'
        format 'JSONEachRow'
    )
)
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(MIN 0 MAX 0)
SETTINGS(input_format_null_as_default = 1);
