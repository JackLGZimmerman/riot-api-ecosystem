-- sqlfluff: disable=PRS
CREATE DICTIONARY IF NOT EXISTS game_data.item_value_map_dict
(
    itemid UInt32,
    attack_damage Float32 DEFAULT 0,
    ability_power Float32 DEFAULT 0,
    lethality Float32 DEFAULT 0,
    on_hit Float32 DEFAULT 0,
    crit Float32 DEFAULT 0,
    tank Float32 DEFAULT 0,
    off_tank Float32 DEFAULT 0,
    utility Float32 DEFAULT 0
)
PRIMARY KEY itemid
SOURCE(
    FILE(
        path '/var/lib/clickhouse/user_files/clickhouse_support/item_value_map.jsonl'
        format 'JSONEachRow'
    )
)
LAYOUT(HASHED())
LIFETIME(MIN 0 MAX 0);
