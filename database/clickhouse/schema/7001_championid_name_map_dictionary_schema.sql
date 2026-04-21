-- sqlfluff: disable=PRS
CREATE OR REPLACE DICTIONARY game_data.championid_name_map_dict
(
    _key Int32,
    name String DEFAULT ''
)
PRIMARY KEY _key
SOURCE(
    FILE(
        path '/var/lib/clickhouse/user_files/clickhouse_support/championid_name_map.jsonl'
        format 'JSONEachRow'
    )
)
LAYOUT(HASHED())
LIFETIME(MIN 0 MAX 0);
