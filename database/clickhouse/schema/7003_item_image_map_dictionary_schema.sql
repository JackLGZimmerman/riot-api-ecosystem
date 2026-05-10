-- sqlfluff: disable=PRS
-- Maps itemid to its Community Dragon image URL and base price.
-- Source: database/clickhouse/support/item_info.jsonl (mounted in the
-- ClickHouse container at /var/lib/clickhouse/user_files/clickhouse_support/).
CREATE OR REPLACE DICTIONARY game_data.item_info_dict
(
    id UInt32,
    name String DEFAULT '',
    price UInt32 DEFAULT 0,
    image String DEFAULT ''
)
PRIMARY KEY id
SOURCE(
    FILE(
        path '/var/lib/clickhouse/user_files/clickhouse_support/item_info.jsonl'
        format 'JSONEachRow'
    )
)
LAYOUT(HASHED())
LIFETIME(MIN 0 MAX 0);
