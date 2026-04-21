-- sqlfluff: disable=PRS
-- Maps itemid to its Community Dragon image URL.
-- Source: database/clickhouse/support/item_images.jsonl (mounted in the
-- ClickHouse container at /var/lib/clickhouse/user_files/clickhouse_support/).
CREATE OR REPLACE DICTIONARY game_data.item_image_map_dict
(
    id UInt32,
    name String DEFAULT '',
    image String DEFAULT ''
)
PRIMARY KEY id
SOURCE(
    FILE(
        path '/var/lib/clickhouse/user_files/clickhouse_support/item_images.jsonl'
        format 'JSONEachRow'
    )
)
LAYOUT(HASHED())
LIFETIME(MIN 0 MAX 0);
