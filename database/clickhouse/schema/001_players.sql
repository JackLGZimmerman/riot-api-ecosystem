CREATE DATABASE IF NOT EXISTS game_data;

CREATE TABLE IF NOT EXISTS game_data.players
(
    puuid String CODEC(ZSTD(3)),
    queue_type Enum8('RANKED_SOLO_5x5' = 1, 'RANKED_FLEX_SR' = 2),
    tier Enum8(
        'IRON' = 1, 'BRONZE' = 2, 'SILVER' = 3, 'GOLD' = 4,
        'PLATINUM' = 5, 'EMERALD' = 6, 'DIAMOND' = 7,
        'MASTER' = 8, 'GRANDMASTER' = 9, 'CHALLENGER' = 10
    ),
    division Enum8('I' = 1, 'II' = 2, 'III' = 3, 'IV' = 4),
    wins UInt16,
    losses UInt16,
    region Enum8(
        'br1' = 1,
        'la1' = 2,
        'la2' = 3,
        'na1' = 4,
        'euw1' = 5,
        'eun1' = 6,
        'ru' = 7,
        'tr1' = 8,
        'me1' = 9,
        'jp1' = 10,
        'kr' = 11,
        'tw2' = 12,
        'oc1' = 13,
        'vn2' = 14,
        'sg2' = 15
    ),
    updated_at DateTime64(3) CODEC(DoubleDelta, ZSTD(1))
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (puuid, queue_type)
TTL updated_at + INTERVAL 2 MONTH DELETE;