CREATE DATABASE IF NOT EXISTS game_data;

CREATE TABLE IF NOT EXISTS game_data.matchid_puuids
(
    puuid String CODEC(ZSTD(3))
)
ENGINE = MergeTree
ORDER BY puuid;