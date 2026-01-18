CREATE DATABASE IF NOT EXISTS game_data;

CREATE TABLE IF NOT EXISTS game_data.matchids
(
    matchid     String CODEC(ZSTD(3)),
    inserted_at  DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(inserted_at)
ORDER BY matchid;