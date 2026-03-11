CREATE TABLE IF NOT EXISTS game_data.matchid_puuids
(
    run_id UUID,
    puuid FixedString (78) CODEC (ZSTD(3)),
    queue_type LowCardinality(String)
)
ENGINE = ReplacingMergeTree
ORDER BY (run_id, puuid, queue_type);
