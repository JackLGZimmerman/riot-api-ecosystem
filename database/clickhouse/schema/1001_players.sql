CREATE TABLE IF NOT EXISTS game_data.players
(
    run_id UUID,
    puuid FixedString (78),
    queue_type LowCardinality (String),
    tier LowCardinality (String),
    division LowCardinality (String),
    wins UInt16,
    losses UInt16,
    region LowCardinality (String),
    updated_at DateTime64 (3, 'UTC')
)
ENGINE = ReplacingMergeTree
PARTITION BY toDate(updated_at)
ORDER BY (puuid, queue_type, region, updated_at, run_id);
