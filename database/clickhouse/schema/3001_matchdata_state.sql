CREATE TABLE IF NOT EXISTS game_data.matchdata_state
(
    matchid String CODEC (ZSTD(3)),
    status Enum8(
        'pending' = 1,
        'processing' = 2,
        'finished' = 3,
        'failed' = 4
    ),
    retry_count UInt16,
    error_message String CODEC (ZSTD(3)),
    run_id Nullable(UUID),
    updated_at DateTime64(3, 'UTC'),
    state_version UInt64,
    INDEX idx_status status TYPE set(4) GRANULARITY 1
)
ENGINE = ReplacingMergeTree(state_version)
ORDER BY (matchid);
