CREATE TABLE IF NOT EXISTS game_data.matchid_puuids
(
    run_id UUID,
    puuid FixedString (78) CODEC (ZSTD(3))
)
ENGINE = ReplacingMergeTree
ORDER BY (run_id, puuid);
