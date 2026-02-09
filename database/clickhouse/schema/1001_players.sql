CREATE TABLE IF NOT EXISTS game_data.players_raw
(
    run_id UUID,
    puuid STRING,
    queue_type STRING,
    tier STRING,
    division STRING,
    wins UINT16,
    losses UINT16,
    region STRING,
    updated_at DATETIME64 (3, 'UTC')
)
ENGINE = REPLACINGMERGETREE
PARTITION BY TODATE(updated_at)
ORDER BY (puuid, queue_type, region, updated_at, run_id);
