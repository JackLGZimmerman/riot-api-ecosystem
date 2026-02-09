CREATE TABLE IF NOT EXISTS game_data.matchid_puuids
(
    run_id UUID,
    puuid STRING CODEC (ZSTD(3))
)
ENGINE = REPLACINGMERGETREE(run_id)
ORDER BY (run_id, puuid);
