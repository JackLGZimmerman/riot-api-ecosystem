INSERT INTO game_data.matchdata_state
    (
        matchid,
        status,
        retry_count,
        error_message,
        run_id,
        updated_at,
        state_version
    )
SELECT
    src.matchid,
    'pending' AS status,
    toUInt16(0) AS retry_count,
    '' AS error_message,
    CAST(NULL, 'Nullable(UUID)') AS run_id,
    now64(3) AS updated_at,
    toUInt64(toUnixTimestamp64Nano(now64(9)) + rowNumberInAllBlocks()) AS state_version
FROM (
    SELECT DISTINCT matchid
    FROM game_data.matchids
) AS src
LEFT JOIN (
    SELECT DISTINCT matchid
    FROM game_data.matchdata_state
) AS existing USING (matchid)
WHERE existing.matchid IS NULL;
