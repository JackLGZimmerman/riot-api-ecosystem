-- noqa: disable=AL09,LT02,LT05,RF02,ST09
-- Label filtered games as chronological train/validation/test splits.
--
-- Read ordering timestamps from source game_data.info filtered by the current
-- valid_game_ids set. This keeps ML splits aligned during fast filter iteration,
-- even when the wider game_data_filtered.info copy has not been refreshed yet.

TRUNCATE TABLE game_data_filtered.ml_game_split;

INSERT INTO game_data_filtered.ml_game_split
WITH
info_one_row AS (
    SELECT
        matchid,
        min(gamestarttimestamp) AS gamestarttimestamp,
        min(gamecreation) AS gamecreation
    FROM game_data.info
    WHERE matchid IN (SELECT matchid FROM game_data_filtered.valid_game_ids)
    GROUP BY matchid
),

ranked_games AS (
    SELECT
        matchid,
        row_number() OVER (
            ORDER BY gamestarttimestamp ASC, gamecreation ASC, matchid ASC
        ) AS split_index,
        count() OVER () AS total_games
    FROM info_one_row
),

split_info AS (
    SELECT
        matchid,
        split_index,
        total_games
            - toUInt64(round(total_games * 0.1))  -- test
            - toUInt64(round(total_games * 0.1))  -- validation
            AS train_games,
        toUInt64(round(total_games * 0.1)) AS validation_games
    FROM ranked_games
)

SELECT
    matchid,
    multiIf(
        split_index <= train_games, 'train',
        split_index <= train_games + validation_games, 'validation',
        'test'
    ) AS split
FROM split_info;
