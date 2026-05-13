-- noqa: disable=AL09,LT02,LT05,RF02,ST09
-- Label eligible filtered games as chronological train/validation/test splits.
-- Eligibility mirrors the ML input shape needed by downstream builds: one row
-- per selected game with exactly 10 usable participants and item-build labels.
--
-- Read ordering timestamps from source game_data.info filtered by the current
-- valid_game_ids set. This keeps ML splits aligned during fast filter iteration,
-- even when the wider game_data_filtered.info copy has not been refreshed yet.

TRUNCATE TABLE game_data_filtered.ml_game_split;

INSERT INTO game_data_filtered.ml_game_split
WITH
0.1 AS validation_fraction,
0.1 AS test_fraction,

info_one_row AS (
    SELECT
        info.matchid AS matchid,
        min(info.gamestarttimestamp) AS gamestarttimestamp,
        min(info.gamecreation) AS gamecreation
    FROM game_data.info AS info FINAL
    INNER JOIN game_data_filtered.valid_game_ids AS valid
        ON info.matchid = valid.matchid
    GROUP BY info.matchid
),

eligible_games AS (
    SELECT
        ps.matchid AS matchid,
        min(info.gamestarttimestamp) AS gamestarttimestamp,
        min(info.gamecreation) AS gamecreation,
        toUInt8(count()) AS participant_count
    FROM game_data_filtered.participant_stats AS ps
    INNER JOIN info_one_row AS info
        ON ps.matchid = info.matchid
    INNER JOIN game_data_filtered.participant_item_value_totals AS ivt
        ON
            ps.matchid = ivt.matchid
            AND ps.participantid = ivt.participantid
    WHERE
        ps.championid IS NOT NULL
        AND ps.teamposition != 'UNKNOWN'
        AND ps.timeplayed > 0
    GROUP BY ps.matchid
    HAVING count() = 10
),

ranked_games AS (
    SELECT
        matchid,
        gamestarttimestamp,
        gamecreation,
        participant_count,
        row_number() OVER (
            ORDER BY gamestarttimestamp ASC, gamecreation ASC, matchid ASC
        ) AS split_index,
        count() OVER () AS total_games
    FROM eligible_games
),

sized_splits AS (
    SELECT
        matchid,
        gamestarttimestamp,
        gamecreation,
        participant_count,
        split_index,
        total_games,
        toUInt64(round(total_games * validation_fraction)) AS validation_games,
        toUInt64(round(total_games * test_fraction)) AS test_games
    FROM ranked_games
),

bounded_splits AS (
    SELECT
        matchid,
        gamestarttimestamp,
        gamecreation,
        participant_count,
        split_index,
        total_games,
        total_games - validation_games - test_games AS train_games,
        validation_games
    FROM sized_splits
)

SELECT
    matchid,
    multiIf(
        split_index <= train_games,
        'train',
        split_index <= train_games + validation_games,
        'validation',
        'test'
    ) AS split,
    split_index,
    total_games,
    gamestarttimestamp,
    gamecreation,
    participant_count
FROM bounded_splits
ORDER BY split_index;
