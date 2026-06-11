-- noqa: disable=AL09,LT02,LT05,RF02,ST09
-- Label filtered games as per-patch chronological train/test splits.
--
-- Each (season, patch) partition is ordered chronologically and split 80/20:
-- the first floor(0.8 * patch_games) games are train, the remainder test, so
-- every patch contributes same-patch history to train-side priors before its
-- tail is scored. Patches with 2+ games keep at least 1 train and 1 test row;
-- a 1-game patch is labelled train.
--
-- Read ordering timestamps from source game_data.info filtered by the current
-- valid_game_ids set. game_data_filtered.info is no longer mirrored because the
-- production ML path only needs the split labels and participant-level rows.

TRUNCATE TABLE game_data_filtered.ml_game_split;

INSERT INTO game_data_filtered.ml_game_split
WITH
info_one_row AS (
    SELECT
        matchid,
        min(season) AS season,
        min(patch) AS patch,
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
            PARTITION BY season, patch
            ORDER BY gamestarttimestamp ASC, gamecreation ASC, matchid ASC
        ) AS split_index,
        count() OVER (PARTITION BY season, patch) AS patch_games
    FROM info_one_row
)

SELECT
    matchid,
    if(
        split_index <= if(
            patch_games <= 1,
            patch_games,
            greatest(toUInt64(floor(patch_games * 0.8)), 1)
        ),
        'train',
        'test'
    ) AS split
FROM ranked_games;
