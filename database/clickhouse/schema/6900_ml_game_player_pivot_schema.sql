-- noqa: disable=LT01,LT05,PRS

-- Per-game pivot of the 5 blue + 5 red player tuples in fixed role order
-- (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY). Used as the join input for the
-- per-token interaction-feature build (6901) and to feed champion / role /
-- build embedding ids into the Python cache builder.

DROP TABLE IF EXISTS game_data_filtered.ml_game_player_pivot;

CREATE TABLE IF NOT EXISTS game_data_filtered.ml_game_player_pivot
(
    matchid String,
    split LowCardinality(String),
    blue_win UInt8,
    -- Each array has length 5, ordered TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY.
    blue_players Array(Tuple(Int32, String, String)),
    red_players Array(Tuple(Int32, String, String))
)
ENGINE = MergeTree
ORDER BY (split, matchid);
