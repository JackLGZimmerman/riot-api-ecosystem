-- noqa: disable=AL09,LT02,LT05,RF02,ST09
--
-- Cross-team 1v1 (championid, teamposition, build) matchup win rates.
-- Mirrors the structure of 6003_1vx_aggregations_build.sql: only the
-- (key, count, win_rate) priors are emitted; scaling-bin, time, and
-- profile-metric columns are intentionally excluded.
--
-- Each ml_game_player_pivot row fans out into 25 (blue, red) pairs.
-- Canonicalisation puts the smaller (championid, teamposition, build)
-- tuple on the left; left_win_rate + right_win_rate = 1.0.
-- Leakage-safe: only train games contribute outcome counts.

TRUNCATE TABLE game_data_filtered.matchup_1v1;

INSERT INTO game_data_filtered.matchup_1v1
SELECT
    'train' AS split,
    tupleElement(pair.1, 1) AS left_championid,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(left_championid),
        ''
    ) AS left_championname,
    tupleElement(pair.1, 2) AS left_teamposition,
    tupleElement(pair.1, 3) AS left_build,
    tupleElement(pair.2, 1) AS right_championid,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(right_championid),
        ''
    ) AS right_championname,
    tupleElement(pair.2, 2) AS right_teamposition,
    tupleElement(pair.2, 3) AS right_build,
    count() AS matchups,
    sum(pair.3) AS left_wins,
    matchups - left_wins AS right_wins,
    toFloat32(left_wins / matchups) AS left_win_rate,
    toFloat32(right_wins / matchups) AS right_win_rate
FROM game_data_filtered.ml_game_player_pivot AS p
ARRAY JOIN arrayFlatten(arrayMap(
    b -> arrayMap(
        r -> if(
            b <= r,
            (b, r, p.blue_win),
            (r, b, toUInt8(1 - p.blue_win))
        ),
        p.red_players
    ),
    p.blue_players
)) AS pair
WHERE p.split = 'train'
GROUP BY
    left_championid, left_teamposition, left_build,
    right_championid, right_teamposition, right_build;
