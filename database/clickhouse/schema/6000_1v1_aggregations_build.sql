-- noqa: disable=AL09,LT02,LT05,RF02,ST09
-- 1v1 (champion, teamposition, build) cross-team matchup win rates.
-- Canonicalised so the smaller (championid, teamposition, build) tuple is on the left.
-- left_win_rate + right_win_rate = 1.0.
-- Built from ml_game_player_pivot so participant/item labels are joined and
-- role-pivoted once for all matchup aggregate builders.
-- Leakage-safe training prior: only train games contribute outcome counts.
-- Validation/test rows should join against split = 'train'; train feature rows
-- must subtract their current match contribution at feature-build time.

TRUNCATE TABLE game_data_filtered.matchup_1v1;

INSERT INTO game_data_filtered.matchup_1v1
WITH
cross_pairs AS (
    SELECT
        p.split AS split,
        p.blue_players[bn.number + 1] AS bp,
        p.red_players[rn.number + 1] AS rp,
        p.blue_win AS blue_win
    FROM game_data_filtered.ml_game_player_pivot AS p
    CROSS JOIN numbers(5) AS bn
    CROSS JOIN numbers(5) AS rn
    WHERE p.split = 'train'
),

canonical AS (
    SELECT
        split,
        bp <= rp AS blue_is_left,
        if(blue_is_left, bp, rp) AS left_p,
        if(blue_is_left, rp, bp) AS right_p,
        if(blue_is_left, blue_win, toUInt8(1 - blue_win)) AS left_win,
        if(blue_is_left, toUInt8(1 - blue_win), blue_win) AS right_win
    FROM cross_pairs
)

SELECT
    split,
    tupleElement(left_p, 1) AS left_championid,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(left_championid), '') AS left_championname,
    tupleElement(left_p, 2) AS left_teamposition,
    tupleElement(left_p, 3) AS left_build,
    tupleElement(right_p, 1) AS right_championid,
    dictGetOrDefault('game_data.championid_name_map_dict', 'name', toString(right_championid), '') AS right_championname,
    tupleElement(right_p, 2) AS right_teamposition,
    tupleElement(right_p, 3) AS right_build,
    count() AS matchups,
    sum(left_win) AS left_wins,
    sum(right_win) AS right_wins,
    toFloat32(left_wins / matchups) AS left_win_rate,
    toFloat32(right_wins / matchups) AS right_win_rate
FROM canonical
GROUP BY
    split,
    left_championid, left_championname, left_teamposition, left_build,
    right_championid, right_championname, right_teamposition, right_build;
