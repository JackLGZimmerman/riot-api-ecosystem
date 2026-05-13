-- noqa: disable=AL09,LT02,LT05,LT08,LT09,RF02,ST05,ST09
-- Build per-game token-level (matchups, primary_wins) by joining the
-- role-pivot table against each model-consumed 6xxx aggregate. Each token
-- block is its own INSERT so memory is released between blocks. All right-side
-- aggregates are filtered to split = 'train' and matchups >= 5.

TRUNCATE TABLE game_data_filtered.ml_interaction_counts;

-- ----------------------------------------------------------------------------
-- Block 1: 1vX synergy blue (token_idx 0..4)
-- ----------------------------------------------------------------------------
INSERT INTO game_data_filtered.ml_interaction_counts
WITH
expanded AS (
    SELECT
        p.matchid AS matchid,
        pn.number AS idx,
        p.blue_players[pn.number + 1] AS player
    FROM game_data_filtered.ml_game_player_pivot AS p
    CROSS JOIN numbers(5) AS pn
)
SELECT
    e.matchid,
    toUInt16(e.idx) AS token_idx,
    s.matchups,
    s.wins AS primary_wins
FROM expanded AS e
INNER JOIN (
    SELECT
        championid, teamposition, build,
        matchups, wins
    FROM game_data_filtered.synergy_1vx
    WHERE split = 'train' AND matchups >= 5
) AS s
    ON
        s.championid = tupleElement(e.player, 1)
        AND s.teamposition = tupleElement(e.player, 2)
        AND s.build = tupleElement(e.player, 3);

-- ----------------------------------------------------------------------------
-- Block 2: 1vX synergy red (token_idx 5..9)
-- ----------------------------------------------------------------------------
INSERT INTO game_data_filtered.ml_interaction_counts
WITH
expanded AS (
    SELECT
        p.matchid AS matchid,
        pn.number AS idx,
        p.red_players[pn.number + 1] AS player
    FROM game_data_filtered.ml_game_player_pivot AS p
    CROSS JOIN numbers(5) AS pn
)
SELECT
    e.matchid,
    toUInt16(5 + e.idx) AS token_idx,
    s.matchups,
    s.wins AS primary_wins
FROM expanded AS e
INNER JOIN (
    SELECT
        championid, teamposition, build,
        matchups, wins
    FROM game_data_filtered.synergy_1vx
    WHERE split = 'train' AND matchups >= 5
) AS s
    ON
        s.championid = tupleElement(e.player, 1)
        AND s.teamposition = tupleElement(e.player, 2)
        AND s.build = tupleElement(e.player, 3);

/*
Temporarily disabled for the current 1vX-only training session. Keep the
token layout in app/ml/config.py intact, but skip populating the heavier
interaction-count blocks so their cache slots remain present and zero-filled.

-- ----------------------------------------------------------------------------
-- Block 3: 1v1 (token_idx 10..34) - 5 blue x 5 red, role-ordered
-- ----------------------------------------------------------------------------
INSERT INTO game_data_filtered.ml_interaction_counts
WITH
expanded AS (
    SELECT
        p.matchid AS matchid,
        bn.number AS bi,
        rn.number AS ri,
        p.blue_players[bn.number + 1] AS bp,
        p.red_players[rn.number + 1] AS rp
    FROM game_data_filtered.ml_game_player_pivot AS p
    CROSS JOIN numbers(5) AS bn
    CROSS JOIN numbers(5) AS rn
),
canonical AS (
    SELECT
        matchid,
        toUInt16(10 + bi * 5 + ri) AS token_idx,
        bp <= rp AS blue_is_left,
        if(bp <= rp, bp, rp) AS left_p,
        if(bp <= rp, rp, bp) AS right_p
    FROM expanded
)
SELECT
    c.matchid,
    c.token_idx,
    m.matchups,
    if(c.blue_is_left, m.left_wins, m.right_wins) AS primary_wins
FROM canonical AS c
INNER JOIN (
    SELECT
        left_championid, left_teamposition, left_build,
        right_championid, right_teamposition, right_build,
        matchups, left_wins, right_wins
    FROM game_data_filtered.matchup_1v1
    WHERE split = 'train' AND matchups >= 5
) AS m
    ON
        m.left_championid = tupleElement(c.left_p, 1)
        AND m.left_teamposition = tupleElement(c.left_p, 2)
        AND m.left_build = tupleElement(c.left_p, 3)
        AND m.right_championid = tupleElement(c.right_p, 1)
        AND m.right_teamposition = tupleElement(c.right_p, 2)
        AND m.right_build = tupleElement(c.right_p, 3);

-- ----------------------------------------------------------------------------
-- Block 4: 2vX synergy blue (token_idx 35..44)
-- ----------------------------------------------------------------------------
INSERT INTO game_data_filtered.ml_interaction_counts
WITH
[(1, 2), (1, 3), (1, 4), (1, 5),
 (2, 3), (2, 4), (2, 5),
 (3, 4), (3, 5),
 (4, 5)] AS combos2_idx,
expanded AS (
    SELECT
        p.matchid AS matchid,
        bp_i.number AS b_idx,
        arraySort([
            p.blue_players[tupleElement(combos2_idx[bp_i.number + 1], 1)],
            p.blue_players[tupleElement(combos2_idx[bp_i.number + 1], 2)]
        ]) AS pair
    FROM game_data_filtered.ml_game_player_pivot AS p
    CROSS JOIN numbers(10) AS bp_i
)
SELECT
    e.matchid,
    toUInt16(35 + e.b_idx) AS token_idx,
    s.matchups,
    s.wins AS primary_wins
FROM expanded AS e
INNER JOIN (
    SELECT
        championid_1, teamposition_1, build_1,
        championid_2, teamposition_2, build_2,
        matchups, wins
    FROM game_data_filtered.synergy_2vx
    WHERE split = 'train' AND matchups >= 5
) AS s
    ON
        s.championid_1 = tupleElement(e.pair[1], 1)
        AND s.teamposition_1 = tupleElement(e.pair[1], 2)
        AND s.build_1 = tupleElement(e.pair[1], 3)
        AND s.championid_2 = tupleElement(e.pair[2], 1)
        AND s.teamposition_2 = tupleElement(e.pair[2], 2)
        AND s.build_2 = tupleElement(e.pair[2], 3);

-- ----------------------------------------------------------------------------
-- Block 5: 2vX synergy red (token_idx 45..54)
-- ----------------------------------------------------------------------------
INSERT INTO game_data_filtered.ml_interaction_counts
WITH
[(1, 2), (1, 3), (1, 4), (1, 5),
 (2, 3), (2, 4), (2, 5),
 (3, 4), (3, 5),
 (4, 5)] AS combos2_idx,
expanded AS (
    SELECT
        p.matchid AS matchid,
        rp_i.number AS r_idx,
        arraySort([
            p.red_players[tupleElement(combos2_idx[rp_i.number + 1], 1)],
            p.red_players[tupleElement(combos2_idx[rp_i.number + 1], 2)]
        ]) AS pair
    FROM game_data_filtered.ml_game_player_pivot AS p
    CROSS JOIN numbers(10) AS rp_i
)
SELECT
    e.matchid,
    toUInt16(45 + e.r_idx) AS token_idx,
    s.matchups,
    s.wins AS primary_wins
FROM expanded AS e
INNER JOIN (
    SELECT
        championid_1, teamposition_1, build_1,
        championid_2, teamposition_2, build_2,
        matchups, wins
    FROM game_data_filtered.synergy_2vx
    WHERE split = 'train' AND matchups >= 5
) AS s
    ON
        s.championid_1 = tupleElement(e.pair[1], 1)
        AND s.teamposition_1 = tupleElement(e.pair[1], 2)
        AND s.build_1 = tupleElement(e.pair[1], 3)
        AND s.championid_2 = tupleElement(e.pair[2], 1)
        AND s.teamposition_2 = tupleElement(e.pair[2], 2)
        AND s.build_2 = tupleElement(e.pair[2], 3);

-- ----------------------------------------------------------------------------
-- Block 6: 2v1 blue-pair vs red-single (token_idx 55..104)
-- ----------------------------------------------------------------------------
INSERT INTO game_data_filtered.ml_interaction_counts
WITH
[(1, 2), (1, 3), (1, 4), (1, 5),
 (2, 3), (2, 4), (2, 5),
 (3, 4), (3, 5),
 (4, 5)] AS combos2_idx,
expanded AS (
    SELECT
        p.matchid AS matchid,
        bp_i.number AS b_idx,
        sn.number AS s_idx,
        arraySort([
            p.blue_players[tupleElement(combos2_idx[bp_i.number + 1], 1)],
            p.blue_players[tupleElement(combos2_idx[bp_i.number + 1], 2)]
        ]) AS pair,
        p.red_players[sn.number + 1] AS single
    FROM game_data_filtered.ml_game_player_pivot AS p
    CROSS JOIN numbers(10) AS bp_i
    CROSS JOIN numbers(5) AS sn
)
SELECT
    e.matchid,
    toUInt16(55 + e.b_idx * 5 + e.s_idx) AS token_idx,
    m.matchups,
    m.pair_wins AS primary_wins
FROM expanded AS e
INNER JOIN (
    SELECT
        pair_championid_1, pair_teamposition_1, pair_build_1,
        pair_championid_2, pair_teamposition_2, pair_build_2,
        single_championid, single_teamposition, single_build,
        matchups, pair_wins
    FROM game_data_filtered.matchup_2v1
    WHERE split = 'train' AND matchups >= 5
) AS m
    ON
        m.pair_championid_1 = tupleElement(e.pair[1], 1)
        AND m.pair_teamposition_1 = tupleElement(e.pair[1], 2)
        AND m.pair_build_1 = tupleElement(e.pair[1], 3)
        AND m.pair_championid_2 = tupleElement(e.pair[2], 1)
        AND m.pair_teamposition_2 = tupleElement(e.pair[2], 2)
        AND m.pair_build_2 = tupleElement(e.pair[2], 3)
        AND m.single_championid = tupleElement(e.single, 1)
        AND m.single_teamposition = tupleElement(e.single, 2)
        AND m.single_build = tupleElement(e.single, 3);

-- ----------------------------------------------------------------------------
-- Block 7: 2v1 red-pair vs blue-single (token_idx 105..154)
-- ----------------------------------------------------------------------------
INSERT INTO game_data_filtered.ml_interaction_counts
WITH
[(1, 2), (1, 3), (1, 4), (1, 5),
 (2, 3), (2, 4), (2, 5),
 (3, 4), (3, 5),
 (4, 5)] AS combos2_idx,
expanded AS (
    SELECT
        p.matchid AS matchid,
        rp_i.number AS r_idx,
        sn.number AS s_idx,
        arraySort([
            p.red_players[tupleElement(combos2_idx[rp_i.number + 1], 1)],
            p.red_players[tupleElement(combos2_idx[rp_i.number + 1], 2)]
        ]) AS pair,
        p.blue_players[sn.number + 1] AS single
    FROM game_data_filtered.ml_game_player_pivot AS p
    CROSS JOIN numbers(10) AS rp_i
    CROSS JOIN numbers(5) AS sn
)
SELECT
    e.matchid,
    toUInt16(105 + e.r_idx * 5 + e.s_idx) AS token_idx,
    m.matchups,
    m.pair_wins AS primary_wins
FROM expanded AS e
INNER JOIN (
    SELECT
        pair_championid_1, pair_teamposition_1, pair_build_1,
        pair_championid_2, pair_teamposition_2, pair_build_2,
        single_championid, single_teamposition, single_build,
        matchups, pair_wins
    FROM game_data_filtered.matchup_2v1
    WHERE split = 'train' AND matchups >= 5
) AS m
    ON
        m.pair_championid_1 = tupleElement(e.pair[1], 1)
        AND m.pair_teamposition_1 = tupleElement(e.pair[1], 2)
        AND m.pair_build_1 = tupleElement(e.pair[1], 3)
        AND m.pair_championid_2 = tupleElement(e.pair[2], 1)
        AND m.pair_teamposition_2 = tupleElement(e.pair[2], 2)
        AND m.pair_build_2 = tupleElement(e.pair[2], 3)
        AND m.single_championid = tupleElement(e.single, 1)
        AND m.single_teamposition = tupleElement(e.single, 2)
        AND m.single_build = tupleElement(e.single, 3);

-- ----------------------------------------------------------------------------
-- Block 8: 3vX synergy blue (token_idx 155..164)
-- ----------------------------------------------------------------------------
INSERT INTO game_data_filtered.ml_interaction_counts
WITH
[(1, 2, 3), (1, 2, 4), (1, 2, 5),
 (1, 3, 4), (1, 3, 5), (1, 4, 5),
 (2, 3, 4), (2, 3, 5), (2, 4, 5),
 (3, 4, 5)] AS combos3_idx,
expanded AS (
    SELECT
        p.matchid AS matchid,
        tp_i.number AS t_idx,
        arraySort([
            p.blue_players[tupleElement(combos3_idx[tp_i.number + 1], 1)],
            p.blue_players[tupleElement(combos3_idx[tp_i.number + 1], 2)],
            p.blue_players[tupleElement(combos3_idx[tp_i.number + 1], 3)]
        ]) AS trio
    FROM game_data_filtered.ml_game_player_pivot AS p
    CROSS JOIN numbers(10) AS tp_i
)
SELECT
    e.matchid,
    toUInt16(155 + e.t_idx) AS token_idx,
    s.matchups,
    s.wins AS primary_wins
FROM expanded AS e
INNER JOIN (
    SELECT
        championid_1, teamposition_1, build_1,
        championid_2, teamposition_2, build_2,
        championid_3, teamposition_3, build_3,
        matchups, wins
    FROM game_data_filtered.synergy_3vx
    WHERE split = 'train' AND matchups >= 5
) AS s
    ON
        s.championid_1 = tupleElement(e.trio[1], 1)
        AND s.teamposition_1 = tupleElement(e.trio[1], 2)
        AND s.build_1 = tupleElement(e.trio[1], 3)
        AND s.championid_2 = tupleElement(e.trio[2], 1)
        AND s.teamposition_2 = tupleElement(e.trio[2], 2)
        AND s.build_2 = tupleElement(e.trio[2], 3)
        AND s.championid_3 = tupleElement(e.trio[3], 1)
        AND s.teamposition_3 = tupleElement(e.trio[3], 2)
        AND s.build_3 = tupleElement(e.trio[3], 3);

-- ----------------------------------------------------------------------------
-- Block 9: 3vX synergy red (token_idx 165..174)
-- ----------------------------------------------------------------------------
INSERT INTO game_data_filtered.ml_interaction_counts
WITH
[(1, 2, 3), (1, 2, 4), (1, 2, 5),
 (1, 3, 4), (1, 3, 5), (1, 4, 5),
 (2, 3, 4), (2, 3, 5), (2, 4, 5),
 (3, 4, 5)] AS combos3_idx,
expanded AS (
    SELECT
        p.matchid AS matchid,
        tp_i.number AS t_idx,
        arraySort([
            p.red_players[tupleElement(combos3_idx[tp_i.number + 1], 1)],
            p.red_players[tupleElement(combos3_idx[tp_i.number + 1], 2)],
            p.red_players[tupleElement(combos3_idx[tp_i.number + 1], 3)]
        ]) AS trio
    FROM game_data_filtered.ml_game_player_pivot AS p
    CROSS JOIN numbers(10) AS tp_i
)
SELECT
    e.matchid,
    toUInt16(165 + e.t_idx) AS token_idx,
    s.matchups,
    s.wins AS primary_wins
FROM expanded AS e
INNER JOIN (
    SELECT
        championid_1, teamposition_1, build_1,
        championid_2, teamposition_2, build_2,
        championid_3, teamposition_3, build_3,
        matchups, wins
    FROM game_data_filtered.synergy_3vx
    WHERE split = 'train' AND matchups >= 5
) AS s
    ON
        s.championid_1 = tupleElement(e.trio[1], 1)
        AND s.teamposition_1 = tupleElement(e.trio[1], 2)
        AND s.build_1 = tupleElement(e.trio[1], 3)
        AND s.championid_2 = tupleElement(e.trio[2], 1)
        AND s.teamposition_2 = tupleElement(e.trio[2], 2)
        AND s.build_2 = tupleElement(e.trio[2], 3)
        AND s.championid_3 = tupleElement(e.trio[3], 1)
        AND s.teamposition_3 = tupleElement(e.trio[3], 2)
        AND s.build_3 = tupleElement(e.trio[3], 3);
*/
