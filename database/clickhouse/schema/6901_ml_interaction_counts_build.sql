-- noqa: disable=AL09,LT02,LT05,LT08,RF02,RF03,ST05,ST09
-- Materialise per-game, per-player-token 1vX build-labeled win-rate priors.
--
-- `synergy_1vx` stores one train-split prior per
-- (championid, teamposition, build). This intentionally excludes the older
-- time-bin and profile-metric columns.

TRUNCATE TABLE game_data_filtered.ml_interaction_counts;

INSERT INTO game_data_filtered.ml_interaction_counts
(
    matchid,
    token_idx,
    championid,
    teamposition,
    build,
    matchups,
    win_rate
)
WITH
expanded AS (
    SELECT
        p.matchid,
        toUInt16(tupleElement(token, 2)) AS token_idx,
        tupleElement(tupleElement(token, 1), 1) AS championid,
        tupleElement(tupleElement(token, 1), 2) AS teamposition,
        tupleElement(tupleElement(token, 1), 3) AS build
    FROM game_data_filtered.ml_game_player_pivot AS p
    ARRAY JOIN [
        tuple(p.blue_players[1], toUInt16(0)),
        tuple(p.blue_players[2], toUInt16(1)),
        tuple(p.blue_players[3], toUInt16(2)),
        tuple(p.blue_players[4], toUInt16(3)),
        tuple(p.blue_players[5], toUInt16(4)),
        tuple(p.red_players[1], toUInt16(5)),
        tuple(p.red_players[2], toUInt16(6)),
        tuple(p.red_players[3], toUInt16(7)),
        tuple(p.red_players[4], toUInt16(8)),
        tuple(p.red_players[5], toUInt16(9))
    ] AS token
)
SELECT
    e.matchid,
    e.token_idx,
    e.championid,
    e.teamposition,
    e.build,
    COALESCE(s.matchups, toUInt32(0)) AS matchups,
    COALESCE(s.win_rate, toFloat32(0.5)) AS win_rate
FROM expanded AS e
ANY LEFT JOIN (
    SELECT
        championid,
        teamposition,
        build,
        matchups,
        win_rate
    FROM game_data_filtered.synergy_1vx
    WHERE split = 'train'
) AS s
    ON
        s.championid = e.championid
        AND s.teamposition = e.teamposition
        AND s.build = e.build
SETTINGS join_use_nulls = 1;
