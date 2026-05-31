-- noqa: disable=AL09,LT02,LT05,RF02,ST09
-- Draft-time-safe variant of 6900_ml_game_player_pivot_build.sql.
--
-- This keeps the same table shape but replaces final item-derived build labels
-- with the constant 'unknown'. Use this before rebuilding no-build aggregate
-- priors for draft-time experiments.

TRUNCATE TABLE game_data_filtered.ml_game_player_pivot;

INSERT INTO game_data_filtered.ml_game_player_pivot
WITH
players AS (
    SELECT
        s.matchid AS matchid,
        s.split AS split,
        ps.teamid AS teamid,
        toUInt8(ps.win > 0) AS win,
        assumeNotNull(ps.championid) AS championid,
        toString(ps.teamposition) AS teamposition,
        'unknown' AS build
    FROM game_data_filtered.ml_game_split AS s
    INNER JOIN game_data_filtered.participant_stats AS ps
        ON s.matchid = ps.matchid
)

SELECT
    matchid,
    any(split) AS split,
    anyIf(win, teamid = 100) AS blue_win,
    [
        anyIf((championid, teamposition, build), teamid = 100 AND teamposition = 'TOP'),
        anyIf((championid, teamposition, build), teamid = 100 AND teamposition = 'JUNGLE'),
        anyIf((championid, teamposition, build), teamid = 100 AND teamposition = 'MIDDLE'),
        anyIf((championid, teamposition, build), teamid = 100 AND teamposition = 'BOTTOM'),
        anyIf((championid, teamposition, build), teamid = 100 AND teamposition = 'UTILITY')
    ] AS blue_players,
    [
        anyIf((championid, teamposition, build), teamid = 200 AND teamposition = 'TOP'),
        anyIf((championid, teamposition, build), teamid = 200 AND teamposition = 'JUNGLE'),
        anyIf((championid, teamposition, build), teamid = 200 AND teamposition = 'MIDDLE'),
        anyIf((championid, teamposition, build), teamid = 200 AND teamposition = 'BOTTOM'),
        anyIf((championid, teamposition, build), teamid = 200 AND teamposition = 'UTILITY')
    ] AS red_players
FROM players
GROUP BY matchid;
