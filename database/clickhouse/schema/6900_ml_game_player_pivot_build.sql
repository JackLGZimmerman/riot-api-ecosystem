-- noqa: disable=AL09,LT02,LT05,RF02,ST09
-- Pivot each game's participants into per-side, role-ordered arrays of
-- (championid, teamposition, build, puuid) tuples.
-- teamid 100 = blue, teamid 200 = red.

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
        toString(ivt.highest_value_label) AS build,
        toString(ps.puuid) AS puuid
    FROM game_data_filtered.ml_game_split AS s
    INNER JOIN game_data_filtered.participant_stats AS ps
        ON s.matchid = ps.matchid
    INNER JOIN game_data_filtered.participant_item_value_totals AS ivt
        ON
            ps.matchid = ivt.matchid
            AND ps.participantid = ivt.participantid
)

SELECT
    matchid,
    any(split) AS split,
    anyIf(win, teamid = 100) AS blue_win,
    [
        anyIf((championid, teamposition, build, puuid), teamid = 100 AND teamposition = 'TOP'),
        anyIf((championid, teamposition, build, puuid), teamid = 100 AND teamposition = 'JUNGLE'),
        anyIf((championid, teamposition, build, puuid), teamid = 100 AND teamposition = 'MIDDLE'),
        anyIf((championid, teamposition, build, puuid), teamid = 100 AND teamposition = 'BOTTOM'),
        anyIf((championid, teamposition, build, puuid), teamid = 100 AND teamposition = 'UTILITY')
    ] AS blue_players,
    [
        anyIf((championid, teamposition, build, puuid), teamid = 200 AND teamposition = 'TOP'),
        anyIf((championid, teamposition, build, puuid), teamid = 200 AND teamposition = 'JUNGLE'),
        anyIf((championid, teamposition, build, puuid), teamid = 200 AND teamposition = 'MIDDLE'),
        anyIf((championid, teamposition, build, puuid), teamid = 200 AND teamposition = 'BOTTOM'),
        anyIf((championid, teamposition, build, puuid), teamid = 200 AND teamposition = 'UTILITY')
    ] AS red_players
FROM players
GROUP BY matchid;
