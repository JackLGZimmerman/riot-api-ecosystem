CREATE VIEW IF NOT EXISTS game_data_filtered.v_matchup_1v1_source AS
WITH players AS (
    SELECT
        matchid,
        teamid,
        championid,
        toString(teamposition) AS team_position,
        toString(build) AS build,
        win > 0 AS win,
        map(
            'kills', kills,
            'deaths', deaths,
            'assists', assists,
            'goldearned', goldearned,
            'totaldamagedealttochampions',
            totaldamagedealttochampions,
            'totaldamagetaken',
            totaldamagetaken,
            'totalminionskilled',
            totalminionskilled,
            'visionscore', visionscore,
            'timeplayed', timeplayed
            -- add further metrics here as needed
        ) AS metrics,
        map(
            'win', win > 0,
            'firstbloodkill', firstbloodkill > 0,
            'firstbloodassist', firstbloodassist > 0,
            'firsttowerkill', firsttowerkill > 0,
            'firsttowerassist', firsttowerassist > 0,
            'gameendedinsurrender', gameendedinsurrender > 0,
            'gameendedinearlysurrender', gameendedinearlysurrender > 0
        ) AS flags
    FROM game_data_filtered.participant_stats
),

cross_pairs AS (
    SELECT
        l.matchid,
        l.teamid AS left_teamid,
        r.teamid AS right_teamid,
        l.championid AS left_champion,
        r.championid AS right_champion,
        l.team_position AS left_team_position,
        r.team_position AS right_team_position,
        l.build AS left_build,
        r.build AS right_build,
        l.win AS left_win,
        r.win AS right_win,
        l.metrics AS left_metrics,
        r.metrics AS right_metrics,
        l.flags AS left_flags,
        r.flags AS right_flags
    FROM players AS l
    INNER JOIN players AS r
        ON
            l.matchid = r.matchid
            AND l.teamid != r.teamid
)

-- Canonicalise: left_champion <= right_champion
-- When swapped, all columns flip so "left" always means
-- the side with the lower champion id
SELECT
    matchid,
    if(needs_swap, right_teamid, left_teamid) AS left_teamid,
    if(needs_swap, left_teamid, right_teamid) AS right_teamid,
    if(needs_swap, right_champion, left_champion) AS left_champion,
    if(needs_swap, left_champion, right_champion) AS right_champion,
    if(needs_swap, right_team_position, left_team_position) AS left_team_position,
    if(needs_swap, left_team_position, right_team_position) AS right_team_position,
    if(needs_swap, right_build, left_build) AS left_build,
    if(needs_swap, left_build, right_build) AS right_build,
    if(needs_swap, right_win, left_win) AS left_win,
    if(needs_swap, left_win, right_win) AS right_win,
    if(needs_swap, right_metrics, left_metrics) AS left_metrics,
    if(needs_swap, left_metrics, right_metrics) AS right_metrics,
    if(needs_swap, right_flags, left_flags) AS left_flags,
    if(needs_swap, left_flags, right_flags) AS right_flags
FROM (
    SELECT
        *,
        right_champion < left_champion AS needs_swap
    FROM cross_pairs
);
