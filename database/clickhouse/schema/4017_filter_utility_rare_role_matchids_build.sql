TRUNCATE TABLE game_data.filter_utility_rare_role_matchids;

INSERT INTO game_data.filter_utility_rare_role_matchids
(
    matchid,
    has_rare_role
)
WITH
champion_teamposition_pick_counts AS (
    SELECT
        championid,
        teamposition,
        count() AS champion_teamposition_picks
    FROM game_data.participant_stats
    WHERE
        championid IS NOT NULL
        AND teamposition != 'UNKNOWN'
    GROUP BY
        championid,
        teamposition
),

champion_pick_totals AS (
    SELECT
        championid,
        sum(champion_teamposition_picks) AS champion_picks
    FROM champion_teamposition_pick_counts
    GROUP BY championid
),

rare_champion_teampositions AS (
    SELECT
        ctpc.championid,
        ctpc.teamposition
    FROM champion_teamposition_pick_counts AS ctpc
    INNER JOIN champion_pick_totals AS cpt
        USING (championid)
    WHERE ctpc.champion_teamposition_picks * 1000 < cpt.champion_picks * 6
),

player_rare_champion_teamposition_pick_counts AS (
    SELECT
        ps.puuid,
        ps.championid,
        ps.teamposition,
        count() AS player_champion_teamposition_picks
    FROM game_data.participant_stats AS ps
    INNER JOIN rare_champion_teampositions AS rct
        ON
            ps.championid = rct.championid
            AND ps.teamposition = rct.teamposition
    GROUP BY
        ps.puuid,
        ps.championid,
        ps.teamposition
    HAVING player_champion_teamposition_picks < 30
)

SELECT
    ps.matchid,
    1 AS has_rare_role
FROM game_data.participant_stats AS ps
INNER JOIN player_rare_champion_teamposition_pick_counts AS prctpc
    ON
        ps.puuid = prctpc.puuid
        AND ps.championid = prctpc.championid
        AND ps.teamposition = prctpc.teamposition
GROUP BY ps.matchid;
