CREATE TABLE IF NOT EXISTS game_data.filter_game_validity
(
    matchid UInt64,
    teamid UInt8,
    participantid UInt8,
    player_rule_mask UInt32,
    team_rule_mask UInt32,
    game_rule_mask UInt32,
    rule_mask UInt32,
    is_valid UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, participantid);

CREATE MATERIALIZED VIEW IF NOT EXISTS game_data.mv_filter_game_validity
TO game_data.filter_game_validity
AS
WITH participant_base AS (
    SELECT
        matchid,
        teamid,
        participantid,
        kills,
        assists,
        deaths,
        timeplayed,
        goldspent,
        goldearned,
        summoner1casts,
        summoner2casts,
        teamposition,
        totalminionskilled,
        totaldamagedealttochampions,
        item0,
        item1,
        item2,
        item3,
        item4,
        item5
    FROM game_data.participant_stats
),

team_stats AS (
    SELECT
        matchid,
        teamid,
        SUM(kills) AS team_kills,
        SUM(assists) AS team_assists,
        SUM(deaths) AS team_deaths,
        SUM(totaldamagedealttochampions) AS team_damage,
        toUInt32(
            IF(
                (
                    (SUM(kills) + SUM(assists))
                    / nullIf(toFloat32(SUM(deaths)), 0.0)
                ) < 0.25,
                256,
                0
            )
        ) AS team_rule_mask
    FROM participant_base
    GROUP BY
        matchid,
        teamid
),

game_stats AS (
    SELECT
        matchid,
        MAX(timeplayed) AS game_max_timeplayed,
        toUInt32(IF(MAX(timeplayed) <= 15, 65536, 0)) AS game_rule_mask
    FROM participant_base
    GROUP BY matchid
),

participant_scored AS (
    SELECT
        p.matchid AS matchid_value,
        p.teamid AS teamid_value,
        p.participantid AS participantid_value,
        t.team_rule_mask AS team_mask,
        g.game_rule_mask AS game_mask,
        toUInt32(
            IF((p.deaths > 0) AND (p.kills + p.assists * 5 < p.deaths), 1, 0)
            + IF((p.goldearned > 0) AND (p.goldspent * 100 < p.goldearned * 60), 2, 0)
            + IF((p.kills + p.assists = 0) AND (p.deaths > 4), 4, 0)
            + IF((p.summoner1casts = 0) OR (p.summoner2casts = 0), 8, 0)
            + IF(
                (t.team_kills > 0) AND (p.kills * 100 > t.team_kills * 65),
                512,
                0
            )
            + IF(
                (p.teamposition != 'UTILITY')
                AND (t.team_damage > 0)
                AND (p.totaldamagedealttochampions * 1000 < t.team_damage * 75),
                1024,
                0
            )
            + IF(
                (p.teamposition != 'UTILITY')
                AND (g.game_max_timeplayed > 0)
                AND (p.totalminionskilled * 10 < g.game_max_timeplayed * 45),
                2048,
                0
            )
            + IF(
                p.item0 = 0
                AND p.item1 = 0
                AND p.item2 = 0
                AND p.item3 = 0
                AND p.item4 = 0
                AND p.item5 = 0,
                4096,
                0
            )
            + IF(
                p.item0 = p.item1
                AND p.item1 = p.item2
                AND p.item2 = p.item3
                AND p.item3 = p.item4
                AND p.item4 = p.item5,
                8192,
                0
            )
        ) AS player_mask
    FROM participant_base AS p
    INNER JOIN team_stats AS t USING (matchid, teamid)
    INNER JOIN game_stats AS g USING (matchid)
)

SELECT
    matchid_value AS matchid,
    teamid_value AS teamid,
    participantid_value AS participantid,
    player_mask AS player_rule_mask,
    team_mask AS team_rule_mask,
    game_mask AS game_rule_mask,
    toUInt32((player_mask + team_mask) + game_mask) AS rule_mask,
    toUInt8(((player_mask + team_mask) + game_mask) = 0) AS is_valid
FROM participant_scored;
