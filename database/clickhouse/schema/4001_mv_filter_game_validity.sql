CREATE MATERIALIZED VIEW IF NOT EXISTS game_data.mv_filter_game_validity
TO game_data.filter_game_validity
AS
SELECT
    gameid,
    rule_mask,
    if(rule_mask = 0, 1, 0) AS is_valid
FROM
(
    SELECT
        gameid,
        (
            groupBitOr(player_rule_mask) +
            if(max(low_team_kda) = 1, 256, 0) +
            if(max(gameendedinearlysurrender_game) = 1, 65536, 0)
        ) AS rule_mask
    FROM
    (
        SELECT
            gameid,
            teamid,
            participantid,
            teamposition,
            kills,
            assists,
            deaths,
            goldspent,
            goldearned,
            summoner1casts,
            summoner2casts,
            item0,
            item1,
            item2,
            item3,
            item4,
            item5,
            item6,
            totalminionskilled,
            totaldamagedealttochampions,
            sum(kills) OVER (PARTITION BY gameid, teamid) AS team_kills,
            sum(assists) OVER (PARTITION BY gameid, teamid) AS team_assists,
            sum(deaths) OVER (PARTITION BY gameid, teamid) AS team_deaths,
            sum(totaldamagedealttochampions) OVER (PARTITION BY gameid, teamid) AS team_totaldamagedealttochampions,
            any(timeplayed) OVER (PARTITION BY gameid) AS timeplayed_game,
            any(gameendedinearlysurrender) OVER (PARTITION BY gameid) AS gameendedinearlysurrender_game,
            (
                (sum(kills) OVER (PARTITION BY gameid, teamid) + sum(assists) OVER (PARTITION BY gameid, teamid))
                / nullIf(toFloat32(sum(deaths) OVER (PARTITION BY gameid, teamid)), 0.0)
            ) < 0.25 AS low_team_kda,
            (
                if(((kills + assists) / nullIf(toFloat32(deaths), 0.0)) < 0.2, 1, 0) +
                if((goldspent / nullIf(toFloat32(goldearned), 0.0)) < 0.60, 2, 0) +
                if((kills + assists = 0) AND (deaths > 4), 4, 0) +
                if((summoner1casts = 0) OR (summoner2casts = 0), 8, 0) +
                if((kills / nullIf(toFloat32(sum(kills) OVER (PARTITION BY gameid, teamid)), 0.0)) > 0.65, 512, 0) +
                if(
                    (totaldamagedealttochampions / nullIf(toFloat32(sum(totaldamagedealttochampions) OVER (PARTITION BY gameid, teamid)), 0.0)) < 0.075
                    AND teamposition != 'UTILITY',
                    1024,
                    0
                ) +
                if(
                    teamposition != 'UTILITY'
                    AND (totalminionskilled / nullIf(toFloat32(max(timeplayed) OVER (PARTITION BY gameid)) / 60.0, 0.0)) < 4.5,
                    2048,
                    0
                ) +
                if(
                    item0 = 0
                    AND item1 = 0
                    AND item2 = 0
                    AND item3 = 0
                    AND item4 = 0
                    AND item5 = 0
                    AND item6 = 0,
                    4096,
                    0
                ) +
                if(
                    greatest(
                        (item0 = item0) + (item1 = item0) + (item2 = item0) + (item3 = item0) + (item4 = item0) + (item5 = item0) + (item6 = item0),
                        (item0 = item1) + (item1 = item1) + (item2 = item1) + (item3 = item1) + (item4 = item1) + (item5 = item1) + (item6 = item1),
                        (item0 = item2) + (item1 = item2) + (item2 = item2) + (item3 = item2) + (item4 = item2) + (item5 = item2) + (item6 = item2),
                        (item0 = item3) + (item1 = item3) + (item2 = item3) + (item3 = item3) + (item4 = item3) + (item5 = item3) + (item6 = item3),
                        (item0 = item4) + (item1 = item4) + (item2 = item4) + (item3 = item4) + (item4 = item4) + (item5 = item4) + (item6 = item4),
                        (item0 = item5) + (item1 = item5) + (item2 = item5) + (item3 = item5) + (item4 = item5) + (item5 = item5) + (item6 = item5),
                        (item0 = item6) + (item1 = item6) + (item2 = item6) + (item3 = item6) + (item4 = item6) + (item5 = item6) + (item6 = item6)
                    ) >= 5,
                    8192,
                    0
                )
            ) AS player_rule_mask
        FROM game_data.participant_stats
    ) AS participant_enriched
    GROUP BY gameid
) AS game_rules;
