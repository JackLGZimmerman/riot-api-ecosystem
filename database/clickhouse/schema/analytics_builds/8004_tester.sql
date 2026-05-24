-- WITH filtered_matches AS (
--     SELECT matchid
--     FROM game_data_filtered.participant_stats
--     GROUP BY matchid
--     HAVING uniqExact((teamid, teamposition)) != 10
-- )

-- SELECT ps.*
-- FROM game_data_filtered.participant_stats AS ps
-- WHERE ps.matchid IN (SELECT fm.matchid FROM filtered_matches AS fm)
-- ORDER BY ps.matchid

select count() from game_data.info;
