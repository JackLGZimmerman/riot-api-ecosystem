-- Inspect individual players matching a specific (championid, teamposition,
-- highest_value_label) combination.  Item slot values are replaced with their
-- Community Dragon image URLs (via game_data.item_image_map_dict), and
-- championid is joined with game_data.championid_name_map_dict for a readable
-- champion name.
--
-- Edit the params CTE below to filter by a specific champion/position/build,
-- or override at runtime via clickhouse-client:
--
--   docker exec -i clickhouse clickhouse-client \
--     --param_champion_id=2 \
--     --param_team_position='JUNGLE' \
--     --param_build_label='mr_tank' \
--     < .../analytics_builds/8001_build_label_player_inspector.sql
--
-- Requires game_data_filtered.participant_item_value_totals to be populated
-- (see analytics_builds/5133_participant_item_value_totals_build.sql).
--
-- The upstream totals are built against item_value_map_dict using the
-- composite key (championid, teamposition, itemid): rows with a specific
-- (championid, teamposition) are applied to that exact pair and fall back to
-- the generic (NULL, NULL) row when no specific entry exists.  This inspector
-- therefore filters by (championid, teamposition, highest_value_label) that
-- already reflect that resolution -- no additional lookup is required here.
-- noqa: disable=ST06,LT02,PRS

WITH params AS (
    SELECT
        toInt32(2) AS champion_id,    -- e.g. 2 = Olaf
        'JUNGLE' AS team_position,  -- TOP / JUNGLE / MIDDLE / BOTTOM / UTILITY
        'mr_tank' AS build_label
)

SELECT
    ps.matchid AS gameid,
    ps.teamid,
    ps.championid,
    dictGet('game_data.championid_name_map_dict', 'name', toInt32(ps.championid))
        AS champion_name,
    dictGet('game_data.item_image_map_dict', 'image', toUInt32(ps.item0)) AS item0,
    dictGet('game_data.item_image_map_dict', 'image', toUInt32(ps.item1)) AS item1,
    dictGet('game_data.item_image_map_dict', 'image', toUInt32(ps.item2)) AS item2,
    dictGet('game_data.item_image_map_dict', 'image', toUInt32(ps.item3)) AS item3,
    dictGet('game_data.item_image_map_dict', 'image', toUInt32(ps.item4)) AS item4,
    dictGet('game_data.item_image_map_dict', 'image', toUInt32(ps.item5)) AS item5,
    dictGet('game_data.item_image_map_dict', 'image', toUInt32(ps.item6)) AS item6,
    ps.kills,
    ps.assists,
    ps.deaths,
    if(ps.win = 1, 'WIN', 'LOSS') AS outcome
FROM game_data_filtered.participant_item_value_totals AS ivt
INNER JOIN game_data_filtered.participant_stats AS ps
    ON ivt.matchid = ps.matchid
        AND ivt.participantid = ps.participantid
CROSS JOIN params
WHERE
    ivt.championid = params.champion_id
    AND ivt.teamposition = params.team_position
    AND ivt.highest_value_label = params.build_label
ORDER BY
    ps.matchid,
    ps.teamid,
    ps.participantid
