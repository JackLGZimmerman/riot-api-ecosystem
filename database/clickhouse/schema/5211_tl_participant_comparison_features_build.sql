-- noqa: disable=LT05
-- Derived participant-minute comparison features for GNN inputs.
--
-- Required schemas/builds:
--   5200_tl_participant_discrete_events_schema.sql
--   5201_tl_participant_discrete_events_build.sql
--   5210_tl_participant_comparison_features_schema.sql

TRUNCATE TABLE game_data_filtered.tl_participant_comparison_features;

INSERT INTO game_data_filtered.tl_participant_comparison_features
(
    matchid,
    frame_timestamp,
    teamid,
    participantid,
    totalgold_team_ratio,
    totalgold_role_diff,
    totalgold_role_ratio,
    totalgold_enemy_team_diff,
    xp_team_ratio,
    xp_role_diff,
    xp_role_ratio,
    xp_enemy_team_diff,
    totalfarm_team_ratio,
    totalfarm_role_diff,
    totalfarm_role_ratio,
    totalfarm_enemy_team_diff,
    kills_team_ratio,
    kills_role_diff,
    kills_enemy_team_diff,
    deaths_role_diff,
    deaths_enemy_team_diff,
    assists_team_ratio,
    assists_role_diff,
    totaldamagedonetochampions_team_ratio,
    totaldamagedonetochampions_role_diff,
    totaldamagedonetochampions_enemy_team_diff,
    totaldamagetaken_team_ratio,
    totaldamagetaken_role_diff,
    totaldamagetaken_enemy_team_diff,
    timeenemyspentcontrolled_team_ratio,
    timeenemyspentcontrolled_role_diff,
    timeenemyspentcontrolled_enemy_team_diff,
    wards_placed_team_ratio,
    wards_placed_role_diff,
    wards_placed_enemy_team_diff,
    wards_killed_team_ratio,
    wards_killed_role_diff,
    wards_killed_enemy_team_diff,
    dragon_advantage,
    dragon_deficit,
    rift_herald_advantage,
    rift_herald_deficit,
    horde_advantage,
    horde_deficit,
    baron_advantage,
    baron_deficit
)
WITH
participant_dim AS (
    SELECT
        matchid,
        participantid,
        any(teamid) AS teamid,
        any(toString(teamposition)) AS teamposition
    FROM game_data_filtered.participant_stats
    GROUP BY
        matchid,
        participantid
),

base_p_stats AS (
    SELECT
        ps.matchid,
        ps.frame_timestamp,
        pd.teamid,
        ps.participantid,
        pd.teamposition,
        toUInt8(if(pd.teamid = 100, 200, 100)) AS opponent_teamid,
        ps.totalgold,
        ps.xp,
        ps.minionskilled + ps.jungleminionskilled AS totalfarm,
        ps.totaldamagedonetochampions,
        ps.totaldamagetaken,
        ps.timeenemyspentcontrolled
    FROM game_data_filtered.tl_participant_stats AS ps
    ANY LEFT JOIN participant_dim AS pd
        ON
            ps.matchid = pd.matchid
            AND ps.participantid = pd.participantid
),

discrete_events AS (
    SELECT
        matchid,
        frame_timestamp,
        participantid,
        toUInt64(sum(kills)) AS kills_minute,
        toUInt64(sum(deaths)) AS deaths_minute,
        toUInt64(sum(assists)) AS assists_minute,
        toUInt64(sum(tower_takedowns)) AS tower_takedowns_minute,
        toUInt64(sum(elite_monster_takedowns_dragon)) AS dragon_minute,
        toUInt64(sum(elite_monster_takedowns_rift_herald))
            AS rift_herald_minute,
        toUInt64(sum(elite_monster_takedowns_horde)) AS horde_minute,
        toUInt64(sum(elite_monster_takedowns_baron)) AS baron_minute,
        toUInt64(sum(wards_placed)) AS wards_placed_minute,
        toUInt64(sum(wards_killed)) AS wards_killed_minute
    FROM game_data_filtered.tl_participant_discrete_events
    GROUP BY
        matchid,
        frame_timestamp,
        participantid
),

base AS (
    SELECT
        bps.*,
        toUInt64(sum(coalesce(
            de.kills_minute, toUInt64(0)
        )) OVER player_w) AS kills,
        toUInt64(sum(coalesce(
            de.deaths_minute, toUInt64(0)
        )) OVER player_w) AS deaths,
        toUInt64(sum(coalesce(
            de.assists_minute, toUInt64(0)
        )) OVER player_w) AS assists,
        toUInt64(sum(coalesce(
            de.tower_takedowns_minute, toUInt64(0)
        )) OVER player_w) AS tower_takedowns,
        toUInt64(sum(coalesce(
            de.dragon_minute, toUInt64(0)
        )) OVER player_w) AS dragon,
        toUInt64(sum(coalesce(
            de.rift_herald_minute, toUInt64(0)
        )) OVER player_w) AS rift_herald,
        toUInt64(sum(coalesce(
            de.horde_minute, toUInt64(0)
        )) OVER player_w) AS horde,
        toUInt64(sum(coalesce(
            de.baron_minute, toUInt64(0)
        )) OVER player_w) AS baron,
        toUInt64(sum(coalesce(
            de.wards_placed_minute, toUInt64(0)
        )) OVER player_w) AS wards_placed,
        toUInt64(sum(coalesce(
            de.wards_killed_minute, toUInt64(0)
        )) OVER player_w) AS wards_killed
    FROM base_p_stats AS bps
    ANY LEFT JOIN discrete_events AS de
        USING (matchid, frame_timestamp, participantid)
    WINDOW player_w AS (
        PARTITION BY matchid, participantid
        ORDER BY frame_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
),

role_base AS (
    SELECT
        matchid,
        frame_timestamp,
        teamid,
        teamposition,
        totalgold,
        xp,
        totalfarm,
        kills,
        deaths,
        assists,
        totaldamagedonetochampions,
        totaldamagetaken,
        timeenemyspentcontrolled,
        wards_placed,
        wards_killed
    FROM base
    WHERE teamposition != '' AND teamposition != 'UNKNOWN'
),

team_context AS (
    SELECT
        matchid,
        frame_timestamp,
        teamid,
        toFloat32(sum(toFloat32(totalgold))) AS totalgold_team_total,
        toFloat32(avg(toFloat32(totalgold))) AS totalgold_team_avg,
        toFloat32(sum(toFloat32(xp))) AS xp_team_total,
        toFloat32(avg(toFloat32(xp))) AS xp_team_avg,
        toFloat32(sum(toFloat32(totalfarm))) AS totalfarm_team_total,
        toFloat32(avg(toFloat32(totalfarm))) AS totalfarm_team_avg,
        toFloat32(sum(toFloat32(kills))) AS kills_team_total,
        toFloat32(avg(toFloat32(kills))) AS kills_team_avg,
        toFloat32(avg(toFloat32(deaths))) AS deaths_team_avg,
        toFloat32(sum(toFloat32(assists))) AS assists_team_total,
        toFloat32(sum(toFloat32(totaldamagedonetochampions)))
            AS totaldamagedonetochampions_team_total,
        toFloat32(avg(toFloat32(totaldamagedonetochampions)))
            AS totaldamagedonetochampions_team_avg,
        toFloat32(sum(toFloat32(totaldamagetaken)))
            AS totaldamagetaken_team_total,
        toFloat32(avg(toFloat32(totaldamagetaken)))
            AS totaldamagetaken_team_avg,
        toFloat32(sum(toFloat32(timeenemyspentcontrolled)))
            AS timeenemyspentcontrolled_team_total,
        toFloat32(avg(toFloat32(timeenemyspentcontrolled)))
            AS timeenemyspentcontrolled_team_avg,
        toFloat32(sum(toFloat32(wards_placed))) AS wards_placed_team_total,
        toFloat32(avg(toFloat32(wards_placed))) AS wards_placed_team_avg,
        toFloat32(sum(toFloat32(wards_killed))) AS wards_killed_team_total,
        toFloat32(avg(toFloat32(wards_killed))) AS wards_killed_team_avg,
        toFloat32(sum(toFloat32(dragon))) AS dragon_team_total,
        toFloat32(sum(toFloat32(rift_herald))) AS rift_herald_team_total,
        toFloat32(sum(toFloat32(horde))) AS horde_team_total,
        toFloat32(sum(toFloat32(baron))) AS baron_team_total
    FROM base
    GROUP BY
        matchid,
        frame_timestamp,
        teamid
)

SELECT
    b.matchid,
    b.frame_timestamp,
    b.teamid,
    b.participantid,
    if(
        coalesce(own_tc.totalgold_team_total, toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.totalgold) / own_tc.totalgold_team_total
    ) AS totalgold_team_ratio,
    toInt64(b.totalgold) - coalesce(
        toInt64(role_opponent.totalgold), toInt64(0)
    ) AS totalgold_role_diff,
    if(
        coalesce(toFloat32(role_opponent.totalgold), toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.totalgold) / toFloat32(role_opponent.totalgold)
    ) AS totalgold_role_ratio,
    toFloat32(b.totalgold) - coalesce(
        enemy_tc.totalgold_team_avg, toFloat32(0)
    ) AS totalgold_enemy_team_diff,
    if(
        coalesce(own_tc.xp_team_total, toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.xp) / own_tc.xp_team_total
    ) AS xp_team_ratio,
    toInt64(b.xp) - coalesce(toInt64(role_opponent.xp), toInt64(0))
        AS xp_role_diff,
    if(
        coalesce(toFloat32(role_opponent.xp), toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.xp) / toFloat32(role_opponent.xp)
    ) AS xp_role_ratio,
    toFloat32(b.xp) - coalesce(enemy_tc.xp_team_avg, toFloat32(0))
        AS xp_enemy_team_diff,
    if(
        coalesce(own_tc.totalfarm_team_total, toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.totalfarm) / own_tc.totalfarm_team_total
    ) AS totalfarm_team_ratio,
    toInt64(b.totalfarm) - coalesce(
        toInt64(role_opponent.totalfarm), toInt64(0)
    ) AS totalfarm_role_diff,
    if(
        coalesce(toFloat32(role_opponent.totalfarm), toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.totalfarm) / toFloat32(role_opponent.totalfarm)
    ) AS totalfarm_role_ratio,
    toFloat32(b.totalfarm) - coalesce(
        enemy_tc.totalfarm_team_avg, toFloat32(0)
    ) AS totalfarm_enemy_team_diff,
    if(
        coalesce(own_tc.kills_team_total, toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.kills) / own_tc.kills_team_total
    ) AS kills_team_ratio,
    toInt64(b.kills) - coalesce(toInt64(role_opponent.kills), toInt64(0))
        AS kills_role_diff,
    toFloat32(b.kills) - coalesce(enemy_tc.kills_team_avg, toFloat32(0))
        AS kills_enemy_team_diff,
    toInt64(b.deaths) - coalesce(toInt64(role_opponent.deaths), toInt64(0))
        AS deaths_role_diff,
    toFloat32(b.deaths) - coalesce(enemy_tc.deaths_team_avg, toFloat32(0))
        AS deaths_enemy_team_diff,
    if(
        coalesce(own_tc.assists_team_total, toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.assists) / own_tc.assists_team_total
    ) AS assists_team_ratio,
    toInt64(b.assists) - coalesce(
        toInt64(role_opponent.assists), toInt64(0)
    ) AS assists_role_diff,
    if(
        coalesce(
            own_tc.totaldamagedonetochampions_team_total, toFloat32(0)
        ) = 0,
        toFloat32(0),
        toFloat32(b.totaldamagedonetochampions)
        / own_tc.totaldamagedonetochampions_team_total
    ) AS totaldamagedonetochampions_team_ratio,
    toInt64(b.totaldamagedonetochampions) - coalesce(
        toInt64(role_opponent.totaldamagedonetochampions), toInt64(0)
    ) AS totaldamagedonetochampions_role_diff,
    toFloat32(b.totaldamagedonetochampions) - coalesce(
        enemy_tc.totaldamagedonetochampions_team_avg, toFloat32(0)
    ) AS totaldamagedonetochampions_enemy_team_diff,
    if(
        coalesce(own_tc.totaldamagetaken_team_total, toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.totaldamagetaken) / own_tc.totaldamagetaken_team_total
    ) AS totaldamagetaken_team_ratio,
    toInt64(b.totaldamagetaken) - coalesce(
        toInt64(role_opponent.totaldamagetaken), toInt64(0)
    ) AS totaldamagetaken_role_diff,
    toFloat32(b.totaldamagetaken) - coalesce(
        enemy_tc.totaldamagetaken_team_avg, toFloat32(0)
    ) AS totaldamagetaken_enemy_team_diff,
    if(
        coalesce(
            own_tc.timeenemyspentcontrolled_team_total, toFloat32(0)
        ) = 0,
        toFloat32(0),
        toFloat32(b.timeenemyspentcontrolled)
        / own_tc.timeenemyspentcontrolled_team_total
    ) AS timeenemyspentcontrolled_team_ratio,
    toInt64(b.timeenemyspentcontrolled) - coalesce(
        toInt64(role_opponent.timeenemyspentcontrolled), toInt64(0)
    ) AS timeenemyspentcontrolled_role_diff,
    toFloat32(b.timeenemyspentcontrolled) - coalesce(
        enemy_tc.timeenemyspentcontrolled_team_avg, toFloat32(0)
    ) AS timeenemyspentcontrolled_enemy_team_diff,
    if(
        coalesce(own_tc.wards_placed_team_total, toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.wards_placed) / own_tc.wards_placed_team_total
    ) AS wards_placed_team_ratio,
    toInt64(b.wards_placed) - coalesce(
        toInt64(role_opponent.wards_placed), toInt64(0)
    ) AS wards_placed_role_diff,
    toFloat32(b.wards_placed) - coalesce(
        enemy_tc.wards_placed_team_avg, toFloat32(0)
    ) AS wards_placed_enemy_team_diff,
    if(
        coalesce(own_tc.wards_killed_team_total, toFloat32(0)) = 0,
        toFloat32(0),
        toFloat32(b.wards_killed) / own_tc.wards_killed_team_total
    ) AS wards_killed_team_ratio,
    toInt64(b.wards_killed) - coalesce(
        toInt64(role_opponent.wards_killed), toInt64(0)
    ) AS wards_killed_role_diff,
    toFloat32(b.wards_killed) - coalesce(
        enemy_tc.wards_killed_team_avg, toFloat32(0)
    ) AS wards_killed_enemy_team_diff,
    toUInt8(
        coalesce(own_tc.dragon_team_total, toFloat32(0))
        > coalesce(enemy_tc.dragon_team_total, toFloat32(0))
    ) AS dragon_advantage,
    toUInt8(
        coalesce(own_tc.dragon_team_total, toFloat32(0))
        < coalesce(enemy_tc.dragon_team_total, toFloat32(0))
    ) AS dragon_deficit,
    toUInt8(
        coalesce(own_tc.rift_herald_team_total, toFloat32(0))
        > coalesce(enemy_tc.rift_herald_team_total, toFloat32(0))
    ) AS rift_herald_advantage,
    toUInt8(
        coalesce(own_tc.rift_herald_team_total, toFloat32(0))
        < coalesce(enemy_tc.rift_herald_team_total, toFloat32(0))
    ) AS rift_herald_deficit,
    toUInt8(
        coalesce(own_tc.horde_team_total, toFloat32(0))
        > coalesce(enemy_tc.horde_team_total, toFloat32(0))
    ) AS horde_advantage,
    toUInt8(
        coalesce(own_tc.horde_team_total, toFloat32(0))
        < coalesce(enemy_tc.horde_team_total, toFloat32(0))
    ) AS horde_deficit,
    toUInt8(
        coalesce(own_tc.baron_team_total, toFloat32(0))
        > coalesce(enemy_tc.baron_team_total, toFloat32(0))
    ) AS baron_advantage,
    toUInt8(
        coalesce(own_tc.baron_team_total, toFloat32(0))
        < coalesce(enemy_tc.baron_team_total, toFloat32(0))
    ) AS baron_deficit
FROM base AS b
ANY LEFT JOIN team_context AS own_tc
    USING (matchid, frame_timestamp, teamid)
ANY LEFT JOIN team_context AS enemy_tc
    ON
        b.matchid = enemy_tc.matchid
        AND b.frame_timestamp = enemy_tc.frame_timestamp
        AND b.opponent_teamid = enemy_tc.teamid
ANY LEFT JOIN role_base AS role_opponent
    ON
        b.matchid = role_opponent.matchid
        AND b.frame_timestamp = role_opponent.frame_timestamp
        AND b.opponent_teamid = role_opponent.teamid
        AND b.teamposition = role_opponent.teamposition;
