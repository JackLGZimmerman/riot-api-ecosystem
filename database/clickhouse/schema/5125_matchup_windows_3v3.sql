CREATE TABLE IF NOT EXISTS game_data_filtered.matchup_windows_3v3
(
    matchid UInt64,
    windowid UInt16,
    left_teamid UInt8,
    right_teamid UInt8,
    left_key String,
    right_key String,
    left_champions Array (Int16),
    right_champions Array (Int16),
    left_team_positions Array (LowCardinality (String)),
    right_team_positions Array (LowCardinality (String)),
    left_metric_maps Array (Map (String, Int64)),
    right_metric_maps Array (Map (String, Int64)),
    left_flag_maps Array (Map (String, UInt8)),
    right_flag_maps Array (Map (String, UInt8)),
    left_metric_prefix Array (Map (String, Int64)),
    right_metric_prefix Array (Map (String, Int64)),
    left_flag_prefix Array (Map (String, UInt8)),
    right_flag_prefix Array (Map (String, UInt8))
)
ENGINE = MergeTree
ORDER BY (left_key, right_key, matchid, windowid);

CREATE VIEW IF NOT EXISTS game_data_filtered.v_matchup_windows_3v3_source AS
WITH team_top3 AS (
    SELECT
        matchid,
        teamid,
        arrayMap(champion_row -> toInt16(tupleElement(champion_row, 1)), top3_aligned)
            AS champions,
        arrayMap(champion_row -> tupleElement(champion_row, 4), top3_aligned)
            AS team_positions,
        arrayMap(champion_row -> tupleElement(champion_row, 5), top3_aligned)
            AS metric_maps,
        arrayMap(champion_row -> tupleElement(champion_row, 6), top3_aligned)
            AS flag_maps
    FROM (
        SELECT
            matchid,
            teamid,
            arraySort(champion_row -> tupleElement(champion_row, 1), top3_ranked)
                AS top3_aligned
        FROM (
            SELECT
                matchid,
                teamid,
                arraySlice(
                    arraySort(
                        champion_row -> tuple(
                            -toInt64(tupleElement(champion_row, 2)),
                            -toInt64(tupleElement(champion_row, 3)),
                            toInt64(tupleElement(champion_row, 1))
                        ),
                        groupArray(
                            tuple(
                                championid,
                                totaldamagedealttochampions,
                                kills,
                                toString(teamposition),
                                map(
                                    'summonerlevel',
                                    toInt64(coalesce(summonerlevel, 0)),
                                    'profileicon',
                                    toInt64(coalesce(profileicon, 0)),
                                    'championtransform',
                                    toInt64(coalesce(championtransform, 0)),
                                    'champlevel',
                                    toInt64(coalesce(champlevel, 0)),
                                    'champexperience',
                                    toInt64(coalesce(champexperience, 0)),
                                    'kills',
                                    toInt64(coalesce(kills, 0)),
                                    'deaths',
                                    toInt64(coalesce(deaths, 0)),
                                    'assists',
                                    toInt64(coalesce(assists, 0)),
                                    'doublekills',
                                    toInt64(coalesce(doublekills, 0)),
                                    'triplekills',
                                    toInt64(coalesce(triplekills, 0)),
                                    'quadrakills',
                                    toInt64(coalesce(quadrakills, 0)),
                                    'pentakills',
                                    toInt64(coalesce(pentakills, 0)),
                                    'killingsprees',
                                    toInt64(coalesce(killingsprees, 0)),
                                    'largestkillingspree',
                                    toInt64(coalesce(largestkillingspree, 0)),
                                    'largestmultikill',
                                    toInt64(coalesce(largestmultikill, 0)),
                                    'largestcriticalstrike',
                                    toInt64(coalesce(largestcriticalstrike, 0)),
                                    'goldearned',
                                    toInt64(coalesce(goldearned, 0)),
                                    'goldspent',
                                    toInt64(coalesce(goldspent, 0)),
                                    'consumablespurchased',
                                    toInt64(coalesce(consumablespurchased, 0)),
                                    'itemspurchased',
                                    toInt64(coalesce(itemspurchased, 0)),
                                    'item0',
                                    toInt64(coalesce(item0, 0)),
                                    'item1',
                                    toInt64(coalesce(item1, 0)),
                                    'item2',
                                    toInt64(coalesce(item2, 0)),
                                    'item3',
                                    toInt64(coalesce(item3, 0)),
                                    'item4',
                                    toInt64(coalesce(item4, 0)),
                                    'item5',
                                    toInt64(coalesce(item5, 0)),
                                    'item6',
                                    toInt64(coalesce(item6, 0)),
                                    'totaldamagedealt',
                                    toInt64(coalesce(totaldamagedealt, 0)),
                                    'totaldamagedealttochampions',
                                    toInt64(coalesce(totaldamagedealttochampions, 0)),
                                    'physicaldamagedealt',
                                    toInt64(coalesce(physicaldamagedealt, 0)),
                                    'physicaldamagedealttochampions',
                                    toInt64(
                                        coalesce(physicaldamagedealttochampions, 0)
                                    ),
                                    'magicdamagedealt',
                                    toInt64(coalesce(magicdamagedealt, 0)),
                                    'magicdamagedealttochampions',
                                    toInt64(coalesce(magicdamagedealttochampions, 0)),
                                    'truedamagedealt',
                                    toInt64(coalesce(truedamagedealt, 0)),
                                    'truedamagedealttochampions',
                                    toInt64(coalesce(truedamagedealttochampions, 0)),
                                    'damagedealttobuildings',
                                    toInt64(coalesce(damagedealttobuildings, 0)),
                                    'damagedealttoturrets',
                                    toInt64(coalesce(damagedealttoturrets, 0)),
                                    'damagedealttoobjectives',
                                    toInt64(coalesce(damagedealttoobjectives, 0)),
                                    'damagedealttoepicmonsters',
                                    toInt64(coalesce(damagedealttoepicmonsters, 0)),
                                    'totaldamagetaken',
                                    toInt64(coalesce(totaldamagetaken, 0)),
                                    'physicaldamagetaken',
                                    toInt64(coalesce(physicaldamagetaken, 0)),
                                    'magicdamagetaken',
                                    toInt64(coalesce(magicdamagetaken, 0)),
                                    'truedamagetaken',
                                    toInt64(coalesce(truedamagetaken, 0)),
                                    'damageselfmitigated',
                                    toInt64(coalesce(damageselfmitigated, 0)),
                                    'totalheal',
                                    toInt64(coalesce(totalheal, 0)),
                                    'totalhealsonteammates',
                                    toInt64(coalesce(totalhealsonteammates, 0)),
                                    'totalunitshealed',
                                    toInt64(coalesce(totalunitshealed, 0)),
                                    'totaldamageshieldedonteammates',
                                    toInt64(
                                        coalesce(totaldamageshieldedonteammates, 0)
                                    ),
                                    'timeccingothers',
                                    toInt64(coalesce(timeccingothers, 0)),
                                    'totaltimeccdealt',
                                    toInt64(coalesce(totaltimeccdealt, 0)),
                                    'totalminionskilled',
                                    toInt64(coalesce(totalminionskilled, 0)),
                                    'neutralminionskilled',
                                    toInt64(coalesce(neutralminionskilled, 0)),
                                    'totalallyjungleminionskilled',
                                    toInt64(coalesce(totalallyjungleminionskilled, 0)),
                                    'totalenemyjungleminionskilled',
                                    toInt64(coalesce(totalenemyjungleminionskilled, 0)),
                                    'baronkills',
                                    toInt64(coalesce(baronkills, 0)),
                                    'dragonkills',
                                    toInt64(coalesce(dragonkills, 0)),
                                    'inhibitorkills',
                                    toInt64(coalesce(inhibitorkills, 0)),
                                    'inhibitortakedowns',
                                    toInt64(coalesce(inhibitortakedowns, 0)),
                                    'inhibitorslost',
                                    toInt64(coalesce(inhibitorslost, 0)),
                                    'turretkills',
                                    toInt64(coalesce(turretkills, 0)),
                                    'turrettakedowns',
                                    toInt64(coalesce(turrettakedowns, 0)),
                                    'turretslost',
                                    toInt64(coalesce(turretslost, 0)),
                                    'objectivesstolen',
                                    toInt64(coalesce(objectivesstolen, 0)),
                                    'objectivesstolenassists',
                                    toInt64(coalesce(objectivesstolenassists, 0)),
                                    'visionscore',
                                    toInt64(coalesce(visionscore, 0)),
                                    'wardsplaced',
                                    toInt64(coalesce(wardsplaced, 0)),
                                    'wardskilled',
                                    toInt64(coalesce(wardskilled, 0)),
                                    'detectorwardsplaced',
                                    toInt64(coalesce(detectorwardsplaced, 0)),
                                    'sightwardsboughtingame',
                                    toInt64(coalesce(sightwardsboughtingame, 0)),
                                    'visionwardsboughtingame',
                                    toInt64(coalesce(visionwardsboughtingame, 0)),
                                    'visionclearedpings',
                                    toInt64(coalesce(visionclearedpings, 0)),
                                    'summoner1id',
                                    toInt64(coalesce(summoner1id, 0)),
                                    'summoner2id',
                                    toInt64(coalesce(summoner2id, 0)),
                                    'summoner1casts',
                                    toInt64(coalesce(summoner1casts, 0)),
                                    'summoner2casts',
                                    toInt64(coalesce(summoner2casts, 0)),
                                    'spell1casts',
                                    toInt64(coalesce(spell1casts, 0)),
                                    'spell2casts',
                                    toInt64(coalesce(spell2casts, 0)),
                                    'spell3casts',
                                    toInt64(coalesce(spell3casts, 0)),
                                    'spell4casts',
                                    toInt64(coalesce(spell4casts, 0)),
                                    'rolebounditem',
                                    toInt64(coalesce(rolebounditem, 0)),
                                    'bountylevel',
                                    toInt64(coalesce(bountylevel, 0)),
                                    'timeplayed',
                                    toInt64(coalesce(timeplayed, 0)),
                                    'totaltimespentdead',
                                    toInt64(coalesce(totaltimespentdead, 0)),
                                    'longesttimespentliving',
                                    toInt64(coalesce(longesttimespentliving, 0)),
                                    'allinpings',
                                    toInt64(coalesce(allinpings, 0)),
                                    'assistmepings',
                                    toInt64(coalesce(assistmepings, 0)),
                                    'basicpings',
                                    toInt64(coalesce(basicpings, 0)),
                                    'commandpings',
                                    toInt64(coalesce(commandpings, 0)),
                                    'dangerpings',
                                    toInt64(coalesce(dangerpings, 0)),
                                    'enemymissingpings',
                                    toInt64(coalesce(enemymissingpings, 0)),
                                    'enemyvisionpings',
                                    toInt64(coalesce(enemyvisionpings, 0)),
                                    'getbackpings',
                                    toInt64(coalesce(getbackpings, 0)),
                                    'holdpings',
                                    toInt64(coalesce(holdpings, 0)),
                                    'needvisionpings',
                                    toInt64(coalesce(needvisionpings, 0)),
                                    'onmywaypings',
                                    toInt64(coalesce(onmywaypings, 0)),
                                    'pushpings',
                                    toInt64(coalesce(pushpings, 0)),
                                    'retreatpings',
                                    toInt64(coalesce(retreatpings, 0)),
                                    'unrealkills',
                                    toInt64(coalesce(unrealkills, 0))
                                ),
                                map(
                                    'win',
                                    toUInt8(win > 0),
                                    'gameendedinearlysurrender',
                                    toUInt8(gameendedinearlysurrender > 0),
                                    'gameendedinsurrender',
                                    toUInt8(gameendedinsurrender > 0),
                                    'teamearlysurrendered',
                                    toUInt8(teamearlysurrendered > 0),
                                    'firstbloodkill',
                                    toUInt8(firstbloodkill > 0),
                                    'firstbloodassist',
                                    toUInt8(firstbloodassist > 0),
                                    'firsttowerkill',
                                    toUInt8(firsttowerkill > 0),
                                    'firsttowerassist',
                                    toUInt8(firsttowerassist > 0)
                                )
                            )
                        )
                    ),
                    1,
                    3
                ) AS top3_ranked
            FROM game_data_filtered.participant_stats
            GROUP BY matchid, teamid
        )
        WHERE length(top3_ranked) > 0
    )
),

match_pairs AS (
    SELECT
        matchid,
        toUInt16(1) AS windowid,
        tupleElement(teams[1], 1) AS teamid_1,
        tupleElement(teams[2], 1) AS teamid_2,
        tupleElement(teams[1], 2) AS champions_1,
        tupleElement(teams[2], 2) AS champions_2,
        tupleElement(teams[1], 3) AS team_positions_1,
        tupleElement(teams[2], 3) AS team_positions_2,
        tupleElement(teams[1], 4) AS metric_maps_1,
        tupleElement(teams[2], 4) AS metric_maps_2,
        tupleElement(teams[1], 5) AS flag_maps_1,
        tupleElement(teams[2], 5) AS flag_maps_2
    FROM (
        SELECT
            matchid,
            arraySort(
                side_row -> tupleElement(side_row, 1),
                groupArray(
                    tuple(
                        teamid,
                        champions,
                        team_positions,
                        metric_maps,
                        flag_maps
                    )
                )
            ) AS teams
        FROM team_top3
        GROUP BY matchid
    )
    WHERE
        length(teams) = 2
        AND length(tupleElement(teams[1], 2)) >= 3
        AND length(tupleElement(teams[2], 2)) >= 3
),

prepared AS (
    SELECT
        matchid,
        windowid,
        if(needs_swap, teamid_2, teamid_1) AS left_teamid,
        if(needs_swap, teamid_1, teamid_2) AS right_teamid,
        if(needs_swap, champions_2, champions_1) AS left_champions,
        if(needs_swap, champions_1, champions_2) AS right_champions,
        if(needs_swap, team_positions_2, team_positions_1) AS left_team_positions,
        if(needs_swap, team_positions_1, team_positions_2) AS right_team_positions,
        if(needs_swap, metric_maps_2, metric_maps_1) AS left_metric_maps,
        if(needs_swap, metric_maps_1, metric_maps_2) AS right_metric_maps,
        if(needs_swap, flag_maps_2, flag_maps_1) AS left_flag_maps,
        if(needs_swap, flag_maps_1, flag_maps_2) AS right_flag_maps
    FROM (
        SELECT
            matchid,
            windowid,
            teamid_1,
            teamid_2,
            champions_1,
            champions_2,
            team_positions_1,
            team_positions_2,
            metric_maps_1,
            metric_maps_2,
            flag_maps_1,
            flag_maps_2,
            champions_2 < champions_1 AS needs_swap
        FROM match_pairs
    )
)

SELECT
    matchid,
    windowid,
    left_teamid,
    right_teamid,
    left_champions,
    right_champions,
    left_team_positions,
    right_team_positions,
    left_metric_maps,
    right_metric_maps,
    left_flag_maps,
    right_flag_maps,
    arrayStringConcat(arrayMap(champion -> toString(champion), left_champions), ',')
        AS left_key,
    arrayStringConcat(arrayMap(champion -> toString(champion), right_champions), ',')
        AS right_key,
    [
        left_metric_maps[1],
        mapAdd(left_metric_maps[1], left_metric_maps[2]),
        mapAdd(left_metric_maps[1], left_metric_maps[2], left_metric_maps[3])
    ] AS left_metric_prefix,
    [
        right_metric_maps[1],
        mapAdd(right_metric_maps[1], right_metric_maps[2]),
        mapAdd(right_metric_maps[1], right_metric_maps[2], right_metric_maps[3])
    ] AS right_metric_prefix,
    [
        left_flag_maps[1],
        mapApply(
            (k, v) -> (k, toUInt8(v > 0)),
            mapAdd(left_flag_maps[1], left_flag_maps[2])
        ),
        mapApply(
            (k, v) -> (k, toUInt8(v > 0)),
            mapAdd(left_flag_maps[1], left_flag_maps[2], left_flag_maps[3])
        )
    ] AS left_flag_prefix,
    [
        right_flag_maps[1],
        mapApply(
            (k, v) -> (k, toUInt8(v > 0)),
            mapAdd(right_flag_maps[1], right_flag_maps[2])
        ),
        mapApply(
            (k, v) -> (k, toUInt8(v > 0)),
            mapAdd(right_flag_maps[1], right_flag_maps[2], right_flag_maps[3])
        )
    ] AS right_flag_prefix
FROM prepared;

-- Manual refresh:
-- TRUNCATE TABLE game_data_filtered.matchup_windows_3v3;
-- INSERT INTO game_data_filtered.matchup_windows_3v3
-- SELECT * FROM game_data_filtered.v_matchup_windows_3v3_source;
