-- noqa: disable=AL09,LT02,LT05,RF02,RF03,ST09
--
-- Build-sibling 2vx synergy priors. This is derived from the exact build-level
-- 2vx table so the sibling rows preserve the same train-only leakage boundary.

TRUNCATE TABLE game_data_filtered.synergy_2vx_build_group;

INSERT INTO game_data_filtered.synergy_2vx_build_group
SELECT
    'train' AS split,
    tupleElement(p1, 1) AS championid_1,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(championid_1),
        ''
    ) AS championname_1,
    tupleElement(p1, 2) AS teamposition_1,
    tupleElement(p1, 3) AS build_group_1,
    tupleElement(p2, 1) AS championid_2,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(championid_2),
        ''
    ) AS championname_2,
    tupleElement(p2, 2) AS teamposition_2,
    tupleElement(p2, 3) AS build_group_2,
    sum(matchups) AS matchups,
    sum(wins) AS wins,
    sum(losses) AS losses,
    toFloat32(wins / matchups) AS win_rate
FROM (
    SELECT
        if(g1 <= g2, g1, g2) AS p1,
        if(g1 <= g2, g2, g1) AS p2,
        matchups,
        wins,
        losses
    FROM (
        SELECT
            (
                championid_1,
                teamposition_1,
                multiIf(
                    build_1 IN ('ability_power', 'ap_off_tank'), 'ap',
                    build_1 IN ('attack_damage', 'ad_off_tank'), 'ad',
                    build_1 IN ('ar_tank', 'mr_tank'), 'tank',
                    build_1 IN ('utility_enchanter', 'utility_protection'), 'utility',
                    build_1
                )
            ) AS g1,
            (
                championid_2,
                teamposition_2,
                multiIf(
                    build_2 IN ('ability_power', 'ap_off_tank'), 'ap',
                    build_2 IN ('attack_damage', 'ad_off_tank'), 'ad',
                    build_2 IN ('ar_tank', 'mr_tank'), 'tank',
                    build_2 IN ('utility_enchanter', 'utility_protection'), 'utility',
                    build_2
                )
            ) AS g2,
            matchups,
            wins,
            losses
        FROM game_data_filtered.synergy_2vx
        WHERE split = 'train'
    )
)
GROUP BY
    championid_1, teamposition_1, build_group_1,
    championid_2, teamposition_2, build_group_2;
