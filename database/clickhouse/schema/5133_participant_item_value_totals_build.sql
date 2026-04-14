TRUNCATE TABLE game_data_filtered.participant_item_value_totals;

INSERT INTO game_data_filtered.participant_item_value_totals
(
    matchid, teamid, participantid, puuid, championid, teamposition,
    attack_damage, ability_power, lethality, on_hit,
    crit, tank, off_tank, utility,
    highest_value, highest_value_label
)
SELECT
    matchid,
    teamid,
    participantid,
    puuid,
    championid,
    teamposition,
    attack_damage,
    ability_power,
    lethality,
    on_hit,
    crit,
    tank,
    off_tank,
    utility,

    greatest(
        attack_damage, ability_power, lethality, on_hit,
        crit, tank, off_tank, utility
    ) AS highest_value,

    multiIf(
        attack_damage = highest_value, 'attack_damage',
        ability_power = highest_value, 'ability_power',
        lethality = highest_value, 'lethality',
        on_hit = highest_value, 'on_hit',
        crit = highest_value, 'crit',
        tank = highest_value, 'tank',
        off_tank = highest_value, 'off_tank',
        'utility'
    ) AS highest_value_label

FROM (
    SELECT
        matchid,
        teamid,
        participantid,
        puuid,
        championid,
        toString(teamposition) AS teamposition,

        arraySum(arr -> arr .1, item_vals) AS attack_damage,
        arraySum(arr -> arr .2, item_vals) AS ability_power,
        arraySum(arr -> arr .3, item_vals) AS lethality,
        arraySum(arr -> arr .4, item_vals) AS on_hit,
        arraySum(arr -> arr .5, item_vals) AS crit,
        arraySum(arr -> arr .6, item_vals) AS tank,
        arraySum(arr -> arr .7, item_vals) AS off_tank,
        arraySum(arr -> arr .8, item_vals) AS utility
    FROM (
        SELECT
            matchid,
            teamid,
            participantid,
            puuid,
            [
                dictGet(
                    'game_data.item_value_map_dict',
                    (
                        'attack_damage',
                        'ability_power',
                        'lethality',
                        'on_hit',
                        'crit',
                        'tank',
                        'off_tank',
                        'utility'
                    ),
                    item0
                ),
                dictGet(
                    'game_data.item_value_map_dict',
                    (
                        'attack_damage',
                        'ability_power',
                        'lethality',
                        'on_hit',
                        'crit',
                        'tank',
                        'off_tank',
                        'utility'
                    ),
                    item1
                ),
                dictGet(
                    'game_data.item_value_map_dict',
                    (
                        'attack_damage',
                        'ability_power',
                        'lethality',
                        'on_hit',
                        'crit',
                        'tank',
                        'off_tank',
                        'utility'
                    ),
                    item2
                ),
                dictGet(
                    'game_data.item_value_map_dict',
                    (
                        'attack_damage',
                        'ability_power',
                        'lethality',
                        'on_hit',
                        'crit',
                        'tank',
                        'off_tank',
                        'utility'
                    ),
                    item3
                ),
                dictGet(
                    'game_data.item_value_map_dict',
                    (
                        'attack_damage',
                        'ability_power',
                        'lethality',
                        'on_hit',
                        'crit',
                        'tank',
                        'off_tank',
                        'utility'
                    ),
                    item4
                ),
                dictGet(
                    'game_data.item_value_map_dict',
                    (
                        'attack_damage',
                        'ability_power',
                        'lethality',
                        'on_hit',
                        'crit',
                        'tank',
                        'off_tank',
                        'utility'
                    ),
                    item5
                ),
                dictGet(
                    'game_data.item_value_map_dict',
                    (
                        'attack_damage',
                        'ability_power',
                        'lethality',
                        'on_hit',
                        'crit',
                        'tank',
                        'off_tank',
                        'utility'
                    ),
                    item6
                )
            ] AS item_vals
        FROM game_data_filtered.participant_stats
    )
)
