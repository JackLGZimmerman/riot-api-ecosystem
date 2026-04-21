-- Populate game_data_filtered.participant_item_value_totals.
-- Requires game_data_filtered.participant_stats to be populated (run 5003 first,
-- or run this standalone after a schema rebuild via 5132).
--
-- item_value_map_dict now carries (championid, teamposition, itemid) as its
-- composite key.  Rows with NULL (championid, teamposition) in the source jsonl
-- collapse to the sentinel key (0, '') and act as the general fallback; rows
-- with populated (championid, teamposition) apply only to that specific pair.
-- The lookup below first tries the specific key, then falls back to (0, '').
--
-- Memory note: the GROUP BY builds an aggregation hash table keyed by
-- (matchid, teamid, participantid, puuid, championid, teamposition) which does
-- not fit under the 5 GiB container cap.  Pause background merges, drop caches,
-- and enable disk-spilling aggregation so the GROUP BY can overflow to /tmp
-- instead of OOMing.

SYSTEM STOP MERGES;
SYSTEM DROP MARK CACHE;
SYSTEM DROP UNCOMPRESSED CACHE;
SYSTEM DROP COMPILED EXPRESSION CACHE;
SYSTEM JEMALLOC PURGE;

TRUNCATE TABLE game_data_filtered.participant_item_value_totals;

INSERT INTO game_data_filtered.participant_item_value_totals
(
    matchid, teamid, participantid, puuid, championid, teamposition,
    attack_damage, ability_power, lethality, on_hit, crit,
    utility_enchanter, utility_protection,
    ar_tank, mr_tank, ad_off_tank, ap_off_tank,
    highest_value, highest_value_label
)
WITH item_stats AS (
    SELECT
        ps.matchid,
        ps.teamid,
        ps.participantid,
        ps.puuid,
        ps.championid,
        ps.teamposition,
        sum(ps.v .1) AS attack_damage,
        sum(ps.v .2) AS ability_power,
        sum(ps.v .3) AS lethality,
        sum(ps.v .4) AS on_hit,
        sum(ps.v .5) AS crit,
        sum(ps.v .6) AS utility_enchanter,
        sum(ps.v .7) AS utility_protection,
        sum(ps.v .8) AS ar_tank,
        sum(ps.v .9) AS mr_tank,
        sum(ps.v .10) AS ad_off_tank,
        sum(ps.v .11) AS ap_off_tank
    FROM (
        SELECT
            matchid,
            teamid,
            participantid,
            puuid,
            championid,
            teamposition,
            item_id,
            if(
                dictHas(
                    'game_data.item_value_map_dict',
                    (toInt32(COALESCE(championid, 0)), teamposition, item_id)
                ),
                dictGet(
                    'game_data.item_value_map_dict',
                    (
                        'attack_damage', 'ability_power', 'lethality', 'on_hit', 'crit',
                        'utility_enchanter', 'utility_protection',
                        'ar_tank', 'mr_tank', 'ad_off_tank', 'ap_off_tank'
                    ),
                    (toInt32(COALESCE(championid, 0)), teamposition, item_id)
                ),
                dictGet(
                    'game_data.item_value_map_dict',
                    (
                        'attack_damage', 'ability_power', 'lethality', 'on_hit', 'crit',
                        'utility_enchanter', 'utility_protection',
                        'ar_tank', 'mr_tank', 'ad_off_tank', 'ap_off_tank'
                    ),
                    (toInt32(0), '', item_id)
                )
            ) AS v
        FROM (
            SELECT
                matchid,
                teamid,
                participantid,
                puuid,
                championid,
                toString(teamposition) AS teamposition,
                arrayJoin([
                    toUInt32(item0), toUInt32(item1), toUInt32(item2),
                    toUInt32(item3), toUInt32(item4), toUInt32(item5), toUInt32(item6)
                ]) AS item_id
            FROM game_data_filtered.participant_stats
        )
    ) AS ps
    GROUP BY
        ps.matchid,
        ps.teamid,
        ps.participantid,
        ps.puuid,
        ps.championid,
        ps.teamposition
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
    utility_enchanter,
    utility_protection,
    ar_tank,
    mr_tank,
    ad_off_tank,
    ap_off_tank,
    greatest(
        attack_damage, ability_power, lethality, on_hit, crit,
        utility_enchanter, utility_protection,
        ar_tank, mr_tank, ad_off_tank, ap_off_tank
    ) AS highest_value,
    multiIf(
        highest_value = 0, 'none',
        crit = highest_value, 'crit',
        lethality = highest_value, 'lethality',
        utility_enchanter = highest_value, 'utility_enchanter',
        utility_protection = highest_value, 'utility_protection',
        ar_tank = highest_value, 'ar_tank',
        mr_tank = highest_value, 'mr_tank',
        ad_off_tank = highest_value, 'ad_off_tank',
        ap_off_tank = highest_value, 'ap_off_tank',
        on_hit = highest_value, 'on_hit',
        ability_power = highest_value, 'ability_power',
        'attack_damage'
    ) AS highest_value_label
FROM item_stats
SETTINGS
    max_threads = 1,
    max_block_size = 4096,
    max_insert_block_size = 16384,
    max_bytes_before_external_group_by = 1500000000,
    max_bytes_ratio_before_external_group_by = 0,
    distributed_aggregation_memory_efficient = 1;

SYSTEM START MERGES;
