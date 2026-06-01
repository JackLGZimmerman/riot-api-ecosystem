-- noqa: disable=PRS,RF01,AL09
-- Filter pipeline: population in dependency order.
--
-- Base population: the latest season in game_data.info is selected first, then
-- filter 14 (gameduration > 990 s) is applied as an explicit pre-stage. All
-- downstream stages SEMI JOIN filter_stg_f14_long_games instead of scanning
-- game_data.info, so no stage touches older seasons or short games.
--
-- Flow:
--   0.  f14_long_games         (pre-filter: latest season, gameduration > 990)
--   1a. player_winrates        (from game_data.players)
--   1b. team_flags             (from game_data.participant_stats SEMI JOIN f14)
--   1.  participant_flags      (stage 1 flags; SEMI JOIN f14)
--   1r. stage1_valid_matchids  (rollup of participant_flags)
--   2.  participant_labels     (build labels + f10 low_build_value;
--                               SEMI JOIN stage1_valid)
--   3f. game_flags             (final rollup)
--   3.  filter_result          (bitmask)
--
-- Stage 1 filters enabled:
--   f01 player_low_kda          KDA < 0.20  ((k+a)*10 < d*2)
--   f02 player_gold_spent       spent < 50% earned AND win = 0 (losses only)
--   f03 player_high_winrate     suspect (>40 games, lifetime WR > 70%) AND
--                               suffix WR of collected games (current onwards,
--                               ordered by gamecreation) >= 85%
--   f04 team_kills_to_deaths    team K/D < 0.40  (kills*5 < deaths*2)
--   f05 solo_carried            win=1 AND kills > 75% of team kills
--   f06 too_little_damage       non-UTILITY dmg share < 2%  (dmg*50 < team_dmg)
--   f07 low_minions_killed      non-UTILITY CS/min < 3.0  ((cs+ncs)*60 < time*3)
--   f08 team_non_utility_avg_cs_per_min gap > 2.0 below enemy
--   f09 team_non_utility_damage_to_champs ratio < 0.50  (team*2 < enemy)
--   f10 low_build_value         highest_value < 0.5 (stage-1-clean pool)
--   f11 unknown_teamposition    game includes any participant with UNKNOWN teamposition
--   f12 game_ruining_behavior   Riot-detected IGNB surrender / severe
--                               transgressor metadata
--   f13 was_severe_transgressor exact participant severe-transgressor flag
--   f14 caused_ignb_surrender   exact participant caused-IGNB-surrender flag
--   f15 team_ignb_surrendered   exact team-side IGNB surrender flag
--   f16 premade_ignb_causer     exact premade-with-IGNB-game-end-causer flag
--   f17 premade_severe          exact premade-with-severe-transgressor flag
--   f18 zero_spell_casts_loss   losing participant cast no champion spells
--   f20 zero_item_purchases_loss losing participant bought no items
--
-- Run with clickhouse-client --multiquery after 4000_filter_schema.sql.

-- =============================================================================
-- Pre-stage (f14): collect matchids from the latest season only, then apply
-- the long-game threshold (gameduration > 990 s). All subsequent stages
-- restrict to this set via SEMI JOIN.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_f14_long_games;

INSERT INTO game_data.filter_stg_f14_long_games (matchid)
SELECT i.matchid
FROM game_data.info AS i
WHERE
    i.season = (SELECT max(latest_i.season) FROM game_data.info AS latest_i)
    AND i.gameduration > 990;

-- =============================================================================
-- Stage 1a: player win rates (from game_data.players, one row per puuid)
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_player_winrates;

INSERT INTO game_data.filter_stg_player_winrates (puuid, wins, losses)
SELECT
    puuid,
    argMax(wins, updated_at) AS wins,
    argMax(losses, updated_at) AS losses
FROM game_data.players
GROUP BY puuid;

-- =============================================================================
-- Stage 1b: team flags with enemy-relative stats (latest-season long games only)
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_team_flags;

INSERT INTO game_data.filter_stg_team_flags
(
    matchid,
    teamid,
    team_kills,
    team_damage_to_champions,
    team_kills_to_deaths,
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy,
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy
)
WITH match_base AS (
    -- LoL matchdata uses team IDs 100 and 200. Aggregating both sides in one
    -- pass avoids the analyzer inlining a self-joined team_base CTE twice.
    SELECT
        ps.matchid,
        sumIf(ps.kills, ps.teamid = 100) AS team_100_kills,
        sumIf(ps.deaths, ps.teamid = 100) AS team_100_deaths,
        sumIf(ps.totaldamagedealttochampions, ps.teamid = 100)
            AS team_100_damage_to_champions,
        avgIf(
            (ps.totalminionskilled + ps.neutralminionskilled) * 60.0 / ps.timeplayed,
            ps.teamid = 100 AND ps.teamposition != 'UTILITY' AND ps.timeplayed > 0
        ) AS team_100_non_utility_avg_cs_per_min,
        sumIf(
            ps.totaldamagedealttochampions,
            ps.teamid = 100 AND ps.teamposition != 'UTILITY'
        ) AS team_100_non_utility_damage_to_champions,
        sumIf(ps.kills, ps.teamid = 200) AS team_200_kills,
        sumIf(ps.deaths, ps.teamid = 200) AS team_200_deaths,
        sumIf(ps.totaldamagedealttochampions, ps.teamid = 200)
            AS team_200_damage_to_champions,
        avgIf(
            (ps.totalminionskilled + ps.neutralminionskilled) * 60.0 / ps.timeplayed,
            ps.teamid = 200 AND ps.teamposition != 'UTILITY' AND ps.timeplayed > 0
        ) AS team_200_non_utility_avg_cs_per_min,
        sumIf(
            ps.totaldamagedealttochampions,
            ps.teamid = 200 AND ps.teamposition != 'UTILITY'
        ) AS team_200_non_utility_damage_to_champions
    FROM game_data.participant_stats_corrected AS ps
    SEMI JOIN game_data.filter_stg_f14_long_games AS f14 ON ps.matchid = f14.matchid
    GROUP BY ps.matchid
),

team_rows AS (
    SELECT
        matchid,
        arrayJoin([
            tuple(
                toUInt8(100),
                team_100_kills,
                team_100_deaths,
                team_100_damage_to_champions,
                team_100_non_utility_avg_cs_per_min,
                team_100_non_utility_damage_to_champions,
                team_200_non_utility_avg_cs_per_min,
                team_200_non_utility_damage_to_champions
            ),
            tuple(
                toUInt8(200),
                team_200_kills,
                team_200_deaths,
                team_200_damage_to_champions,
                team_200_non_utility_avg_cs_per_min,
                team_200_non_utility_damage_to_champions,
                team_100_non_utility_avg_cs_per_min,
                team_100_non_utility_damage_to_champions
            )
        ]) AS team_stats
    FROM match_base
)

SELECT
    matchid,
    tupleElement(team_stats, 1) AS teamid,
    tupleElement(team_stats, 2) AS team_kills,
    tupleElement(team_stats, 4) AS team_damage_to_champions,
    -- Team K/D < 0.40: kills * 5 < deaths * 2
    tupleElement(team_stats, 2) * 5 < tupleElement(team_stats, 3) * 2
        AS team_kills_to_deaths,
    -- CS/min gap > 2.0 below enemy
    tupleElement(team_stats, 7) - tupleElement(team_stats, 5) > 2.0
        AS team_non_utility_avg_cs_per_min_gt_1_0_below_enemy,
    -- Team dmg ratio < 0.50 (1/2): team < 0.50 * enemy -> team * 2 < enemy
    tupleElement(team_stats, 8) > 0
    AND tupleElement(team_stats, 6) * 2 < tupleElement(team_stats, 8)
        AS team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy
FROM team_rows;

-- =============================================================================
-- Stage 1c2: f03 player_high_winrate flags (latest-season long games only).
-- Identify suspect players (lifetime > 40 games AND WR > 70%), then within
-- each suspect player's collected long-games sorted by gamecreation ASC,
-- flag games from the earliest while the suffix WR (games from current row
-- onwards) is >= 85%. Equivalent to: trim earliest games one at a time
-- until the remaining-window WR drops below 85%.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_player_high_winrate_flags;

INSERT INTO game_data.filter_stg_player_high_winrate_flags
(
    matchid, teamid, participantid, player_high_winrate
)
WITH suspect_games AS (
    SELECT
        ps.matchid AS matchid,
        ps.teamid AS teamid,
        ps.participantid AS participantid,
        ps.puuid AS puuid,
        ps.win AS win,
        i.gamecreation AS gamecreation
    FROM game_data.participant_stats_corrected AS ps
    SEMI JOIN game_data.filter_stg_f14_long_games AS f14 ON ps.matchid = f14.matchid
    SEMI JOIN (
        SELECT puuid
        FROM game_data.filter_stg_player_winrates
        WHERE wins + losses > 40 AND wins * 100 > (wins + losses) * 70
    ) AS sp ON ps.puuid = sp.puuid
    ANY INNER JOIN game_data.info AS i ON ps.matchid = i.matchid
)

SELECT
    matchid,
    teamid,
    participantid,
    toUInt8(
        sum(win) OVER (
            PARTITION BY puuid ORDER BY gamecreation DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) * 100
        >= row_number() OVER (
            PARTITION BY puuid ORDER BY gamecreation DESC
        ) * 85
    ) AS player_high_winrate
FROM suspect_games;

-- =============================================================================
-- Stage 1: per-participant flags for all active filters
-- (latest-season long games only).
-- Only rows from the latest season with gameduration > 990 are inserted; older
-- seasons and short games never enter the staging tables.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_participant_flags;

INSERT INTO game_data.filter_stg_participant_flags
(
    matchid,
    teamid,
    participantid,
    player_low_kda,
    player_gold_spent,
    player_high_winrate,
    team_kills_to_deaths,
    solo_carried,
    too_little_damage,
    low_minions_killed,
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy,
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy,
    unknown_teamposition,
    game_ruining_behavior,
    was_severe_transgressor,
    caused_game_end_from_ignb_surrender,
    team_ignb_surrendered,
    was_premade_with_ignb_game_end_causer,
    was_premade_with_severe_transgressor,
    zero_spell_casts_loss,
    zero_item_purchases_loss
)
SELECT
    ps.matchid,
    ps.teamid,
    ps.participantid,
    -- KDA < 0.20: (k + a) * 10 < d * 2
    (ps.kills + ps.assists) * 10 < ps.deaths * 2 AS player_low_kda,
    -- Gold spent < 50% earned, losses only
    ps.win = 0
    AND ps.goldearned > 0
    AND ps.goldspent * 100 < ps.goldearned * 50
        AS player_gold_spent,
    COALESCE(hw.player_high_winrate, toUInt8(0)) AS player_high_winrate,
    tf.team_kills_to_deaths,
    ps.win = 1 AND tf.team_kills > 0 AND ps.kills * 100 > tf.team_kills * 75
        AS solo_carried,
    -- Non-UTILITY damage share < 2%: dmg * 50 < team_dmg
    ps.teamposition != 'UTILITY'
    AND tf.team_damage_to_champions > 0
    AND ps.totaldamagedealttochampions * 50 < tf.team_damage_to_champions
        AS too_little_damage,
    -- Non-UTILITY CS/min < 3.0: (cs + ncs) * 60 < timeplayed * 3
    ps.teamposition != 'UTILITY'
    AND ps.timeplayed > 0
    AND (ps.totalminionskilled + ps.neutralminionskilled) * 60 < ps.timeplayed * 3
        AS low_minions_killed,
    tf.team_non_utility_avg_cs_per_min_gt_1_0_below_enemy,
    tf.team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy,
    ps.teamposition = 'UNKNOWN' AS unknown_teamposition,
    (
        COALESCE(ps.gameendedinignbsurrender, toUInt8(0))
        OR COALESCE(ps.causedgameendfromignbsurrender, toUInt8(0))
        OR COALESCE(ps.teamignbsurrendered, toUInt8(0))
        OR COALESCE(ps.waspremadewithignbgameendcauser, toUInt8(0))
        OR COALESCE(ps.waspremadewithseveretransgressor, toUInt8(0))
        OR COALESCE(ps.wasseveretransgressor, toUInt8(0))
    ) AS game_ruining_behavior,
    COALESCE(ps.wasseveretransgressor, toUInt8(0)) AS was_severe_transgressor,
    COALESCE(ps.causedgameendfromignbsurrender, toUInt8(0))
        AS caused_game_end_from_ignb_surrender,
    COALESCE(ps.teamignbsurrendered, toUInt8(0)) AS team_ignb_surrendered,
    COALESCE(ps.waspremadewithignbgameendcauser, toUInt8(0))
        AS was_premade_with_ignb_game_end_causer,
    COALESCE(ps.waspremadewithseveretransgressor, toUInt8(0))
        AS was_premade_with_severe_transgressor,
    ps.win = 0
    AND toUInt32(ps.spell1casts) + toUInt32(ps.spell2casts)
        + toUInt32(ps.spell3casts) + toUInt32(ps.spell4casts) = 0
        AS zero_spell_casts_loss,
    ps.win = 0 AND ps.itemspurchased = 0 AS zero_item_purchases_loss
FROM game_data.participant_stats_corrected AS ps
SEMI JOIN game_data.filter_stg_f14_long_games AS f14 ON ps.matchid = f14.matchid
ANY LEFT JOIN game_data.filter_stg_team_flags AS tf
    ON ps.matchid = tf.matchid AND ps.teamid = tf.teamid
ANY LEFT JOIN game_data.filter_stg_player_high_winrate_flags AS hw
    ON
        ps.matchid = hw.matchid
        AND ps.teamid = hw.teamid
        AND ps.participantid = hw.participantid;

-- =============================================================================
-- Stage 1 rollup: matchids that pass every stage-1 filter.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_stage1_valid_matchids;

INSERT INTO game_data.filter_stg_stage1_valid_matchids (matchid)
SELECT matchid
FROM game_data.filter_stg_participant_flags
GROUP BY matchid
HAVING
    max(player_low_kda) = 0
    AND max(player_gold_spent) = 0
    AND max(player_high_winrate) = 0
    AND max(team_kills_to_deaths) = 0
    AND max(solo_carried) = 0
    AND max(too_little_damage) = 0
    AND max(low_minions_killed) = 0
    AND max(team_non_utility_avg_cs_per_min_gt_1_0_below_enemy) = 0
    AND max(team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy) = 0
    AND max(unknown_teamposition) = 0
    AND max(game_ruining_behavior) = 0
    AND max(was_severe_transgressor) = 0
    AND max(caused_game_end_from_ignb_surrender) = 0
    AND max(team_ignb_surrendered) = 0
    AND max(was_premade_with_ignb_game_end_causer) = 0
    AND max(was_premade_with_severe_transgressor) = 0
    AND max(zero_spell_casts_loss) = 0
    AND max(zero_item_purchases_loss) = 0;

-- =============================================================================
-- Stage 2: build labels + f10 low_build_value detection over stage-1-clean games.
-- Threshold for low_build_value: highest_value < 0.5.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_participant_labels;

INSERT INTO game_data.filter_stg_participant_labels
(
    matchid, teamid, participantid, championid, teamposition,
    highest_value, highest_value_label, low_build_value
)
WITH
item_stats AS (
    SELECT
        ps.matchid,
        ps.teamid,
        ps.participantid,
        ps.championid,
        ps.teamposition,
        sum(ps.v.1) AS attack_damage, -- noqa: LT01
        sum(ps.v.2) AS ability_power, -- noqa: LT01
        sum(ps.v.3) AS lethality, -- noqa: LT01
        sum(ps.v.4) AS on_hit, -- noqa: LT01
        sum(ps.v.5) AS crit, -- noqa: LT01
        sum(ps.v.6) AS utility_enchanter, -- noqa: LT01
        sum(ps.v.7) AS utility_protection, -- noqa: LT01
        sum(ps.v.8) AS ar_tank, -- noqa: LT01
        sum(ps.v.9) AS mr_tank, -- noqa: LT01
        sum(ps.v.10) AS ad_off_tank, -- noqa: LT01
        sum(ps.v.11) AS ap_off_tank -- noqa: LT01
    FROM (
        SELECT
            matchid,
            teamid,
            participantid,
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
                dictGetOrDefault(
                    'game_data.item_value_map_dict',
                    (
                        'attack_damage', 'ability_power', 'lethality', 'on_hit', 'crit',
                        'utility_enchanter', 'utility_protection',
                        'ar_tank', 'mr_tank', 'ad_off_tank', 'ap_off_tank'
                    ),
                    (toInt32(0), '', item_id),
                    (
                        toFloat32(0), toFloat32(0), toFloat32(0), toFloat32(0),
                        toFloat32(0), toFloat32(0), toFloat32(0), toFloat32(0),
                        toFloat32(0), toFloat32(0), toFloat32(0)
                    )
                )
            ) AS v
        FROM (
            SELECT
                ps.matchid,
                ps.teamid,
                ps.participantid,
                ps.championid,
                toString(ps.teamposition) AS teamposition,
                arrayJoin(if(
                    empty(ps.nonzero_item_ids),
                    [toUInt32(0)],
                    ps.nonzero_item_ids
                )) AS item_id
            FROM (
                SELECT
                    ps.matchid,
                    ps.teamid,
                    ps.participantid,
                    ps.championid,
                    ps.teamposition,
                    arrayFilter(
                        slot -> slot != 0, -- noqa: RF03
                        arrayConcat(
                            [
                                toUInt32(ps.item0), toUInt32(ps.item1),
                                toUInt32(ps.item2), toUInt32(ps.item3),
                                toUInt32(ps.item4), toUInt32(ps.item5),
                                toUInt32(ps.item6)
                            ],
                            if(
                                isNull(ps.rolebounditem),
                                CAST([], 'Array(UInt32)'),
                                [toUInt32(assumeNotNull(ps.rolebounditem))]
                            )
                        )
                    ) AS nonzero_item_ids
                FROM game_data.participant_stats_corrected AS ps
                SEMI JOIN game_data.filter_stg_stage1_valid_matchids AS sv
                    ON ps.matchid = sv.matchid
            ) AS ps
        )
    ) AS ps
    GROUP BY ps.matchid, ps.teamid, ps.participantid, ps.championid, ps.teamposition
)

SELECT
    matchid,
    teamid,
    participantid,
    championid,
    teamposition,
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
    ) AS highest_value_label,
    toUInt8(highest_value < 0.5) AS low_build_value
FROM item_stats;

-- =============================================================================
-- Final rollup: filter_stg_game_flags populated in a single write.
-- Reads only from the narrow staging tables; no participant_stats scan.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_game_flags;

INSERT INTO game_data.filter_stg_game_flags
(
    matchid,
    player_low_kda,
    player_gold_spent,
    player_high_winrate,
    team_kills_to_deaths,
    solo_carried,
    too_little_damage,
    low_minions_killed,
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy,
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy,
    unknown_teamposition,
    game_ruining_behavior,
    was_severe_transgressor,
    caused_game_end_from_ignb_surrender,
    team_ignb_surrendered,
    was_premade_with_ignb_game_end_causer,
    was_premade_with_severe_transgressor,
    zero_spell_casts_loss,
    zero_item_purchases_loss,
    low_build_value,
    any_filter_triggered
)
WITH
stage1_rollup AS (
    SELECT
        matchid,
        max(player_low_kda) AS player_low_kda,
        max(player_gold_spent) AS player_gold_spent,
        max(player_high_winrate) AS player_high_winrate,
        max(team_kills_to_deaths) AS team_kills_to_deaths,
        max(solo_carried) AS solo_carried,
        max(too_little_damage) AS too_little_damage,
        max(low_minions_killed) AS low_minions_killed,
        max(team_non_utility_avg_cs_per_min_gt_1_0_below_enemy)
            AS team_non_utility_avg_cs_per_min_gt_1_0_below_enemy,
        max(team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy)
            AS team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy,
        max(unknown_teamposition) AS unknown_teamposition,
        max(game_ruining_behavior) AS game_ruining_behavior,
        max(was_severe_transgressor) AS was_severe_transgressor,
        max(caused_game_end_from_ignb_surrender)
            AS caused_game_end_from_ignb_surrender,
        max(team_ignb_surrendered) AS team_ignb_surrendered,
        max(was_premade_with_ignb_game_end_causer)
            AS was_premade_with_ignb_game_end_causer,
        max(was_premade_with_severe_transgressor)
            AS was_premade_with_severe_transgressor,
        max(zero_spell_casts_loss) AS zero_spell_casts_loss,
        max(zero_item_purchases_loss) AS zero_item_purchases_loss
    FROM game_data.filter_stg_participant_flags
    GROUP BY matchid
),

low_build_value_rollup AS (
    SELECT
        matchid,
        max(low_build_value) AS low_build_value
    FROM game_data.filter_stg_participant_labels
    GROUP BY matchid
)

SELECT
    s.matchid,
    s.player_low_kda,
    s.player_gold_spent,
    s.player_high_winrate,
    s.team_kills_to_deaths,
    s.solo_carried,
    s.too_little_damage,
    s.low_minions_killed,
    s.team_non_utility_avg_cs_per_min_gt_1_0_below_enemy,
    s.team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy,
    s.unknown_teamposition,
    s.game_ruining_behavior,
    s.was_severe_transgressor,
    s.caused_game_end_from_ignb_surrender,
    s.team_ignb_surrendered,
    s.was_premade_with_ignb_game_end_causer,
    s.was_premade_with_severe_transgressor,
    s.zero_spell_casts_loss,
    s.zero_item_purchases_loss,
    COALESCE(lbv.low_build_value, toUInt8(0)) AS low_build_value,
    (
        s.player_low_kda
        OR s.player_gold_spent
        OR s.player_high_winrate
        OR s.team_kills_to_deaths
        OR s.solo_carried
        OR s.too_little_damage
        OR s.low_minions_killed
        OR s.team_non_utility_avg_cs_per_min_gt_1_0_below_enemy
        OR s.team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy
        OR s.unknown_teamposition
        OR s.game_ruining_behavior
        OR s.was_severe_transgressor
        OR s.caused_game_end_from_ignb_surrender
        OR s.team_ignb_surrendered
        OR s.was_premade_with_ignb_game_end_causer
        OR s.was_premade_with_severe_transgressor
        OR s.zero_spell_casts_loss
        OR s.zero_item_purchases_loss
        OR COALESCE(lbv.low_build_value, toUInt8(0))
    ) AS any_filter_triggered
FROM stage1_rollup AS s
LEFT JOIN low_build_value_rollup AS lbv ON s.matchid = lbv.matchid;

-- =============================================================================
-- Stage 3: bitmask result (one row per matchid+teamid+participantid).
-- Bit assignments (matching the f01..f20 hard-rule numbering used in
-- filter_evidence.md):
--   Player:  0=f01 player_low_kda            1=f02 player_gold_spent
--            2=f03 player_high_winrate       4=f05 solo_carried
--            5=f06 too_little_damage         6=f07 low_minions_killed
--            9=f10 low_build_value          17=f18 zero_spell_casts_loss
--           19=f20 zero_item_purchases_loss
--   Team:    3=f04 team_kills_to_deaths
--            7=f08 team_non_utility_avg_cs_per_min_gt_2_0_below_enemy
--            8=f09 team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy
--   Game:   10=f11 unknown_teamposition    11=f12 game_ruining_behavior
--           12=f13 was_severe_transgressor
--           13=f14 caused_game_end_from_ignb_surrender
--           14=f15 team_ignb_surrendered   15=f16 premade_ignb_causer
--           16=f17 premade_severe
-- =============================================================================

TRUNCATE TABLE game_data.filter_result;

INSERT INTO game_data.filter_result
(
    matchid,
    teamid,
    participantid,
    player_rule_mask,
    team_rule_mask,
    game_rule_mask,
    rule_mask,
    is_valid
)
SELECT
    pf.matchid,
    pf.teamid,
    pf.participantid,
    pf.player_low_kda * 1
    + pf.player_gold_spent * 2
    + pf.player_high_winrate * 4
    + pf.solo_carried * 16
    + pf.too_little_damage * 32
    + pf.low_minions_killed * 64
    + COALESCE(pl.low_build_value, toUInt8(0)) * 512
    + pf.zero_spell_casts_loss * 131072
    + pf.zero_item_purchases_loss * 524288 AS player_rule_mask,
    pf.team_kills_to_deaths * 8
    + pf.team_non_utility_avg_cs_per_min_gt_1_0_below_enemy * 128
    + pf.team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy * 256
        AS team_rule_mask,
    gf.unknown_teamposition * 1024
    + gf.game_ruining_behavior * 2048
    + gf.was_severe_transgressor * 4096
    + gf.caused_game_end_from_ignb_surrender * 8192
    + gf.team_ignb_surrendered * 16384
    + gf.was_premade_with_ignb_game_end_causer * 32768
    + gf.was_premade_with_severe_transgressor * 65536 AS game_rule_mask,
    gf.player_low_kda * 1
    + gf.player_gold_spent * 2
    + gf.player_high_winrate * 4
    + gf.team_kills_to_deaths * 8
    + gf.solo_carried * 16
    + gf.too_little_damage * 32
    + gf.low_minions_killed * 64
    + gf.team_non_utility_avg_cs_per_min_gt_1_0_below_enemy * 128
    + gf.team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy * 256
    + gf.low_build_value * 512
    + gf.unknown_teamposition * 1024
    + gf.game_ruining_behavior * 2048
    + gf.was_severe_transgressor * 4096
    + gf.caused_game_end_from_ignb_surrender * 8192
    + gf.team_ignb_surrendered * 16384
    + gf.was_premade_with_ignb_game_end_causer * 32768
    + gf.was_premade_with_severe_transgressor * 65536
    + gf.zero_spell_casts_loss * 131072
    + gf.zero_item_purchases_loss * 524288 AS rule_mask,
    gf.any_filter_triggered = 0 AS is_valid
FROM game_data.filter_stg_participant_flags AS pf
INNER JOIN game_data.filter_stg_game_flags AS gf ON pf.matchid = gf.matchid
ANY LEFT JOIN game_data.filter_stg_participant_labels AS pl
    ON
        pf.matchid = pl.matchid
        AND pf.teamid = pl.teamid
        AND pf.participantid = pl.participantid;
