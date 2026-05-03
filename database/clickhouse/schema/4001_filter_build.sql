-- noqa: disable=PRS
-- Filter pipeline: population in dependency order.
--
-- Flow:
--   1a. player_winrates   (from game_data.players)
--   1b. player_role_rates (from game_data.participant_stats, GROUP BY puuid+role)
--   1c. team_flags        (from game_data.participant_stats, self-join on
--                          the per-team aggregate only)
--   1.  participant_flags (stage 1 flags; single scan of participant_stats
--                          joined to player_winrates + player_role_rates +
--                          team_flags + info)
--   1r. stage1_valid_matchids (cheap rollup of participant_flags)
--   2.  rare_roles        (SEMI JOIN stage1_valid_matchids so the expensive
--                          pick-count scans see only stage-1-clean games)
--   2r. stage2_valid_matchids (stage1_valid MINUS rare_roles)
--   3.  rare_builds       (SEMI JOIN stage2_valid_matchids so the item_value
--                          join sees an even smaller pool)
--   4f. game_flags        (final rollup: stage-1 participant_flags + rare_roles
--                          + rare_builds; written once)
--   4.  filter_result     (bitmask, built from participant_flags + game_flags
--                          + rare_builds; avoids re-scanning participant_stats)
--
-- Key memory/CPU notes:
--   * participant_stats is scanned in stage 1b (GROUP BY puuid+role for
--     off-role rates), once in stage 1 (main participant flags), once
--     (SEMI-joined to ~66%) in stage 2, and once (SEMI-joined to the even
--     smaller stage-2 pool) in stage 3.
--   * game_time_lte_16_5 is read directly from info.gameduration instead of a
--     max(timeplayed) aggregate.
--   * SEMI JOIN on the small matchid helper tables keeps the hash-build side
--     tiny and avoids materialising the full matchid set into memory.
--
-- Run with clickhouse-client --multiquery after 4000_filter_schema.sql.

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
-- Stage 1b: player role rates (per-puuid, per-teamposition game counts)
-- Scans participant_stats once; the resulting table is tiny (one row per
-- distinct puuid+teamposition) and fits in memory for the ANY LEFT JOIN in 1c.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_player_role_rates;

INSERT INTO game_data.filter_stg_player_role_rates (
    puuid, teamposition, role_games, total_games
)
SELECT
    puuid,
    teamposition,
    count() AS role_games,
    sum(count()) OVER (PARTITION BY puuid) AS total_games
FROM game_data.participant_stats
GROUP BY puuid, teamposition;

-- =============================================================================
-- Stage 1c: team flags with enemy-relative stats (one row per matchid+teamid)
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_team_flags;

INSERT INTO game_data.filter_stg_team_flags
(
    matchid,
    teamid,
    team_kills,
    team_damage_to_champions,
    team_kills_to_deaths,
    team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy
)
WITH team_base AS (
    SELECT
        matchid,
        teamid,
        sum(kills) AS team_kills,
        sum(deaths) AS team_deaths,
        sum(totaldamagedealttochampions) AS team_damage_to_champions,
        avgIf(
            (totalminionskilled + neutralminionskilled) * 60.0 / timeplayed,
            teamposition != 'UTILITY' AND timeplayed > 0
        ) AS team_non_utility_avg_cs_per_min,
        sumIf(totaldamagedealttochampions, teamposition != 'UTILITY')
            AS team_non_utility_damage_to_champions
    FROM game_data.participant_stats
    GROUP BY matchid, teamid
)

SELECT
    tb.matchid,
    tb.teamid,
    tb.team_kills,
    tb.team_damage_to_champions,
    tb.team_kills * 3 < tb.team_deaths AS team_kills_to_deaths,
    enemy.team_non_utility_avg_cs_per_min - tb.team_non_utility_avg_cs_per_min > 2.5
        AS team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    tb.team_non_utility_damage_to_champions
    / enemy.team_non_utility_damage_to_champions < (1.0 / 3.0)
        AS team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy
FROM team_base AS tb
LEFT JOIN team_base AS enemy
    ON tb.matchid = enemy.matchid AND tb.teamid != enemy.teamid;

-- =============================================================================
-- Stage 1: per-participant flags for all 13 cheap filters.
-- game_time_lte_16_5 is derived from info.gameduration; info has exactly one
-- row per matchid, so ANY LEFT JOIN is safe.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_participant_flags;

INSERT INTO game_data.filter_stg_participant_flags
(
    matchid,
    teamid,
    participantid,
    player_low_kda,
    player_gold_spent,
    no_contribution_kda,
    bad_summoner_usage,
    player_high_winrate,
    team_kills_to_deaths,
    solo_carried,
    too_little_damage,
    low_minions_killed,
    team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy,
    off_role_low_experience,
    game_time_lte_16_5
)
SELECT
    ps.matchid,
    ps.teamid,
    ps.participantid,
    (ps.kills + ps.assists) * 6 < ps.deaths AS player_low_kda,
    ps.goldearned > 0 AND ps.goldspent * 100 < ps.goldearned * 50
        AS player_gold_spent,
    ps.kills + ps.assists = 0 AND ps.deaths > 4 AS no_contribution_kda,
    ps.summoner1casts = 0 OR ps.summoner2casts = 0 AS bad_summoner_usage,
    pl.wins + pl.losses > 40
    AND pl.wins * 100 > (pl.wins + pl.losses) * 70 AS player_high_winrate,
    tf.team_kills_to_deaths,
    tf.team_kills > 0 AND ps.kills * 100 > tf.team_kills * 75 AS solo_carried,
    (
        ps.teamposition != 'UTILITY'
        AND tf.team_damage_to_champions > 0
        AND ps.totaldamagedealttochampions * 1000 < tf.team_damage_to_champions * 50
    ) AS too_little_damage,
    (
        ps.teamposition != 'UTILITY'
        AND ps.timeplayed > 0
        AND (ps.totalminionskilled + ps.neutralminionskilled) * 60.0 / ps.timeplayed
        < 4.0
    ) AS low_minions_killed,
    tf.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    tf.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy,
    toUInt8(0) AS off_role_low_experience, -- disabled
    i.gameduration <= 16 * 60 + 30 AS game_time_lte_16_5
FROM game_data.participant_stats AS ps
ANY LEFT JOIN game_data.filter_stg_player_winrates AS pl ON ps.puuid = pl.puuid
ANY LEFT JOIN game_data.filter_stg_team_flags AS tf
    ON ps.matchid = tf.matchid AND ps.teamid = tf.teamid
ANY LEFT JOIN game_data.info AS i ON ps.matchid = i.matchid;

-- =============================================================================
-- Stage 1 rollup: matchids that pass every stage-1 filter.
-- Produced by aggregating filter_stg_participant_flags; no scan of the large
-- participant_stats table.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_stage1_valid_matchids;

INSERT INTO game_data.filter_stg_stage1_valid_matchids (matchid)
SELECT matchid
FROM game_data.filter_stg_participant_flags
GROUP BY matchid
HAVING
    max(player_low_kda) = 0
    AND max(player_gold_spent) = 0
    AND max(no_contribution_kda) = 0
    AND max(bad_summoner_usage) = 0
    AND max(player_high_winrate) = 0
    AND max(team_kills_to_deaths) = 0
    AND max(solo_carried) = 0
    AND max(too_little_damage) = 0
    AND max(low_minions_killed) = 0
    AND max(team_non_utility_avg_cs_per_min_gt_2_5_below_enemy) = 0
    AND max(team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy) = 0
    AND max(off_role_low_experience) = 0
    AND max(game_time_lte_16_5) = 0;

-- =============================================================================
-- Stage 2: rare-role detection over stage-1-clean games only.
-- Each participant_stats scan is SEMI-joined to stage1_valid_matchids so the
-- global pick counts that drive the "rare" threshold are measured against the
-- cleaned pool (counts shift materially after stage 1 strips ~33% of games).
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_rare_roles;

INSERT INTO game_data.filter_stg_rare_roles (matchid, has_rare_role)
WITH
champion_teamposition_pick_counts AS (
    SELECT
        ps.championid,
        ps.teamposition,
        count() AS champion_teamposition_picks
    FROM game_data.participant_stats AS ps
    SEMI JOIN game_data.filter_stg_stage1_valid_matchids AS sv
        ON ps.matchid = sv.matchid
    WHERE ps.championid IS NOT NULL AND ps.teamposition != 'UNKNOWN'
    GROUP BY championid, teamposition
),

champion_pick_totals AS (
    SELECT
        championid,
        sum(champion_teamposition_picks) AS champion_picks
    FROM champion_teamposition_pick_counts
    GROUP BY championid
),

rare_champion_teampositions AS ( -- noqa: ST03
    SELECT
        ctpc.championid,
        ctpc.teamposition
    FROM champion_teamposition_pick_counts AS ctpc
    INNER JOIN champion_pick_totals AS cpt USING (championid)
    WHERE ctpc.champion_teamposition_picks * 250 < cpt.champion_picks
),

player_rare_picks AS ( -- noqa: ST03
    SELECT
        ps.puuid AS puuid, -- noqa: AL09
        ps.championid AS championid, -- noqa: AL09
        ps.teamposition AS teamposition, -- noqa: AL09
        count() AS player_champion_teamposition_picks
    FROM game_data.participant_stats AS ps
    SEMI JOIN game_data.filter_stg_stage1_valid_matchids AS sv
        ON ps.matchid = sv.matchid
    INNER JOIN rare_champion_teampositions AS rct
        ON ps.championid = rct.championid AND ps.teamposition = rct.teamposition
    GROUP BY ps.puuid, ps.championid, ps.teamposition
    HAVING player_champion_teamposition_picks < 30
)

SELECT
    ps.matchid,
    1 AS has_rare_role
FROM game_data.participant_stats AS ps
SEMI JOIN game_data.filter_stg_stage1_valid_matchids AS sv
    ON ps.matchid = sv.matchid
INNER JOIN player_rare_picks AS prp
    ON
        ps.puuid = prp.puuid
        AND ps.championid = prp.championid
        AND ps.teamposition = prp.teamposition
GROUP BY ps.matchid;

-- =============================================================================
-- Stage 2 rollup: stage-1-valid matchids minus rare-role matchids.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_stage2_valid_matchids;

INSERT INTO game_data.filter_stg_stage2_valid_matchids (matchid)
SELECT s1.matchid
FROM game_data.filter_stg_stage1_valid_matchids AS s1
LEFT ANTI JOIN game_data.filter_stg_rare_roles AS rr -- noqa: ST11
    ON s1.matchid = rr.matchid;

-- =============================================================================
-- Stage 3: rare-build detection over stage-2-clean games only.
-- Each participant's build label is computed once and persisted to
-- filter_stg_participant_labels.  Rare-build flagging then iterates: every
-- pass measures label counts on the pool that remains after earlier passes
-- removed their rare-build matches, and flags any participant whose label
-- now has < 8 occurrences.  Iteration is required because removing a match
-- for one rare participant also drops the other participants in that match
-- from the pool, which can push previously-borderline labels below the
-- threshold.  The loop converges in a handful of passes; three fixed passes
-- are enough in practice.  Labels match 5133's multiIf ordering exactly so
-- the threshold measured here is the same label that 5133 emits downstream.
-- =============================================================================

TRUNCATE TABLE game_data.filter_stg_participant_labels;

INSERT INTO game_data.filter_stg_participant_labels
(matchid, teamid, participantid, championid, teamposition, highest_value_label)
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
                arrayJoin(arrayConcat(
                    [
                        toUInt32(ps.item0), toUInt32(ps.item1), toUInt32(ps.item2),
                        toUInt32(ps.item3), toUInt32(ps.item4),
                        toUInt32(ps.item5), toUInt32(ps.item6)
                    ],
                    if(
                        isNull(ps.rolebounditem),
                        CAST([], 'Array(UInt32)'),
                        [toUInt32(assumeNotNull(ps.rolebounditem))]
                    )
                )) AS item_id
            FROM game_data.participant_stats AS ps
            SEMI JOIN game_data.filter_stg_stage2_valid_matchids AS sv
                ON ps.matchid = sv.matchid
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
    multiIf(
        greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ) = 0, 'none',
        crit = greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ), 'crit',
        lethality = greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ), 'lethality',
        utility_enchanter = greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ), 'utility_enchanter',
        utility_protection = greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ), 'utility_protection',
        ar_tank = greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ), 'ar_tank',
        mr_tank = greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ), 'mr_tank',
        ad_off_tank = greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ), 'ad_off_tank',
        ap_off_tank = greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ), 'ap_off_tank',
        on_hit = greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ), 'on_hit',
        ability_power = greatest(
            attack_damage, ability_power, lethality, on_hit, crit,
            utility_enchanter, utility_protection,
            ar_tank, mr_tank, ad_off_tank, ap_off_tank
        ), 'ability_power',
        'attack_damage'
    ) AS highest_value_label
FROM item_stats;

TRUNCATE TABLE game_data.filter_stg_rare_builds;

-- Pass 1: measure label counts over the full stage-2-clean pool.
INSERT INTO game_data.filter_stg_rare_builds
(matchid, teamid, participantid, rare_build_label)
WITH build_dist AS (
    SELECT
        championid,
        teamposition,
        highest_value_label,
        count() AS label_count
    FROM game_data.filter_stg_participant_labels
    GROUP BY championid, teamposition, highest_value_label
    HAVING label_count < 8
)

SELECT
    pl.matchid,
    pl.teamid,
    pl.participantid,
    toUInt8(1) AS rare_build_label
FROM game_data.filter_stg_participant_labels AS pl
INNER JOIN build_dist AS bd
    ON
        pl.championid = bd.championid
        AND pl.teamposition = bd.teamposition
        AND pl.highest_value_label = bd.highest_value_label;

-- Pass 2: remeasure on the pool that excludes pass-1 rare-build matches.
-- Any (champ, pos, label) whose count fell below 8 due to cascading match
-- removal is flagged here.
INSERT INTO game_data.filter_stg_rare_builds
(matchid, teamid, participantid, rare_build_label)
WITH
excluded_matchids AS (
    SELECT DISTINCT matchid FROM game_data.filter_stg_rare_builds
),

surviving_labels AS (
    SELECT pl.*
    FROM game_data.filter_stg_participant_labels AS pl
    LEFT ANTI JOIN excluded_matchids AS em ON pl.matchid = em.matchid -- noqa: ST11
),

build_dist AS (
    SELECT
        championid,
        teamposition,
        highest_value_label,
        count() AS label_count
    FROM surviving_labels
    GROUP BY championid, teamposition, highest_value_label
    HAVING label_count < 8
)

SELECT
    pl.matchid,
    pl.teamid,
    pl.participantid,
    toUInt8(1) AS rare_build_label
FROM surviving_labels AS pl
INNER JOIN build_dist AS bd
    ON
        pl.championid = bd.championid
        AND pl.teamposition = bd.teamposition
        AND pl.highest_value_label = bd.highest_value_label;

-- Pass 3: one more sweep to catch cascades introduced by pass 2.
INSERT INTO game_data.filter_stg_rare_builds
(matchid, teamid, participantid, rare_build_label)
WITH
excluded_matchids AS (
    SELECT DISTINCT matchid FROM game_data.filter_stg_rare_builds
),

surviving_labels AS (
    SELECT pl.*
    FROM game_data.filter_stg_participant_labels AS pl
    LEFT ANTI JOIN excluded_matchids AS em ON pl.matchid = em.matchid -- noqa: ST11
),

build_dist AS (
    SELECT
        championid,
        teamposition,
        highest_value_label,
        count() AS label_count
    FROM surviving_labels
    GROUP BY championid, teamposition, highest_value_label
    HAVING label_count < 8
)

SELECT
    pl.matchid,
    pl.teamid,
    pl.participantid,
    toUInt8(1) AS rare_build_label
FROM surviving_labels AS pl
INNER JOIN build_dist AS bd
    ON
        pl.championid = bd.championid
        AND pl.teamposition = bd.teamposition
        AND pl.highest_value_label = bd.highest_value_label;

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
    no_contribution_kda,
    bad_summoner_usage,
    player_high_winrate,
    team_kills_to_deaths,
    solo_carried,
    too_little_damage,
    low_minions_killed,
    team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy,
    off_role_low_experience,
    game_time_lte_16_5,
    has_rare_role,
    rare_build_label,
    any_filter_triggered
)
WITH
stage1_rollup AS (
    SELECT
        matchid,
        max(player_low_kda) AS player_low_kda,
        max(player_gold_spent) AS player_gold_spent,
        max(no_contribution_kda) AS no_contribution_kda,
        max(bad_summoner_usage) AS bad_summoner_usage,
        max(player_high_winrate) AS player_high_winrate,
        max(team_kills_to_deaths) AS team_kills_to_deaths,
        max(solo_carried) AS solo_carried,
        max(too_little_damage) AS too_little_damage,
        max(low_minions_killed) AS low_minions_killed,
        max(team_non_utility_avg_cs_per_min_gt_2_5_below_enemy)
            AS team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
        max(team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy)
            AS team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy,
        max(off_role_low_experience) AS off_role_low_experience,
        max(game_time_lte_16_5) AS game_time_lte_16_5
    FROM game_data.filter_stg_participant_flags
    GROUP BY matchid
),

rare_build_rollup AS (
    SELECT
        matchid,
        max(rare_build_label) AS rare_build_label
    FROM game_data.filter_stg_rare_builds
    GROUP BY matchid
)

SELECT
    s.matchid,
    s.player_low_kda,
    s.player_gold_spent,
    s.no_contribution_kda,
    s.bad_summoner_usage,
    s.player_high_winrate,
    s.team_kills_to_deaths,
    s.solo_carried,
    s.too_little_damage,
    s.low_minions_killed,
    s.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy,
    s.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy,
    s.off_role_low_experience,
    s.game_time_lte_16_5,
    COALESCE(rr.has_rare_role, toUInt8(0)) AS has_rare_role,
    COALESCE(rb.rare_build_label, toUInt8(0)) AS rare_build_label,
    (
        s.player_low_kda
        OR s.player_gold_spent
        OR s.no_contribution_kda
        OR s.bad_summoner_usage
        OR s.player_high_winrate
        OR s.team_kills_to_deaths
        OR s.solo_carried
        OR s.too_little_damage
        OR s.low_minions_killed
        OR s.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy
        OR s.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy
        OR s.off_role_low_experience
        OR s.game_time_lte_16_5
        OR COALESCE(rr.has_rare_role, toUInt8(0))
        OR COALESCE(rb.rare_build_label, toUInt8(0))
    ) AS any_filter_triggered
FROM stage1_rollup AS s
LEFT JOIN game_data.filter_stg_rare_roles AS rr ON s.matchid = rr.matchid
LEFT JOIN rare_build_rollup AS rb ON s.matchid = rb.matchid;

-- =============================================================================
-- Stage 4: bitmask result (one row per matchid+teamid+participantid).
-- Built from participant_flags + game_flags + rare_builds — no scan of
-- participant_stats here either.  Bit assignments:
--   Player:  0=player_low_kda  1=player_gold_spent  2=no_contribution_kda
--            3=bad_summoner_usage  4=player_high_winrate  6=solo_carried
--            7=too_little_damage  8=low_minions_killed
--            15=rare_build_label
--            16=off_role_low_experience
--   Team:    5=team_kills_to_deaths  9=team_non_utility_avg_cs_per_min
--            10=team_non_utility_damage_to_champions_ratio
--   Game:    13=game_time_lte_16_5  14=has_rare_role
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
    + pf.no_contribution_kda * 4
    + pf.bad_summoner_usage * 8
    + pf.player_high_winrate * 16
    + pf.solo_carried * 64
    + pf.too_little_damage * 128
    + pf.low_minions_killed * 256
    + COALESCE(rb.rare_build_label, toUInt8(0)) * 32768
    + pf.off_role_low_experience * 65536 AS player_rule_mask,
    pf.team_kills_to_deaths * 32
    + pf.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy * 512
    + pf.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy * 1024
        AS team_rule_mask,
    gf.game_time_lte_16_5 * 8192
    + gf.has_rare_role * 16384 AS game_rule_mask,
    gf.player_low_kda * 1
    + gf.player_gold_spent * 2
    + gf.no_contribution_kda * 4
    + gf.bad_summoner_usage * 8
    + gf.player_high_winrate * 16
    + gf.team_kills_to_deaths * 32
    + gf.solo_carried * 64
    + gf.too_little_damage * 128
    + gf.low_minions_killed * 256
    + gf.team_non_utility_avg_cs_per_min_gt_2_5_below_enemy * 512
    + gf.team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy * 1024
    + gf.game_time_lte_16_5 * 8192
    + gf.has_rare_role * 16384
    + gf.rare_build_label * 32768
    + gf.off_role_low_experience * 65536 AS rule_mask,
    gf.any_filter_triggered = 0 AS is_valid
FROM game_data.filter_stg_participant_flags AS pf
INNER JOIN game_data.filter_stg_game_flags AS gf ON pf.matchid = gf.matchid
ANY LEFT JOIN game_data.filter_stg_rare_builds AS rb
    ON
        pf.matchid = rb.matchid
        AND pf.teamid = rb.teamid
        AND pf.participantid = rb.participantid;
