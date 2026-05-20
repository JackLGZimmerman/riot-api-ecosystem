-- noqa: disable=AL05,AL09,LT01,LT02,LT05,RF02,RF03,ST03,ST09
-- UPDATED REDUCED FEATURE SET FOR DRAFT-PHASE TRANSFORMER INPUT
--
-- Changes made:
--   1. Reduced the final emitted model features to the core historical
--      champion/role/build/bin profile signals:
--        identity, reliability, scaling/economy, threat, damage identity,
--        survivability, utility, objectives/map, and vision.
--   2. Removed highly correlated or lower-priority emitted features:
--        avg_physical_damage_dealt, avg_magic_damage_dealt,
--        avg_true_damage_dealt, avg_damage_to_buildings,
--        avg_damage_to_turrets, avg_damage_to_epic_monsters,
--        avg_baron_kills, avg_dragon_kills, avg_wards_placed,
--        avg_wards_killed, avg_vision_actions, avg_deaths, kda,
--        damage_per_death, damage_taken_per_death,
--        avg_total_time_cc_dealt, avg_minions_killed,
--        avg_neutral_minions_killed, jungle CS split metrics,
--        and separate heal/shield components.
--   3. Kept raw inputs in the temporary table only when they are needed to
--      calculate the reduced emitted metrics.
--   4. Added an explicit INSERT column list so the intended reduced output
--      schema is visible and aligned with the target table.
--
-- Per-(champion, role, build) scaling priors. Each valid train participant
-- contributes one singleton row; enemies are intentionally ignored.
-- Every (championid, teamposition, build) emits up to four rows, one per
-- 8007 legendary-item scaling bin. Each game is assigned to a bin by its
-- total duration using the deterministic probabilistic smoothing from
-- 8009 (8009_legendary_items_smoothed_scaling_bins): instead of a hard cut
-- at t3/t4/t5, rows near a boundary move to the neighbouring bin with a
-- normal-CDF probability, so bin membership does not jump discontinuously
-- at the thresholds. The per-bin metrics are participant_stats values
-- over the games assigned to that bin.
--
-- NORMALISATION: every avg_* metric is a per-minute rate -- each game's
-- end-of-game stat is divided by that game's duration in minutes, then
-- averaged over the bin. The columns carry no _per_min suffix; the
-- per-minute basis is global and implicit. NOT per-minute:
-- log_matchups, win_rate, damage shares, damage_to_taken_ratio, and
-- avg_item_completions.
--
-- REDUCED FINAL FEATURE GROUPS:
--
--   Identity:
--       championid, championname, teamposition, build, bin_idx, bin_label
--
--   Reliability / outcome prior:
--       log_matchups, win_rate
--
--   Scaling / economy:
--       avg_gold, avg_xp, avg_item_completions, avg_total_cs
--
--   Threat:
--       avg_kills, avg_kills_assists, avg_total_damage_dealt
--
--   Damage identity:
--       physical_damage_share, magic_damage_share, true_damage_share
--
--   Survivability:
--       avg_damage_taken, avg_durability, damage_to_taken_ratio
--
--   Utility:
--       avg_time_ccing_others, avg_protection
--
--   Objectives / map:
--       avg_epic_monster_takedowns, avg_turret_takedowns,
--       avg_damage_to_objectives
--
--   Vision:
--       avg_vision_score, avg_control_wards_bought
--
--   bin 1  early-mid (2-3 items)  16.5 min .. t3
--   bin 2  mid       (3-4 items)  t3       .. t4
--   bin 3  mid-late  (4-5 items)  t4       .. t5
--   bin 4  late      (5+  items)  t5       .. inf
--
-- t3/t4/t5 are the 8007 thresholds, recomputed here over the train split
-- only so no validation/test signal leaks into the priors. The smoothing
-- transitions span the train-split medians of the adjacent strict bins.
--
-- The participant_stats <-> split <-> info <-> item-label join is the
-- expensive step, so it is materialised once into a session temporary
-- table; the threshold search, bin medians and per-bin aggregation then
-- all read that lean table instead of re-running the join.

TRUNCATE TABLE game_data_filtered.synergy_1vx;

DROP TEMPORARY TABLE IF EXISTS tmp_1vx_participants;

-- One row per valid train participant: identity, outcome, game duration,
-- selected end-of-game stat columns, and the 8007 legendary-item count.
CREATE TEMPORARY TABLE tmp_1vx_participants AS
SELECT
    ps.matchid AS matchid,
    ps.participantid AS participantid,
    assumeNotNull(ps.championid) AS championid,
    toString(ps.teamposition) AS teamposition,
    ivt.highest_value_label AS build,
    toUInt8(ps.win > 0) AS win,
    i.gameduration AS gameduration,

    -- Scaling / economy.
    ps.goldearned AS goldearned,
    ps.champexperience AS champexperience,
    ps.totalminionskilled AS minions_killed,
    ps.neutralminionskilled AS neutral_minions_killed,

    -- Threat and damage identity.
    ps.kills AS kills,
    ps.kills + ps.assists AS kills_assists,
    ps.totaldamagedealttochampions AS total_damage,
    ps.physicaldamagedealttochampions AS physical_damage,
    ps.magicdamagedealttochampions AS magic_damage,
    ps.truedamagedealttochampions AS true_damage,

    -- Survivability.
    ps.totaldamagetaken AS damage_taken,
    ps.damageselfmitigated AS damage_self_mitigated,
    if(
        ps.totalheal >= ps.totalhealsonteammates,
        ps.totalheal - ps.totalhealsonteammates,
        0
    ) AS self_heal,

    -- Utility.
    ps.timeccingothers AS time_ccing_others,
    ps.totalhealsonteammates AS heals_on_teammates,
    ps.totaldamageshieldedonteammates AS damage_shielded_on_teammates,

    -- Objectives / map.
    ps.baronkills AS baron_kills,
    ps.dragonkills AS dragon_kills,
    ps.turrettakedowns AS turret_takedowns,
    ps.damagedealttoobjectives AS damage_to_objectives,

    -- Vision.
    ps.visionscore AS vision_score,
    ps.visionwardsboughtingame AS control_wards_bought,

    -- Build scaling bin driver.
    arraySum(arrayMap(
        x -> if(
            x != 0
            AND dictHas(
                'game_data.item_value_map_dict',
                (toInt32(0), toString(''), toUInt32(x))
            ),
            1,
            0
        ),
        [ps.item0, ps.item1, ps.item2,
         ps.item3, ps.item4, ps.item5, ps.item6]
    )) AS legendary_items
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS s
    ON ps.matchid = s.matchid
INNER JOIN game_data_filtered.info AS i
    ON ps.matchid = i.matchid
INNER JOIN game_data_filtered.participant_item_value_totals AS ivt
    ON
        ps.matchid = ivt.matchid
        AND ps.participantid = ivt.participantid
WHERE
    s.split = 'train'
    AND ps.championid IS NOT NULL
    AND ps.teamposition IN ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
    AND i.gameduration >= 16.5 * 60
-- participant_stats and participant_item_value_totals are both physically
-- sorted by (matchid, participantid), so full_sorting_merge joins them
-- without building a ~19.5M-row hash table -- peak memory ~2 GiB instead of
-- ~4.3 GiB. The small dimension tables still fall back to parallel_hash.
SETTINGS join_algorithm = 'full_sorting_merge,parallel_hash';

INSERT INTO game_data_filtered.synergy_1vx
(
    split,
    championid,
    championname,
    teamposition,
    build,
    bin_idx,
    bin_label,
    log_matchups,
    win_rate,
    avg_gold,
    avg_xp,
    avg_item_completions,
    avg_total_cs,
    avg_kills,
    avg_kills_assists,
    avg_total_damage_dealt,
    physical_damage_share,
    magic_damage_share,
    true_damage_share,
    avg_damage_taken,
    avg_durability,
    damage_to_taken_ratio,
    avg_time_ccing_others,
    avg_protection,
    avg_epic_monster_takedowns,
    avg_turret_takedowns,
    avg_damage_to_objectives,
    avg_vision_score,
    avg_control_wards_bought
)
WITH
-- Minimum game duration at which the per-participant average legendary-item
-- count first reaches 3 / 4 / 5 (8007's threshold search, train split only),
-- plus the smoothing constants (see 8009): hash_seed reshuffles the
-- deterministic assignment, transition_sigma sets the S-curve sharpness.
thresholds AS (
    SELECT
        minIf(gameduration, avg_count >= 3) AS t3,
        minIf(gameduration, avg_count >= 4) AS t4,
        minIf(gameduration, avg_count >= 5) AS t5,
        toUInt64(2654435761) AS hash_seed,
        toFloat64(0.15) AS transition_sigma
    FROM (
        SELECT
            gameduration,
            avg(legendary_items) AS avg_count
        FROM tmp_1vx_participants
        GROUP BY gameduration
    )
),

-- Median gameduration of each strict bin (train split). These medians are
-- the endpoints of the smoothing transitions.
bin_medians AS (
    SELECT
        quantileExactIf(0.5)(gameduration, strict_bin_idx = 1) AS m1,
        quantileExactIf(0.5)(gameduration, strict_bin_idx = 2) AS m2,
        quantileExactIf(0.5)(gameduration, strict_bin_idx = 3) AS m3,
        quantileExactIf(0.5)(gameduration, strict_bin_idx = 4) AS m4
    FROM (
        SELECT
            p.gameduration,
            multiIf(
                p.gameduration < t.t3, 1,
                p.gameduration < t.t4, 2,
                p.gameduration < t.t5, 3,
                4
            ) AS strict_bin_idx
        FROM tmp_1vx_participants AS p
        CROSS JOIN thresholds AS t
    )
),

-- One row per participant with the smoothed bin. Between two adjacent bin
-- medians the row moves to the higher bin with a normal-CDF probability,
-- compared against a deterministic 0-1 hash of (matchid, participantid,
-- boundary_id, hash_seed). Outside m1..m4 the row keeps its extreme bin.
assigned AS (
    SELECT
        p.championid,
        p.teamposition,
        p.build,
        p.win,
        p.goldearned,
        p.champexperience,
        p.kills,
        p.kills_assists,
        p.total_damage,
        p.physical_damage,
        p.magic_damage,
        p.true_damage,
        p.damage_taken,
        p.self_heal,
        p.damage_self_mitigated,
        p.legendary_items,
        p.minions_killed,
        p.neutral_minions_killed,
        p.gameduration / 60 AS game_minutes,
        p.heals_on_teammates,
        p.damage_shielded_on_teammates,
        p.time_ccing_others,
        p.baron_kills,
        p.dragon_kills,
        p.turret_takedowns,
        p.vision_score,
        p.control_wards_bought,
        p.damage_to_objectives,
        multiIf(
            p.gameduration < bm.m1, toUInt8(1),
            p.gameduration < bm.m2,
            if(
                toFloat64(cityHash64(
                    p.matchid, p.participantid, toUInt8(3), t.hash_seed
                )) / 1.8446744073709552e19
                < 0.5 * (1 + erf(
                    ((p.gameduration - bm.m1) / (bm.m2 - bm.m1) - 0.5)
                    / (t.transition_sigma * sqrt(2))
                )),
                toUInt8(2), toUInt8(1)
            ),
            p.gameduration < bm.m3,
            if(
                toFloat64(cityHash64(
                    p.matchid, p.participantid, toUInt8(4), t.hash_seed
                )) / 1.8446744073709552e19
                < 0.5 * (1 + erf(
                    ((p.gameduration - bm.m2) / (bm.m3 - bm.m2) - 0.5)
                    / (t.transition_sigma * sqrt(2))
                )),
                toUInt8(3), toUInt8(2)
            ),
            p.gameduration < bm.m4,
            if(
                toFloat64(cityHash64(
                    p.matchid, p.participantid, toUInt8(5), t.hash_seed
                )) / 1.8446744073709552e19
                < 0.5 * (1 + erf(
                    ((p.gameduration - bm.m3) / (bm.m4 - bm.m3) - 0.5)
                    / (t.transition_sigma * sqrt(2))
                )),
                toUInt8(4), toUInt8(3)
            ),
            toUInt8(4)
        ) AS bin_idx
    FROM tmp_1vx_participants AS p
    CROSS JOIN thresholds AS t
    CROSS JOIN bin_medians AS bm
)

SELECT
    'train' AS split,
    championid,
    dictGetOrDefault(
        'game_data.championid_name_map_dict',
        'name',
        toString(championid),
        ''
    ) AS championname,
    teamposition,
    build,
    bin_idx,
    multiIf(
        bin_idx = 1, 'early-mid',
        bin_idx = 2, 'mid',
        bin_idx = 3, 'mid-late',
        'late'
    ) AS bin_label,

    toFloat32(log1p(count())) AS log_matchups,
    toFloat32(sum(win) / count()) AS win_rate,

    -- Scaling / economy.
    toFloat32(avg(goldearned / game_minutes)) AS avg_gold,
    toFloat32(avg(champexperience / game_minutes)) AS avg_xp,
    toFloat32(avg(legendary_items)) AS avg_item_completions,
    toFloat32(
        avg((minions_killed + neutral_minions_killed) / game_minutes)
    ) AS avg_total_cs,

    -- Threat.
    toFloat32(avg(kills / game_minutes)) AS avg_kills,
    toFloat32(avg(kills_assists / game_minutes)) AS avg_kills_assists,
    toFloat32(avg(total_damage / game_minutes)) AS avg_total_damage_dealt,

    -- Damage identity.
    toFloat32(sum(physical_damage) / greatest(sum(total_damage), 1)) AS physical_damage_share,
    toFloat32(sum(magic_damage) / greatest(sum(total_damage), 1)) AS magic_damage_share,
    toFloat32(sum(true_damage) / greatest(sum(total_damage), 1)) AS true_damage_share,

    -- Survivability.
    toFloat32(avg(damage_taken / game_minutes)) AS avg_damage_taken,
    toFloat32(avg((self_heal + damage_self_mitigated) / game_minutes)) AS avg_durability,
    toFloat32(avg(total_damage / greatest(damage_taken, 1))) AS damage_to_taken_ratio,

    -- Utility.
    toFloat32(avg(time_ccing_others / game_minutes)) AS avg_time_ccing_others,
    toFloat32(
        avg((heals_on_teammates + damage_shielded_on_teammates) / game_minutes)
    ) AS avg_protection,

    -- Objectives / map.
    toFloat32(avg((baron_kills + dragon_kills) / game_minutes)) AS avg_epic_monster_takedowns,
    toFloat32(avg(turret_takedowns / game_minutes)) AS avg_turret_takedowns,
    toFloat32(avg(damage_to_objectives / game_minutes)) AS avg_damage_to_objectives,

    -- Vision.
    toFloat32(avg(vision_score / game_minutes)) AS avg_vision_score,
    toFloat32(avg(control_wards_bought / game_minutes)) AS avg_control_wards_bought
FROM assigned
GROUP BY
    championid, teamposition, build,
    bin_idx;

DROP TEMPORARY TABLE IF EXISTS tmp_1vx_participants;
