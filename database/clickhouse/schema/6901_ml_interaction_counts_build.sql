-- noqa: disable=AL09,LT02,LT05,LT08,LT09,RF02,RF03,ST05,ST09
-- Materialise per-game, per-player-token 1vX historical profile features.
--
-- `synergy_1vx` now stores reduced profile features by
-- (championid, teamposition, build, bin_idx), so this build preserves the
-- scaling-bin grain instead of reconstructing raw matchups/wins. Each output
-- row is one match token in one 1vX scaling bin:
--
--   token_idx 0..4 = blue TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY
--   token_idx 5..9 = red  TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY
--   bin_idx   1..4 = 6002_1vx scaling bin
--
-- The fixed 10-token ARRAY JOIN scans ml_game_player_pivot once, avoids the
-- per-row arrayMap/range construction cost, and keeps all four profile bins
-- from the right side via an explicit ALL INNER JOIN.

TRUNCATE TABLE game_data_filtered.ml_interaction_counts;

INSERT INTO game_data_filtered.ml_interaction_counts
(
    matchid,
    token_idx,
    bin_idx,
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
expanded AS (
    SELECT
        p.matchid,
        toUInt16(tupleElement(token, 2)) AS token_idx,
        tupleElement(tupleElement(token, 1), 1) AS championid,
        tupleElement(tupleElement(token, 1), 2) AS teamposition,
        tupleElement(tupleElement(token, 1), 3) AS build
    FROM game_data_filtered.ml_game_player_pivot AS p
    ARRAY JOIN [
        tuple(p.blue_players[1], toUInt16(0)),
        tuple(p.blue_players[2], toUInt16(1)),
        tuple(p.blue_players[3], toUInt16(2)),
        tuple(p.blue_players[4], toUInt16(3)),
        tuple(p.blue_players[5], toUInt16(4)),
        tuple(p.red_players[1], toUInt16(5)),
        tuple(p.red_players[2], toUInt16(6)),
        tuple(p.red_players[3], toUInt16(7)),
        tuple(p.red_players[4], toUInt16(8)),
        tuple(p.red_players[5], toUInt16(9))
    ] AS token
)
SELECT
    e.matchid,
    e.token_idx,
    s.bin_idx,
    s.log_matchups,
    s.win_rate,
    s.avg_gold,
    s.avg_xp,
    s.avg_item_completions,
    s.avg_total_cs,
    s.avg_kills,
    s.avg_kills_assists,
    s.avg_total_damage_dealt,
    s.physical_damage_share,
    s.magic_damage_share,
    s.true_damage_share,
    s.avg_damage_taken,
    s.avg_durability,
    s.damage_to_taken_ratio,
    s.avg_time_ccing_others,
    s.avg_protection,
    s.avg_epic_monster_takedowns,
    s.avg_turret_takedowns,
    s.avg_damage_to_objectives,
    s.avg_vision_score,
    s.avg_control_wards_bought
FROM expanded AS e
ALL INNER JOIN (
    SELECT
        championid,
        teamposition,
        build,
        bin_idx,
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
    FROM game_data_filtered.synergy_1vx
    WHERE split = 'train'
) AS s
    ON
        s.championid = e.championid
        AND s.teamposition = e.teamposition
        AND s.build = e.build
SETTINGS join_algorithm = 'hash';
