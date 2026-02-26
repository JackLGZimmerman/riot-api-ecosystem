CREATE DATABASE IF NOT EXISTS game_data_filtered;

CREATE TABLE IF NOT EXISTS game_data_filtered.valid_game_ids
(
    matchid UInt64
)
ENGINE = MergeTree
ORDER BY matchid;

CREATE TABLE IF NOT EXISTS game_data_filtered.participant_perk_values
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.participant_perk_values AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.participant_perk_ids
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.participant_perk_ids AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.tl_participant_stats
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_participant_stats AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.tl_payload_event
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_payload_event AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.metadata
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.metadata AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON toUInt64OrNull(arrayElement(splitByChar('_', t.matchid), 2)) = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.info
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.info AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.bans
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.bans AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.feats
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.feats AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.objectives
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.objectives AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.participant_stats
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.participant_stats AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.participant_challenges
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.participant_challenges AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.tl_building_kill
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_building_kill AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.tl_champion_kill
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_champion_kill AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.tl_champion_special_kill
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_champion_special_kill AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.tl_dragon_soul_given
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_dragon_soul_given AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.tl_elite_monster_kill
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_elite_monster_kill AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.tl_turret_plate_destroyed
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_turret_plate_destroyed AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.tl_ck_victim_damage_dealt
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_ck_victim_damage_dealt AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;

CREATE TABLE IF NOT EXISTS game_data_filtered.tl_ck_victim_damage_received
ENGINE = MergeTree
ORDER BY matchid
AS
SELECT t.* EXCEPT (run_id)
FROM game_data.tl_ck_victim_damage_received AS t
INNER JOIN game_data_filtered.valid_game_ids AS v
    ON t.matchid = v.matchid
WHERE 0;
