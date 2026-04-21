-- Item purchase frequency per (championid, teamposition).
-- Unpivots item0-item6 into rows and counts how often each item appears.
-- Excludes item_id = 0 (empty slot).
--
-- Requires game_data_filtered.participant_stats to be populated.
--
-- This query does not consult item_value_map_dict: it only measures raw pick
-- frequency by (championid, teamposition).  The (championid, teamposition,
-- itemid) tuples emitted here are the same shape used as the specific key in
-- the value dictionary, so callers can join these counts against that
-- dictionary to see which (champion, position) combinations have specific
-- value overrides vs. fall through to the generic (NULL, NULL) rows.

SELECT
    championid,
    teamposition,
    item_id AS item,
    COUNT() AS pick_count
FROM (
    SELECT
        championid,
        toString(teamposition) AS teamposition,
        arrayJoin([
            toUInt32(item0), toUInt32(item1), toUInt32(item2),
            toUInt32(item3), toUInt32(item4), toUInt32(item5), toUInt32(item6)
        ]) AS item_id
    FROM game_data_filtered.participant_stats
)
WHERE item_id != 0
GROUP BY
    championid,
    teamposition,
    item_id
ORDER BY
    championid ASC,
    teamposition ASC,
    pick_count DESC
