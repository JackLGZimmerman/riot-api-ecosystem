-- Reload the 1v1 matchup dictionary from the freshly rebuilt matchup_1v1 table.
-- Run this after 6000_1v1_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.matchup_1v1_dict;
