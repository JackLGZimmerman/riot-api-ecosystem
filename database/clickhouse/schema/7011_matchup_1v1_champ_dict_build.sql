-- Reload the champion-pair 1v1 matchup dictionary from the rebuilt table.
-- Run this after 6021_1v1_champ_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.matchup_1v1_champ_dict;
