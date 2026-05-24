-- Reload the prior dictionary from the freshly rebuilt synergy_1vx table.
-- Run this after 6003_1vx_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.synergy_1vx_dict;
