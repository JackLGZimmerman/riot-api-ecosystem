-- Reload the 2vx team-synergy dictionary from the freshly rebuilt synergy_2vx table.
-- Run this after 6004_2vx_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.synergy_2vx_dict;
