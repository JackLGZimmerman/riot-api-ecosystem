-- Reload the no-build 2vx synergy dictionary from the rebuilt table.
-- Run this after 6022_2vx_nobuild_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.synergy_2vx_nobuild_dict;
