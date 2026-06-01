-- Reload the build-sibling 2vx synergy dictionary from the rebuilt table.
-- Run this after 6024_2vx_build_group_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.synergy_2vx_build_group_dict;
