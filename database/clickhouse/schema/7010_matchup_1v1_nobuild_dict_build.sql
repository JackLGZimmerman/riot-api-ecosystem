-- Reload the no-build 1v1 matchup dictionary from the rebuilt table.
-- Run this after 6020_1v1_nobuild_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.matchup_1v1_nobuild_dict;
