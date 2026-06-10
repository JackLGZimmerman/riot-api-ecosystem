-- Reload the per-(player, role) experience dictionary from the freshly
-- rebuilt player_role_1vx table. Run this after
-- 6032_player_role_1vx_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.player_role_1vx_dict;
