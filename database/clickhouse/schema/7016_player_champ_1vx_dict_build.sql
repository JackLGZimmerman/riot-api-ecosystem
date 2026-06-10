-- Reload the per-(player, champion) prior dictionary from the freshly rebuilt
-- player_champ_1vx table. Run this after
-- 6031_player_champ_1vx_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.player_champ_1vx_dict;
