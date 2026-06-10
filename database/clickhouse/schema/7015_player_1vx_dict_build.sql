-- Reload the per-player prior dictionary from the freshly rebuilt player_1vx
-- table. Run this after 6030_player_1vx_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.player_1vx_dict;
