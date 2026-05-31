-- Reload the champion-pair 2vx synergy dictionary from the rebuilt table.
-- Run this after 6023_2vx_champ_aggregations_build.sql.
SYSTEM RELOAD DICTIONARY game_data_filtered.synergy_2vx_champ_dict;
