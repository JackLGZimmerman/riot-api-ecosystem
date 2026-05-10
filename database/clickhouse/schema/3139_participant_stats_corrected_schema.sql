-- Corrected snapshot of game_data.participant_stats with end-of-game
-- "stat padding" removed. Padding is defined as any event whose
-- timestamp is within 15 seconds of the match's tl_game_end timestamp.
--
-- Adjustments applied:
--   CHAMPION_KILL: kills, deaths, assists, *damagedealttochampions, *damagetaken
--   ITEM_PURCHASED: goldspent decremented by item price
--   ITEM_SOLD:      goldspent incremented by 70% of item price (sell value recovered)
--   ITEM_UNDO:      goldspent adjusted by goldgain (reverses previous transaction)
--
-- Populated by 3139_participant_stats_corrected_build.sql.
-- Placed in game_data (pre-filter) so 4000_filter_build.sql applies all filters
-- against the corrected stats rather than the raw values.
DROP TABLE IF EXISTS game_data.participant_stats_corrected;
CREATE TABLE IF NOT EXISTS game_data.participant_stats_corrected
AS game_data.participant_stats
ENGINE = MergeTree
ORDER BY (matchid, participantid, puuid, run_id);
