ALTER TABLE game_data.participant_stats
    ADD COLUMN IF NOT EXISTS causedgameendfromignbsurrender Nullable (UInt8) AFTER gameendedinsurrender,
    ADD COLUMN IF NOT EXISTS waspremadewithignbgameendcauser Nullable (UInt8) AFTER teamignbsurrendered,
    ADD COLUMN IF NOT EXISTS waspremadewithseveretransgressor Nullable (UInt8) AFTER waspremadewithignbgameendcauser,
    ADD COLUMN IF NOT EXISTS wasseveretransgressor Nullable (UInt8) AFTER waspremadewithseveretransgressor;

ALTER TABLE game_data.participant_stats_corrected
    ADD COLUMN IF NOT EXISTS causedgameendfromignbsurrender Nullable (UInt8) AFTER gameendedinsurrender,
    ADD COLUMN IF NOT EXISTS waspremadewithignbgameendcauser Nullable (UInt8) AFTER teamignbsurrendered,
    ADD COLUMN IF NOT EXISTS waspremadewithseveretransgressor Nullable (UInt8) AFTER waspremadewithignbgameendcauser,
    ADD COLUMN IF NOT EXISTS wasseveretransgressor Nullable (UInt8) AFTER waspremadewithseveretransgressor;

ALTER TABLE game_data_filtered.participant_stats
    ADD COLUMN IF NOT EXISTS causedgameendfromignbsurrender Nullable (UInt8) AFTER gameendedinsurrender,
    ADD COLUMN IF NOT EXISTS waspremadewithignbgameendcauser Nullable (UInt8) AFTER teamignbsurrendered,
    ADD COLUMN IF NOT EXISTS waspremadewithseveretransgressor Nullable (UInt8) AFTER waspremadewithignbgameendcauser,
    ADD COLUMN IF NOT EXISTS wasseveretransgressor Nullable (UInt8) AFTER waspremadewithseveretransgressor;
