# timeline events

## Grain and design

- All metrics in this file are at the **participant + minute** grain. Every event row already carries the parser-assigned `frame_timestamp` of its enclosing Riot timeline frame, and the build maps each event into one or more `(matchid, frame_timestamp, participantid)` cells.
- Metrics fall into two tiers:
  - **Primitive** (atomic): a single `count`, `sum`, or `avg` taken directly from one event source, scoped to the participants the event mentions. Killer-side primitives use `killerid`; victim-side use `victimid`; assist-side explode `assistingparticipantids`. These are the only metrics the build actually aggregates from raw event tables.
  - **Composite**: a pure formula over primitives at the same `(matchid, frame_timestamp, participantid)`. Composites are ratios, shares, additive totals, and concentration indices. They never re-read source events; the build evaluates them inline once per output row.
- Catalog entries that are originally written with `grouped by (matchid)` or `grouped by (matchid, killerid)` are still expressed at participant + minute grain here. The grouping in the formula column is the *catalog* grain; the build's output grain is always participant + minute, and assists are credited to the assisting participants by exploding `assistingparticipantids`.
- The build reads each raw event table once, aggregates each source to `(matchid, frame_timestamp, participantid)`, and joins those source rollups onto `p_stats` on the shared participant-minute bucket.
- `frame_timestamp` is the canonical minute bucket shared with `tl_participant_stats`; the build intentionally does not re-bucket event rows with `intDiv(timestamp, 60000)`.
- Ratio-style metrics use the documented zero-protection rule `greatest(denominator, 1)`.

## Source families

- `tl_champion_kill`: kill/death/assist counts plus killer-only bounty and kill-streak metrics
- `tl_champion_special_kill`: first-blood, ace, and multi-kill primitives
- `tl_ward_kill`, `tl_ward_placed`, `tl_item_destroyed`, `tl_item_purchased`, `tl_item_undo`: payload-style action counts
- `tl_turret_plate_destroyed`: lane-specific plate pressure primitives
- `tl_building_kill`: lane/building takedown counts plus building bounty gold
- `tl_elite_monster_kill`: objective involvement counts plus killer-side monster bounty gold
- `tl_level_up`: level-up count per participant-minute bucket

| id | name | data_source | description | calculation | version | fields |
|---|---|---|---|---|---|---|
| TLE_R_001 | Champion Kill Event Type | game_data.tl_champion_kill | Event stream for champion kill events. | identity(type) | 1.0.0 | type |
| TLE_R_002 | Building Kill Event Type | game_data.tl_building_kill | Event stream for building destruction events. | identity(type) | 1.0.0 | type |
| TLE_R_003 | Elite Monster Kill Event Type | game_data.tl_elite_monster_kill | Event stream for elite monster takedown events. | identity(type) | 1.0.0 | type |
| TLE_R_004 | Champion Special Kill Event Type | game_data.tl_champion_special_kill | Event stream for champion special kill events. | identity(type) | 1.0.0 | type |
| TLE_R_005 | Dragon Soul Given Event Type | game_data.tl_dragon_soul_given | Event stream for dragon soul assignment events. | identity(type) | 1.0.0 | type |
| TLE_R_006 | Turret Plate Destroyed Event Type | game_data.tl_turret_plate_destroyed | Event stream for turret plate destruction events. | identity(type) | 1.0.0 | type |
| TLE_R_008 | Victim Damage Dealt Event Type | game_data.tl_ck_victim_damage_dealt | Event stream for champion-kill victim damage dealt records. | identity(direction) | 1.0.0 | direction |
| TLE_R_009 | Victim Damage Received Event Type | game_data.tl_ck_victim_damage_received | Event stream for champion-kill victim damage received records. | identity(direction) | 1.0.0 | direction |
| TLE_R_010 | Ward Kill Event Type | game_data.tl_ward_kill | Event stream for ward kill events. | row_exists() | 1.0.0 | matchid |
| TLE_R_011 | Ward Placed Event Type | game_data.tl_ward_placed | Event stream for ward placement events. | row_exists() | 1.0.0 | matchid |
| TLE_R_012 | Game End Event Type | game_data.tl_game_end | Event stream for game end events. | row_exists() | 1.0.0 | matchid |
| TLE_R_013 | Item Destroyed Event Type | game_data.tl_item_destroyed | Event stream for item destroyed events. | row_exists() | 1.0.0 | matchid |
| TLE_R_014 | Item Purchased Event Type | game_data.tl_item_purchased | Event stream for item purchased events. | row_exists() | 1.0.0 | matchid |
| TLE_R_015 | Item Sold Event Type | game_data.tl_item_sold | Event stream for item sold events. | row_exists() | 1.0.0 | matchid |
| TLE_R_016 | Item Undo Event Type | game_data.tl_item_undo | Event stream for item undo events. | row_exists() | 1.0.0 | matchid |
| TLE_R_017 | Level Up Event Type | game_data.tl_level_up | Event stream for level-up events. | row_exists() | 1.0.0 | matchid |
| TLE_R_018 | Pause End Event Type | game_data.tl_pause_end | Event stream for pause-end events. | row_exists() | 1.0.0 | matchid |
| TLE_R_019 | Skill Level Up Event Type | game_data.tl_skill_level_up | Event stream for skill-level-up events. | row_exists() | 1.0.0 | matchid |
| TLE_R_020 | Objective Bounty Prestart Event Type | game_data.tl_objective_bounty_prestart | Event stream for objective bounty prestart events. | row_exists() | 1.0.0 | matchid |
| TLE_R_021 | Feat Update Event Type | game_data.tl_feat_update | Event stream for feat update events. | row_exists() | 1.0.0 | matchid |
| TLE_R_022 | Objective Bounty Finish Event Type | game_data.tl_objective_bounty_finish | Event stream for objective bounty finish events. | row_exists() | 1.0.0 | matchid |
| TLE_R_023 | Champion Transform Event Type | game_data.tl_champion_transform | Event stream for champion transform events. | row_exists() | 1.0.0 | matchid |
| TLE_S_024 | Champion Kill Events Per-Minute Bin | game_data.tl_champion_kill | Per-killer kill count per minute bin. | count() grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | killerid, matchid, frame_timestamp |
| TLE_S_025 | Champion Death Events Per-Minute Bin | game_data.tl_champion_kill | Per-player death count per minute bin. | countIf(victimid = participantid) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | victimid, participantid, matchid, frame_timestamp |
| TLE_S_026 | Champion Assist Events Per-Minute Bin | game_data.tl_champion_kill | Per-player assist count per minute bin from nullable assisting participant IDs. | countIf(has(ifNull(assistingParticipantIds, []), participantid)) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | assistingParticipantIds, participantid, matchid, frame_timestamp |
| TLE_S_027 | Wards Killed Per-Minute Bin | game_data.tl_ward_kill | Per-player ward-kill count per minute bin. | countIf(killerid = participantid) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | killerid, participantid, matchid, frame_timestamp |
| TLE_S_028 | Wards Placed Per-Minute Bin | game_data.tl_ward_placed | Per-player ward-placement count per minute bin. | countIf(creatorid = participantid) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | creatorid, participantid, matchid, frame_timestamp |
| TLE_S_029 | Items Destroyed Per-Minute Bin | game_data.tl_item_destroyed | Per-player item-destroyed count per minute bin. | countIf(participantid = participantid) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | participantid, matchid, frame_timestamp |
| TLE_S_030 | Items Purchased Per-Minute Bin | game_data.tl_item_purchased | Per-player item-purchased count per minute bin. | countIf(participantid = participantid) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | participantid, matchid, frame_timestamp |
| TLE_S_031 | Item Undos Per-Minute Bin | game_data.tl_item_undo | Per-player item-undo count per minute bin. | countIf(participantid = participantid) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | participantid, matchid, frame_timestamp |
| TLE_S_032 | Level Up Timer Seconds | game_data.tl_level_up | Per-player level-up timer in seconds from timeline start. | maxIf(timestamp / 1000, participantid = participantid) | 1.0.0 | timestamp, participantid |
| TLE_S_033 | Total Kill Bounty Gold | game_data.tl_champion_kill | Total kill gold value earned from champion kill events per killer. | sum(bounty) grouped by (matchid, killerid) | 1.0.0 | bounty, killerid, matchid |
| TLE_S_034 | Average Kill Streak Length | game_data.tl_champion_kill | Average kill streak length across a killer's champion-kill events. | avg(killstreaklength) grouped by (matchid, killerid) | 1.0.0 | killstreaklength, killerid, matchid |
| TLE_S_035 | Total Shutdown Bounty Gold | game_data.tl_champion_kill | Total shutdown bounty gold earned by killer from advantaged targets. | sum(shutdownbounty) grouped by (matchid, killerid) | 1.0.0 | shutdownbounty, killerid, matchid |
| TLE_S_036 | Outer Tower Kills Top Lane Sum | game_data.tl_building_kill | Sum of top-lane outer tower kills by killer. | sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'TOP_LANE' AND towertype = 'OUTER_TURRET') grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_037 | Outer Tower Kills Mid Lane Sum | game_data.tl_building_kill | Sum of mid-lane outer tower kills by killer. | sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'MID_LANE' AND towertype = 'OUTER_TURRET') grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_038 | Outer Tower Kills Bot Lane Sum | game_data.tl_building_kill | Sum of bot-lane outer tower kills by killer. | sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'BOT_LANE' AND towertype = 'OUTER_TURRET') grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_039 | Inner Tower Kills Top Lane Sum | game_data.tl_building_kill | Sum of top-lane inner tower kills by killer. | sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'TOP_LANE' AND towertype = 'INNER_TURRET') grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_040 | Inner Tower Kills Mid Lane Sum | game_data.tl_building_kill | Sum of mid-lane inner tower kills by killer. | sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'MID_LANE' AND towertype = 'INNER_TURRET') grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_041 | Inner Tower Kills Bot Lane Sum | game_data.tl_building_kill | Sum of bot-lane inner tower kills by killer. | sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'BOT_LANE' AND towertype = 'INNER_TURRET') grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_042 | Base Tower Kills Top Lane Sum | game_data.tl_building_kill | Sum of top-lane base tower kills by killer. | sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'TOP_LANE' AND towertype = 'BASE_TURRET') grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_043 | Base Tower Kills Mid Lane Sum | game_data.tl_building_kill | Sum of mid-lane base tower kills by killer. | sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'MID_LANE' AND towertype = 'BASE_TURRET') grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_044 | Base Tower Kills Bot Lane Sum | game_data.tl_building_kill | Sum of bot-lane base tower kills by killer. | sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'BOT_LANE' AND towertype = 'BASE_TURRET') grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_045 | Nexus Tower Kills Mid Lane Sum | game_data.tl_building_kill | Sum of mid-lane nexus tower kills by killer (nexus towers are mid-only). | sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'MID_LANE' AND towertype = 'NEXUS_TURRET') grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_046 | Inhibitor Building Kills Top Lane Sum | game_data.tl_building_kill | Sum of top-lane inhibitor building kills by killer. | sumIf(1, buildingtype = 'INHIBITOR_BUILDING' AND lanetype = 'TOP_LANE' AND towertype IS NULL) grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_047 | Inhibitor Building Kills Mid Lane Sum | game_data.tl_building_kill | Sum of mid-lane inhibitor building kills by killer. | sumIf(1, buildingtype = 'INHIBITOR_BUILDING' AND lanetype = 'MID_LANE' AND towertype IS NULL) grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_048 | Inhibitor Building Kills Bot Lane Sum | game_data.tl_building_kill | Sum of bot-lane inhibitor building kills by killer. | sumIf(1, buildingtype = 'INHIBITOR_BUILDING' AND lanetype = 'BOT_LANE' AND towertype IS NULL) grouped by (matchid, killerid) | 1.0.0 | buildingtype, lanetype, towertype, killerid, matchid |
| TLE_S_049 | Total Building Bounty Gold | game_data.tl_building_kill | Total building bounty gold earned from building kill events per killer. | sum(bounty) grouped by (matchid, killerid) | 1.0.0 | bounty, killerid, matchid |
| TLE_S_050 | Total Elite Monster Bounty Gold | game_data.tl_elite_monster_kill | Total bounty gold earned from elite monster kill events per killer. | sum(bounty) grouped by (matchid, killerid) | 1.0.0 | bounty, killerid, matchid |
| TLE_S_051 | Atakhan Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Atakhan events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'ATAKHAN' AND monstersubtype IS NULL) grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_052 | Baron Nashor Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Baron Nashor events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'BARON_NASHOR' AND monstersubtype IS NULL) grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_053 | Dragon Air Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Air Dragon events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'AIR_DRAGON') grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_054 | Dragon Chemtech Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Chemtech Dragon events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'CHEMTECH_DRAGON') grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_055 | Dragon Earth Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Earth Dragon events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'EARTH_DRAGON') grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_056 | Dragon Elder Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Elder Dragon events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'ELDER_DRAGON') grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_057 | Dragon Fire Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Fire Dragon events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'FIRE_DRAGON') grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_058 | Dragon Hextech Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Hextech Dragon events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'HEXTECH_DRAGON') grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_059 | Dragon Water Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Water Dragon events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'WATER_DRAGON') grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_060 | Horde Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Horde events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'HORDE' AND monstersubtype IS NULL) grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_061 | Rift Herald Kill+Assist Involvement Sum | game_data.tl_elite_monster_kill | Total kill-plus-assist involvements for Rift Herald events. | sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'RIFTHERALD' AND monstersubtype IS NULL) grouped by (matchid) | 1.0.0 | monstertype, monstersubtype, assistingparticipantids, matchid |
| TLE_S_062 | First Blood Events Per-Minute Bin | game_data.tl_champion_special_kill | Per-killer first blood count per minute bin. | countIf(killtype = 'KILL_FIRST_BLOOD') grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | killtype, killerid, matchid, frame_timestamp |
| TLE_S_063 | Ace Events Per-Minute Bin | game_data.tl_champion_special_kill | Per-killer ace count per minute bin. | countIf(killtype = 'KILL_ACE') grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | killtype, killerid, matchid, frame_timestamp |
| TLE_S_064 | Multi-Kill Events Per-Minute Bin | game_data.tl_champion_special_kill | Per-killer multi-kill event count per minute bin. | countIf(killtype = 'KILL_MULTI') grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | killtype, killerid, matchid, frame_timestamp |
| TLE_S_065 | Double-Kill Total Kills Per-Minute Sum | game_data.tl_champion_special_kill | Per-killer summed kills contributed by double-kill events per minute bin. | sumIf(multikilllength, killtype = 'KILL_MULTI' AND multikilllength = 2) grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | killtype, multikilllength, killerid, matchid, frame_timestamp |
| TLE_S_066 | Triple-Kill Total Kills Per-Minute Sum | game_data.tl_champion_special_kill | Per-killer summed kills contributed by triple-kill events per minute bin. | sumIf(multikilllength, killtype = 'KILL_MULTI' AND multikilllength = 3) grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | killtype, multikilllength, killerid, matchid, frame_timestamp |
| TLE_S_067 | Quadra-Kill Total Kills Per-Minute Sum | game_data.tl_champion_special_kill | Per-killer summed kills contributed by quadra-kill events per minute bin. | sumIf(multikilllength, killtype = 'KILL_MULTI' AND multikilllength = 4) grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | killtype, multikilllength, killerid, matchid, frame_timestamp |
| TLE_S_068 | Penta-Kill Total Kills Per-Minute Sum | game_data.tl_champion_special_kill | Per-killer summed kills contributed by penta-kill events per minute bin. | sumIf(multikilllength, killtype = 'KILL_MULTI' AND multikilllength = 5) grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | killtype, multikilllength, killerid, matchid, frame_timestamp |
| TLE_S_069 | Top Lane Plates Destroyed Per-Minute Sum | game_data.tl_turret_plate_destroyed | Per-killer summed top-lane turret plates destroyed per minute bin. | sumIf(1, lanetype = 'TOP_LANE') grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | lanetype, killerid, matchid, frame_timestamp |
| TLE_S_070 | Mid Lane Plates Destroyed Per-Minute Sum | game_data.tl_turret_plate_destroyed | Per-killer summed mid-lane turret plates destroyed per minute bin. | sumIf(1, lanetype = 'MID_LANE') grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | lanetype, killerid, matchid, frame_timestamp |
| TLE_S_071 | Bot Lane Plates Destroyed Per-Minute Sum | game_data.tl_turret_plate_destroyed | Per-killer summed bot-lane turret plates destroyed per minute bin. | sumIf(1, lanetype = 'BOT_LANE') grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | lanetype, killerid, matchid, frame_timestamp |
| TLE_S_072 | KDA Per-Minute Bin | derived:TLE_S_024_025_026 | Per-player KDA ratio per minute bin derived from synthetic kill/death/assist bins (with killerid key from 024 aligned to participantid). | (TLE_S_024 + TLE_S_026) / greatest(TLE_S_025, 1) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_024, TLE_S_025, TLE_S_026 |
| TLE_S_073 | KD Per-Minute Bin | derived:TLE_S_024_025 | Per-player KD ratio per minute bin derived from synthetic kill/death bins (with killerid key from 024 aligned to participantid). | TLE_S_024 / greatest(TLE_S_025, 1) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_024, TLE_S_025 |
| TLE_S_075 | KA Per-Minute Bin | derived:TLE_S_024_026 | Per-player KA ratio per minute bin derived from synthetic kill/assist bins (with killerid key from 024 aligned to participantid). | TLE_S_024 / greatest(TLE_S_026, 1) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_024, TLE_S_026 |
| TLE_S_076 | Kill Participation Events Per-Minute Bin | derived:TLE_S_024_026 | Per-player takedown participation count per minute bin derived from kill and assist bins (with killerid key from 024 aligned to participantid). | TLE_S_024 + TLE_S_026 grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_024, TLE_S_026 |
| TLE_S_077 | KDA Activity Per-Minute Bin | derived:TLE_S_024_025_026 | Per-player total kill, death, and assist event activity per minute bin (with killerid key from 024 aligned to participantid). | TLE_S_024 + TLE_S_025 + TLE_S_026 grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_024, TLE_S_025, TLE_S_026 |
| TLE_S_078 | Net Takedown Margin Per-Minute Bin | derived:TLE_S_024_025_026 | Per-player net takedown margin per minute bin, positive when kill-plus-assist involvement exceeds deaths (with killerid key from 024 aligned to participantid). | TLE_S_024 + TLE_S_026 - TLE_S_025 grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_024, TLE_S_025, TLE_S_026 |
| TLE_S_079 | Vision Activity Per-Minute Bin | derived:TLE_S_027_028 | Per-player total ward kill and ward placement activity per minute bin. | TLE_S_027 + TLE_S_028 grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_027, TLE_S_028 |
| TLE_S_080 | Vision Denial Share Per-Minute Bin | derived:TLE_S_027_028 | Share of per-minute vision actions coming from ward kills rather than ward placements. | TLE_S_027 / greatest(TLE_S_027 + TLE_S_028, 1) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_027, TLE_S_028 |
| TLE_S_081 | Item Activity Per-Minute Bin | derived:TLE_S_029_030_031 | Per-player total item destroy, purchase, and undo activity per minute bin. | TLE_S_029 + TLE_S_030 + TLE_S_031 grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_029, TLE_S_030, TLE_S_031 |
| TLE_S_082 | Net Item Purchase Actions Per-Minute Bin | derived:TLE_S_030_031 | Per-player net item purchase actions after item undos per minute bin. | TLE_S_030 - TLE_S_031 grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_030, TLE_S_031 |
| TLE_S_083 | Multi-Kill Kill Share Per-Minute Bin | derived:TLE_S_024_065_066_067_068 | Share of per-minute kills that came from multi-kill chains (with killerid keys from the source metrics aligned to participantid). | (TLE_S_065 + TLE_S_066 + TLE_S_067 + TLE_S_068) / greatest(TLE_S_024, 1) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_024, TLE_S_065, TLE_S_066, TLE_S_067, TLE_S_068 |
| TLE_S_084 | Total Plates Destroyed Per-Minute Sum | derived:TLE_S_069_070_071 | Per-killer total turret plates destroyed across all lanes per minute bin. | TLE_S_069 + TLE_S_070 + TLE_S_071 grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | TLE_S_069, TLE_S_070, TLE_S_071 |
| TLE_S_085 | Total Tower Kills Sum | derived:TLE_S_036_037_038_039_040_041_042_043_044_045 | Total tower kills across all tower tiers and lanes by killer. | TLE_S_036 + TLE_S_037 + TLE_S_038 + TLE_S_039 + TLE_S_040 + TLE_S_041 + TLE_S_042 + TLE_S_043 + TLE_S_044 + TLE_S_045 grouped by (matchid, killerid) | 1.0.0 | TLE_S_036, TLE_S_037, TLE_S_038, TLE_S_039, TLE_S_040, TLE_S_041, TLE_S_042, TLE_S_043, TLE_S_044, TLE_S_045 |
| TLE_S_086 | Total Inhibitor Kills Sum | derived:TLE_S_046_047_048 | Total inhibitor building kills across all lanes by killer. | TLE_S_046 + TLE_S_047 + TLE_S_048 grouped by (matchid, killerid) | 1.0.0 | TLE_S_046, TLE_S_047, TLE_S_048 |
| TLE_S_087 | Total Structure Kills Sum | derived:TLE_S_085_086 | Total tower and inhibitor building kills by killer. | TLE_S_085 + TLE_S_086 grouped by (matchid, killerid) | 1.0.0 | TLE_S_085, TLE_S_086 |
| TLE_S_088 | Total Dragon Kill+Assist Involvement Sum | derived:TLE_S_053_054_055_056_057_058_059 | Total kill-plus-assist involvements across all dragon types. | TLE_S_053 + TLE_S_054 + TLE_S_055 + TLE_S_056 + TLE_S_057 + TLE_S_058 + TLE_S_059 grouped by (matchid) | 1.0.0 | TLE_S_053, TLE_S_054, TLE_S_055, TLE_S_056, TLE_S_057, TLE_S_058, TLE_S_059 |
| TLE_S_089 | Elemental Dragon Kill+Assist Involvement Sum | derived:TLE_S_053_054_055_057_058_059 | Total kill-plus-assist involvements across non-Elder dragon types. | TLE_S_053 + TLE_S_054 + TLE_S_055 + TLE_S_057 + TLE_S_058 + TLE_S_059 grouped by (matchid) | 1.0.0 | TLE_S_053, TLE_S_054, TLE_S_055, TLE_S_057, TLE_S_058, TLE_S_059 |
| TLE_S_090 | Non-Dragon Epic Monster Kill+Assist Involvement Sum | derived:TLE_S_051_052_060_061 | Total kill-plus-assist involvements across non-dragon epic monsters. | TLE_S_051 + TLE_S_052 + TLE_S_060 + TLE_S_061 grouped by (matchid) | 1.0.0 | TLE_S_051, TLE_S_052, TLE_S_060, TLE_S_061 |
| TLE_S_091 | Total Epic Monster Kill+Assist Involvement Sum | derived:TLE_S_051_052_053_054_055_056_057_058_059_060_061 | Total kill-plus-assist involvements across all tracked epic monsters. | TLE_S_051 + TLE_S_052 + TLE_S_053 + TLE_S_054 + TLE_S_055 + TLE_S_056 + TLE_S_057 + TLE_S_058 + TLE_S_059 + TLE_S_060 + TLE_S_061 grouped by (matchid) | 1.0.0 | TLE_S_051, TLE_S_052, TLE_S_053, TLE_S_054, TLE_S_055, TLE_S_056, TLE_S_057, TLE_S_058, TLE_S_059, TLE_S_060, TLE_S_061 |
| TLE_S_092 | Total Objective Bounty Gold | derived:TLE_S_049_050 | Total bounty gold earned from buildings and elite monsters. | TLE_S_049 + TLE_S_050 grouped by (matchid, killerid) | 1.0.0 | TLE_S_049, TLE_S_050 |
| TLE_S_093 | Total Event Bounty Gold | derived:TLE_S_033_049_050 | Total bounty gold earned across champion kills, building kills, and elite monster kills. | TLE_S_033 + TLE_S_049 + TLE_S_050 grouped by (matchid, killerid) | 1.0.0 | TLE_S_033, TLE_S_049, TLE_S_050 |
| TLE_S_094 | Shutdown Bounty Share | derived:TLE_S_033_035 | Share of champion-kill bounty gold coming from shutdown bounty. | TLE_S_035 / greatest(TLE_S_033, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_033, TLE_S_035 |
| TLE_S_095 | Ward Placement To Kill Ratio Per-Minute Bin | derived:TLE_S_027_028 | Ratio of ward placements to ward kills per minute bin. | TLE_S_028 / greatest(TLE_S_027, 1) grouped by (matchid, participantid, frame_timestamp) | 1.0.0 | TLE_S_027, TLE_S_028 |
| TLE_S_096 | Kill To Building Bounty Ratio | derived:TLE_S_033_049 | Ratio of champion kill bounty gold to building bounty gold by killer. | TLE_S_033 / greatest(TLE_S_049, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_033, TLE_S_049 |
| TLE_S_097 | Kill To Monster Bounty Ratio | derived:TLE_S_033_050 | Ratio of champion kill bounty gold to elite monster bounty gold by killer. | TLE_S_033 / greatest(TLE_S_050, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_033, TLE_S_050 |
| TLE_S_098 | Building To Monster Bounty Ratio | derived:TLE_S_049_050 | Ratio of building bounty gold to elite monster bounty gold by killer. | TLE_S_049 / greatest(TLE_S_050, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_049, TLE_S_050 |
| TLE_S_099 | Top Lane Plate Share Per-Minute Bin | derived:TLE_S_069_084 | Share of per-minute plate pressure spent in top lane. | TLE_S_069 / greatest(TLE_S_084, 1) grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | TLE_S_069, TLE_S_084 |
| TLE_S_100 | Mid Lane Plate Share Per-Minute Bin | derived:TLE_S_070_084 | Share of per-minute plate pressure spent in mid lane. | TLE_S_070 / greatest(TLE_S_084, 1) grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | TLE_S_070, TLE_S_084 |
| TLE_S_101 | Bot Lane Plate Share Per-Minute Bin | derived:TLE_S_071_084 | Share of per-minute plate pressure spent in bot lane. | TLE_S_071 / greatest(TLE_S_084, 1) grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | TLE_S_071, TLE_S_084 |
| TLE_S_102 | Top Lane Structure Share | derived:TLE_S_036_039_042_046_087 | Share of total structure takedowns coming from top lane. | (TLE_S_036 + TLE_S_039 + TLE_S_042 + TLE_S_046) / greatest(TLE_S_087, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_036, TLE_S_039, TLE_S_042, TLE_S_046, TLE_S_087 |
| TLE_S_103 | Mid Lane Structure Share | derived:TLE_S_037_040_043_045_047_087 | Share of total structure takedowns coming from mid lane. | (TLE_S_037 + TLE_S_040 + TLE_S_043 + TLE_S_045 + TLE_S_047) / greatest(TLE_S_087, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_037, TLE_S_040, TLE_S_043, TLE_S_045, TLE_S_047, TLE_S_087 |
| TLE_S_104 | Bot Lane Structure Share | derived:TLE_S_038_041_044_048_087 | Share of total structure takedowns coming from bot lane. | (TLE_S_038 + TLE_S_041 + TLE_S_044 + TLE_S_048) / greatest(TLE_S_087, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_038, TLE_S_041, TLE_S_044, TLE_S_048, TLE_S_087 |
| TLE_S_105 | Dragon To Horde Kill+Assist Involvement Ratio | derived:TLE_S_088_060 | Ratio of total dragon objective involvement to horde involvement. | TLE_S_088 / greatest(TLE_S_060, 1) grouped by (matchid) | 1.0.0 | TLE_S_088, TLE_S_060 |
| TLE_S_106 | Dragon To Herald Kill+Assist Involvement Ratio | derived:TLE_S_088_061 | Ratio of total dragon objective involvement to Rift Herald involvement. | TLE_S_088 / greatest(TLE_S_061, 1) grouped by (matchid) | 1.0.0 | TLE_S_088, TLE_S_061 |
| TLE_S_107 | Horde To Herald Kill+Assist Involvement Ratio | derived:TLE_S_060_061 | Ratio of horde objective involvement to Rift Herald involvement. | TLE_S_060 / greatest(TLE_S_061, 1) grouped by (matchid) | 1.0.0 | TLE_S_060, TLE_S_061 |
| TLE_S_108 | Dragon Objective Trio Share | derived:TLE_S_088_060_061 | Share of dragon-horde-herald objective involvement coming from dragons. | TLE_S_088 / greatest(TLE_S_088 + TLE_S_060 + TLE_S_061, 1) grouped by (matchid) | 1.0.0 | TLE_S_088, TLE_S_060, TLE_S_061 |
| TLE_S_109 | Horde Objective Trio Share | derived:TLE_S_088_060_061 | Share of dragon-horde-herald objective involvement coming from horde. | TLE_S_060 / greatest(TLE_S_088 + TLE_S_060 + TLE_S_061, 1) grouped by (matchid) | 1.0.0 | TLE_S_088, TLE_S_060, TLE_S_061 |
| TLE_S_110 | Herald Objective Trio Share | derived:TLE_S_088_060_061 | Share of dragon-horde-herald objective involvement coming from Rift Herald. | TLE_S_061 / greatest(TLE_S_088 + TLE_S_060 + TLE_S_061, 1) grouped by (matchid) | 1.0.0 | TLE_S_088, TLE_S_060, TLE_S_061 |
| TLE_S_111 | Objective Bounty Share Of Event Gold | derived:TLE_S_092_093 | Share of total event bounty gold coming from objective bounty sources. | TLE_S_092 / greatest(TLE_S_093, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_092, TLE_S_093 |
| TLE_S_112 | Building Bounty Share Of Event Gold | derived:TLE_S_049_093 | Share of total event bounty gold coming from building kills. | TLE_S_049 / greatest(TLE_S_093, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_049, TLE_S_093 |
| TLE_S_113 | Monster Bounty Share Of Event Gold | derived:TLE_S_050_093 | Share of total event bounty gold coming from elite monster kills. | TLE_S_050 / greatest(TLE_S_093, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_050, TLE_S_093 |
| TLE_S_114 | Kill Bounty Share Of Event Gold | derived:TLE_S_033_093 | Share of total event bounty gold coming from champion kills. | TLE_S_033 / greatest(TLE_S_093, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_033, TLE_S_093 |
| TLE_S_115 | Shutdown Share Of Event Gold | derived:TLE_S_035_093 | Share of total event bounty gold coming from shutdown bounty gold. | TLE_S_035 / greatest(TLE_S_093, 1) grouped by (matchid, killerid) | 1.0.0 | TLE_S_035, TLE_S_093 |
| TLE_S_116 | Objective Trio Concentration | derived:TLE_S_108_109_110 | Concentration index for dragon-horde-herald objective involvement, higher when one objective family dominates. | TLE_S_108 * TLE_S_108 + TLE_S_109 * TLE_S_109 + TLE_S_110 * TLE_S_110 grouped by (matchid) | 1.0.0 | TLE_S_108, TLE_S_109, TLE_S_110 |
| TLE_S_117 | Structure Lane Concentration | derived:TLE_S_102_103_104 | Concentration index for structure takedowns by lane, higher when takedowns are focused into a single lane. | TLE_S_102 * TLE_S_102 + TLE_S_103 * TLE_S_103 + TLE_S_104 * TLE_S_104 grouped by (matchid, killerid) | 1.0.0 | TLE_S_102, TLE_S_103, TLE_S_104 |
| TLE_S_118 | Plate Lane Concentration Per-Minute Bin | derived:TLE_S_099_100_101 | Concentration index for per-minute plate pressure by lane, higher when plate pressure is focused into one lane within the bin. | TLE_S_099 * TLE_S_099 + TLE_S_100 * TLE_S_100 + TLE_S_101 * TLE_S_101 grouped by (matchid, killerid, frame_timestamp) | 1.0.0 | TLE_S_099, TLE_S_100, TLE_S_101 |
| TLE_S_119 | Event Gold Source Concentration | derived:TLE_S_112_113_114 | Concentration index for event gold sources, higher when total event bounty gold is dominated by one source family. | TLE_S_112 * TLE_S_112 + TLE_S_113 * TLE_S_113 + TLE_S_114 * TLE_S_114 grouped by (matchid, killerid) | 1.0.0 | TLE_S_112, TLE_S_113, TLE_S_114 |

## Metric List

These are the timeline event metrics. Raw metrics are listed together; derived metrics are grouped by meaning. New derived metrics added in this pass are marked `(new)`.

Where per-killer event bins are combined with per-player bins, `killerid` from kill-side metrics is aligned to `participantid`, following the existing KDA metrics.

### Raw

- [TLE_R_001] `championKillEventType`: `identity(type)`
- [TLE_R_002] `buildingKillEventType`: `identity(type)`
- [TLE_R_003] `eliteMonsterKillEventType`: `identity(type)`
- [TLE_R_004] `championSpecialKillEventType`: `identity(type)`
- [TLE_R_005] `dragonSoulGivenEventType`: `identity(type)`
- [TLE_R_006] `turretPlateDestroyedEventType`: `identity(type)`
- [TLE_R_007] `payloadEventType`: `identity(type)`
- [TLE_R_008] `victimDamageDealtEventType`: `identity(type)`
- [TLE_R_009] `victimDamageReceivedEventType`: `identity(type)`
- [TLE_R_010] `payloadWardKillType`: `type = 'WARD_KILL'`
- [TLE_R_011] `payloadWardPlacedType`: `type = 'WARD_PLACED'`
- [TLE_R_012] `payloadGameEndType`: `type = 'GAME_END'`
- [TLE_R_013] `payloadItemDestroyedType`: `type = 'ITEM_DESTROYED'`
- [TLE_R_014] `payloadItemPurchasedType`: `type = 'ITEM_PURCHASED'`
- [TLE_R_015] `payloadItemSoldType`: `type = 'ITEM_SOLD'`
- [TLE_R_016] `payloadItemUndoType`: `type = 'ITEM_UNDO'`
- [TLE_R_017] `payloadLevelUpType`: `type = 'LEVEL_UP'`
- [TLE_R_018] `payloadPauseEndType`: `type = 'PAUSE_END'`
- [TLE_R_019] `payloadSkillLevelUpType`: `type = 'SKILL_LEVEL_UP'`
- [TLE_R_020] `payloadObjectiveBountyPrestartType`: `type = 'OBJECTIVE_BOUNTY_PRESTART'`
- [TLE_R_021] `payloadFeatUpdateType`: `type = 'FEAT_UPDATE'`
- [TLE_R_022] `payloadObjectiveBountyFinishType`: `type = 'OBJECTIVE_BOUNTY_FINISH'`
- [TLE_R_023] `payloadChampionTransformType`: `type = 'CHAMPION_TRANSFORM'`

### Derived Per-Minute Activity

- [TLE_S_024] `championKillEventsPerMinuteBin`: `count() grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_025] `championDeathEventsPerMinuteBin`: `countIf(victimid = participantid) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_026] `championAssistEventsPerMinuteBin`: `countIf(has(ifNull(assistingParticipantIds, []), participantid)) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_062] `firstBloodEventsPerMinuteBin`: `countIf(killtype = 'KILL_FIRST_BLOOD') grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_063] `aceEventsPerMinuteBin`: `countIf(killtype = 'KILL_ACE') grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_064] `multiKillEventsPerMinuteBin`: `countIf(killtype = 'KILL_MULTI') grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_027] `wardsKilledPerMinuteBin`: `countIf(type = 'WARD_KILL' AND JSONExtractInt(payload, 'killerId') = participantid) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_028] `wardsPlacedPerMinuteBin`: `countIf(type = 'WARD_PLACED' AND JSONExtractInt(payload, 'creatorId') = participantid) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_029] `itemsDestroyedPerMinuteBin`: `countIf(type = 'ITEM_DESTROYED' AND JSONExtractInt(payload, 'participantId') = participantid) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_030] `itemsPurchasedPerMinuteBin`: `countIf(type = 'ITEM_PURCHASED' AND JSONExtractInt(payload, 'participantId') = participantid) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_031] `itemUndosPerMinuteBin`: `countIf(type = 'ITEM_UNDO' AND JSONExtractInt(payload, 'participantId') = participantid) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_076] `killParticipationEventsPerMinuteBin` (new): `TLE_S_024 + TLE_S_026 grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_077] `kdaActivityPerMinuteBin` (new): `TLE_S_024 + TLE_S_025 + TLE_S_026 grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_078] `netTakedownMarginPerMinuteBin` (new): `TLE_S_024 + TLE_S_026 - TLE_S_025 grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_079] `visionActivityPerMinuteBin` (new): `TLE_S_027 + TLE_S_028 grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_081] `itemActivityPerMinuteBin` (new): `TLE_S_029 + TLE_S_030 + TLE_S_031 grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_082] `netItemPurchaseActionsPerMinuteBin` (new): `TLE_S_030 - TLE_S_031 grouped by (matchid, participantid, frame_timestamp)`

### Derived Timers And Averages

- [TLE_S_032] `levelUpTimerSeconds`: `maxIf(timestamp / 1000, type = 'LEVEL_UP' AND JSONExtractInt(payload, 'participantId') = participantid)`
- [TLE_S_034] `averageKillStreakLength`: `avg(killstreaklength) grouped by (matchid, killerid)`

### Derived Direct Totals

- [TLE_S_065] `doubleKillTotalKillsPerMinuteSum`: `sumIf(multikilllength, killtype = 'KILL_MULTI' AND multikilllength = 2) grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_066] `tripleKillTotalKillsPerMinuteSum`: `sumIf(multikilllength, killtype = 'KILL_MULTI' AND multikilllength = 3) grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_067] `quadraKillTotalKillsPerMinuteSum`: `sumIf(multikilllength, killtype = 'KILL_MULTI' AND multikilllength = 4) grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_068] `pentaKillTotalKillsPerMinuteSum`: `sumIf(multikilllength, killtype = 'KILL_MULTI' AND multikilllength = 5) grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_069] `topLanePlatesDestroyedPerMinuteSum`: `sumIf(1, lanetype = 'TOP_LANE') grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_070] `midLanePlatesDestroyedPerMinuteSum`: `sumIf(1, lanetype = 'MID_LANE') grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_071] `botLanePlatesDestroyedPerMinuteSum`: `sumIf(1, lanetype = 'BOT_LANE') grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_033] `totalKillBountyGold`: `sum(bounty) grouped by (matchid, killerid)`
- [TLE_S_035] `totalShutdownBountyGold`: `sum(shutdownbounty) grouped by (matchid, killerid)`
- [TLE_S_049] `totalBuildingBountyGold`: `sum(bounty) grouped by (matchid, killerid)`
- [TLE_S_050] `totalEliteMonsterBountyGold`: `sum(bounty) grouped by (matchid, killerid)`
- [TLE_S_036] `outerTowerKillsTopLaneSum`: `sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'TOP_LANE' AND towertype = 'OUTER_TURRET') grouped by (matchid, killerid)`
- [TLE_S_037] `outerTowerKillsMidLaneSum`: `sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'MID_LANE' AND towertype = 'OUTER_TURRET') grouped by (matchid, killerid)`
- [TLE_S_038] `outerTowerKillsBotLaneSum`: `sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'BOT_LANE' AND towertype = 'OUTER_TURRET') grouped by (matchid, killerid)`
- [TLE_S_039] `innerTowerKillsTopLaneSum`: `sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'TOP_LANE' AND towertype = 'INNER_TURRET') grouped by (matchid, killerid)`
- [TLE_S_040] `innerTowerKillsMidLaneSum`: `sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'MID_LANE' AND towertype = 'INNER_TURRET') grouped by (matchid, killerid)`
- [TLE_S_041] `innerTowerKillsBotLaneSum`: `sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'BOT_LANE' AND towertype = 'INNER_TURRET') grouped by (matchid, killerid)`
- [TLE_S_042] `baseTowerKillsTopLaneSum`: `sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'TOP_LANE' AND towertype = 'BASE_TURRET') grouped by (matchid, killerid)`
- [TLE_S_043] `baseTowerKillsMidLaneSum`: `sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'MID_LANE' AND towertype = 'BASE_TURRET') grouped by (matchid, killerid)`
- [TLE_S_044] `baseTowerKillsBotLaneSum`: `sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'BOT_LANE' AND towertype = 'BASE_TURRET') grouped by (matchid, killerid)`
- [TLE_S_045] `nexusTowerKillsMidLaneSum`: `sumIf(1, buildingtype = 'TOWER_BUILDING' AND lanetype = 'MID_LANE' AND towertype = 'NEXUS_TURRET') grouped by (matchid, killerid)`
- [TLE_S_046] `inhibitorBuildingKillsTopLaneSum`: `sumIf(1, buildingtype = 'INHIBITOR_BUILDING' AND lanetype = 'TOP_LANE' AND towertype IS NULL) grouped by (matchid, killerid)`
- [TLE_S_047] `inhibitorBuildingKillsMidLaneSum`: `sumIf(1, buildingtype = 'INHIBITOR_BUILDING' AND lanetype = 'MID_LANE' AND towertype IS NULL) grouped by (matchid, killerid)`
- [TLE_S_048] `inhibitorBuildingKillsBotLaneSum`: `sumIf(1, buildingtype = 'INHIBITOR_BUILDING' AND lanetype = 'BOT_LANE' AND towertype IS NULL) grouped by (matchid, killerid)`
- [TLE_S_051] `atakhanKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'ATAKHAN' AND monstersubtype IS NULL) grouped by (matchid)`
- [TLE_S_052] `baronNashorKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'BARON_NASHOR' AND monstersubtype IS NULL) grouped by (matchid)`
- [TLE_S_053] `dragonAirKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'AIR_DRAGON') grouped by (matchid)`
- [TLE_S_054] `dragonChemtechKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'CHEMTECH_DRAGON') grouped by (matchid)`
- [TLE_S_055] `dragonEarthKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'EARTH_DRAGON') grouped by (matchid)`
- [TLE_S_056] `dragonElderKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'ELDER_DRAGON') grouped by (matchid)`
- [TLE_S_057] `dragonFireKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'FIRE_DRAGON') grouped by (matchid)`
- [TLE_S_058] `dragonHextechKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'HEXTECH_DRAGON') grouped by (matchid)`
- [TLE_S_059] `dragonWaterKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'DRAGON' AND monstersubtype = 'WATER_DRAGON') grouped by (matchid)`
- [TLE_S_060] `hordeKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'HORDE' AND monstersubtype IS NULL) grouped by (matchid)`
- [TLE_S_061] `riftHeraldKillAssistInvolvementSum`: `sumIf(1 + length(ifNull(assistingparticipantids, [])), monstertype = 'RIFTHERALD' AND monstersubtype IS NULL) grouped by (matchid)`

### Derived Aggregated Totals

- [TLE_S_084] `totalPlatesDestroyedPerMinuteSum` (new): `TLE_S_069 + TLE_S_070 + TLE_S_071 grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_092] `totalObjectiveBountyGold` (new): `TLE_S_049 + TLE_S_050 grouped by (matchid, killerid)`
- [TLE_S_093] `totalEventBountyGold` (new): `TLE_S_033 + TLE_S_049 + TLE_S_050 grouped by (matchid, killerid)`
- [TLE_S_085] `totalTowerKillsSum` (new): `TLE_S_036 + TLE_S_037 + TLE_S_038 + TLE_S_039 + TLE_S_040 + TLE_S_041 + TLE_S_042 + TLE_S_043 + TLE_S_044 + TLE_S_045 grouped by (matchid, killerid)`
- [TLE_S_086] `totalInhibitorKillsSum` (new): `TLE_S_046 + TLE_S_047 + TLE_S_048 grouped by (matchid, killerid)`
- [TLE_S_087] `totalStructureKillsSum` (new): `TLE_S_085 + TLE_S_086 grouped by (matchid, killerid)`
- [TLE_S_088] `totalDragonKillAssistInvolvementSum` (new): `TLE_S_053 + TLE_S_054 + TLE_S_055 + TLE_S_056 + TLE_S_057 + TLE_S_058 + TLE_S_059 grouped by (matchid)`
- [TLE_S_089] `elementalDragonKillAssistInvolvementSum` (new): `TLE_S_053 + TLE_S_054 + TLE_S_055 + TLE_S_057 + TLE_S_058 + TLE_S_059 grouped by (matchid)`
- [TLE_S_090] `nonDragonEpicMonsterKillAssistInvolvementSum` (new): `TLE_S_051 + TLE_S_052 + TLE_S_060 + TLE_S_061 grouped by (matchid)`
- [TLE_S_091] `totalEpicMonsterKillAssistInvolvementSum` (new): `TLE_S_051 + TLE_S_052 + TLE_S_053 + TLE_S_054 + TLE_S_055 + TLE_S_056 + TLE_S_057 + TLE_S_058 + TLE_S_059 + TLE_S_060 + TLE_S_061 grouped by (matchid)`

### Derived Ratios

- [TLE_S_072] `kdaPerMinuteBin`: `(TLE_S_024 + TLE_S_026) / greatest(TLE_S_025, 1) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_073] `kdPerMinuteBin`: `TLE_S_024 / greatest(TLE_S_025, 1) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_075] `kaPerMinuteBin`: `TLE_S_024 / greatest(TLE_S_026, 1) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_083] `multiKillKillSharePerMinuteBin` (new): `(TLE_S_065 + TLE_S_066 + TLE_S_067 + TLE_S_068) / greatest(TLE_S_024, 1) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_080] `visionDenialSharePerMinuteBin` (new): `TLE_S_027 / greatest(TLE_S_027 + TLE_S_028, 1) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_095] `wardPlacementToKillRatioPerMinuteBin` (new): `TLE_S_028 / greatest(TLE_S_027, 1) grouped by (matchid, participantid, frame_timestamp)`
- [TLE_S_094] `shutdownBountyShare` (new): `TLE_S_035 / greatest(TLE_S_033, 1) grouped by (matchid, killerid)`
- [TLE_S_096] `killToBuildingBountyRatio` (new): `TLE_S_033 / greatest(TLE_S_049, 1) grouped by (matchid, killerid)`
- [TLE_S_097] `killToMonsterBountyRatio` (new): `TLE_S_033 / greatest(TLE_S_050, 1) grouped by (matchid, killerid)`
- [TLE_S_098] `buildingToMonsterBountyRatio` (new): `TLE_S_049 / greatest(TLE_S_050, 1) grouped by (matchid, killerid)`
- [TLE_S_099] `topLanePlateSharePerMinuteBin` (new): `TLE_S_069 / greatest(TLE_S_084, 1) grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_100] `midLanePlateSharePerMinuteBin` (new): `TLE_S_070 / greatest(TLE_S_084, 1) grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_101] `botLanePlateSharePerMinuteBin` (new): `TLE_S_071 / greatest(TLE_S_084, 1) grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_102] `topLaneStructureShare` (new): `(TLE_S_036 + TLE_S_039 + TLE_S_042 + TLE_S_046) / greatest(TLE_S_087, 1) grouped by (matchid, killerid)`
- [TLE_S_103] `midLaneStructureShare` (new): `(TLE_S_037 + TLE_S_040 + TLE_S_043 + TLE_S_045 + TLE_S_047) / greatest(TLE_S_087, 1) grouped by (matchid, killerid)`
- [TLE_S_104] `botLaneStructureShare` (new): `(TLE_S_038 + TLE_S_041 + TLE_S_044 + TLE_S_048) / greatest(TLE_S_087, 1) grouped by (matchid, killerid)`
- [TLE_S_105] `dragonToHordeKillAssistInvolvementRatio` (new): `TLE_S_088 / greatest(TLE_S_060, 1) grouped by (matchid)`
- [TLE_S_106] `dragonToHeraldKillAssistInvolvementRatio` (new): `TLE_S_088 / greatest(TLE_S_061, 1) grouped by (matchid)`
- [TLE_S_107] `hordeToHeraldKillAssistInvolvementRatio` (new): `TLE_S_060 / greatest(TLE_S_061, 1) grouped by (matchid)`
- [TLE_S_108] `dragonObjectiveTrioShare` (new): `TLE_S_088 / greatest(TLE_S_088 + TLE_S_060 + TLE_S_061, 1) grouped by (matchid)`
- [TLE_S_109] `hordeObjectiveTrioShare` (new): `TLE_S_060 / greatest(TLE_S_088 + TLE_S_060 + TLE_S_061, 1) grouped by (matchid)`
- [TLE_S_110] `heraldObjectiveTrioShare` (new): `TLE_S_061 / greatest(TLE_S_088 + TLE_S_060 + TLE_S_061, 1) grouped by (matchid)`
- [TLE_S_111] `objectiveBountyShareOfEventGold` (new): `TLE_S_092 / greatest(TLE_S_093, 1) grouped by (matchid, killerid)`
- [TLE_S_112] `buildingBountyShareOfEventGold` (new): `TLE_S_049 / greatest(TLE_S_093, 1) grouped by (matchid, killerid)`
- [TLE_S_113] `monsterBountyShareOfEventGold` (new): `TLE_S_050 / greatest(TLE_S_093, 1) grouped by (matchid, killerid)`
- [TLE_S_114] `killBountyShareOfEventGold` (new): `TLE_S_033 / greatest(TLE_S_093, 1) grouped by (matchid, killerid)`
- [TLE_S_115] `shutdownShareOfEventGold` (new): `TLE_S_035 / greatest(TLE_S_093, 1) grouped by (matchid, killerid)`

### Derived Concentration

- [TLE_S_116] `objectiveTrioConcentration` (new): `TLE_S_108 * TLE_S_108 + TLE_S_109 * TLE_S_109 + TLE_S_110 * TLE_S_110 grouped by (matchid)`
- [TLE_S_117] `structureLaneConcentration` (new): `TLE_S_102 * TLE_S_102 + TLE_S_103 * TLE_S_103 + TLE_S_104 * TLE_S_104 grouped by (matchid, killerid)`
- [TLE_S_118] `plateLaneConcentrationPerMinuteBin` (new): `TLE_S_099 * TLE_S_099 + TLE_S_100 * TLE_S_100 + TLE_S_101 * TLE_S_101 grouped by (matchid, killerid, frame_timestamp)`
- [TLE_S_119] `eventGoldSourceConcentration` (new): `TLE_S_112 * TLE_S_112 + TLE_S_113 * TLE_S_113 + TLE_S_114 * TLE_S_114 grouped by (matchid, killerid)`
