# Filtering Architecture

## Objective

Identify and exclude low-quality or bot-like games from `game_data.participant_stats`,
then produce `game_data_filtered.*` — a persistent, byte-for-byte copy of **every**
`game_data.*` table containing only rows from valid matches.

Correctness goal: no source table is skipped. Performance goal: the copy is a
single streaming `SEMI JOIN` per table, with `valid_game_ids` as the small
build-side hash table.

## Pipeline at a glance

```text
game_data.participant_stats
  │
  ├─► STAGE 1 — filter_stg_participant_flags         (11 cheap per-participant flags, 1 scan)
  │     └─► filter_stg_stage1_valid_matchids
  │
  ├─► STAGE 2 — filter_stg_rare_roles                (rare (champion, role) in stage-1 pool)
  │     └─► filter_stg_stage2_valid_matchids
  │
  ├─► STAGE 3 — filter_stg_rare_builds               (rare (champion, role, build) in stage-2 pool)
  │
  ├─► filter_stg_game_flags                          (rollup: stage1 + rare_roles + rare_builds)
  ├─► filter_result                                  (bitmask, one row per participant)
  │
  └─► game_data_filtered.valid_game_ids
        └─► game_data_filtered.*                     (SEMI JOIN copy of every game_data.* table)
```

Stages 2 and 3 recompute their rarity thresholds against the pool surviving the
previous stage, which keeps the "rare" cutoff meaningful after noise has been
stripped.

## Files

| File | Role |
| --- | --- |
| `4000_filter_schema.sql` | DROP + CREATE for all `filter_stg_*` + `filter_result` tables |
| `4001_filter_build.sql` | Populate all three stages, rollups, and `filter_result` |
| `5000_create_filtered_db_schema.sql` | `game_data_filtered` database + DROP/CREATE for persistent copies |
| `5001_valid_game_ids_schema.sql` | `game_data_filtered.valid_game_ids` (not dropped by 5000) |
| `5002_valid_game_ids_build.sql` | Populate `valid_game_ids` from `filter_stg_game_flags` |
| `5003_filtered_tables_build.sql` | Populate every `game_data_filtered.*` via `SEMI JOIN valid_game_ids` |
| `5130 / 5131`, `5132 / 5133` | Derived analytical tables on top of `game_data_filtered.*` |
| `analytics_builds/8xxx_*.sql` | Human-facing inspection / reporting queries |

## Stage objects

### Helpers (built before stage 1)

- `filter_stg_player_winrates` — latest per-`puuid` snapshot from `game_data.players`
  via `argMax(..., updated_at)`.
- `filter_stg_team_flags` — per-team aggregates (kills, damage, CS/min, etc.)
  self-joined against the enemy team for relative stats.

**Stage 1 — `filter_stg_participant_flags`** — one row per
`(matchid, teamid, participantid)`, 11 flag columns. Built from a single scan of
`participant_stats` joined to the two helpers plus `game_data.info.gameduration`
(no `max(timeplayed)` aggregate). Rolled up into
`filter_stg_stage1_valid_matchids` (matchids where every flag is 0).

**Stage 2 — `filter_stg_rare_roles`** — `participant_stats` SEMI-joined to
`stage1_valid_matchids`, with rare role detection. A `(championid, teamposition)`
is rare if it's < 0.4% of that champion's picks in the stage-1 pool; a player is
flagged if they have < 30 picks on such a combo in that pool. The LEFT ANTI JOIN
of stage-1-valid against this table is `filter_stg_stage2_valid_matchids`.

**Stage 3 — `filter_stg_rare_builds`** — `participant_stats` item slots
`item0`–`item6`, plus optional `roleBoundItem` when present, SEMI-joined to
`stage2_valid_matchids`, labelled via `item_value_map_dict` on
`(championid, teamposition, highest_value_label)`. Flags participants whose
build label appears < 8 times in the stage-2 pool for their (champion, role).

### Rollups

- `filter_stg_game_flags` — one row per matchid, 14 flag columns +
  `any_filter_triggered`.
- `filter_result` — one row per participant; `rule_mask` is a bitmask packed
  from the player/team/game masks. `rule_mask = 0 ↔ is_valid = 1`.

## Filtered dataset

- `game_data_filtered.valid_game_ids` — one row per matchid where
  `any_filter_triggered = 0`.
- `game_data_filtered.*` — persistent MergeTree copies, each populated by the
  pattern below. `valid_game_ids` is small and matchid-sorted, so it fits in
  memory as the build side and the copy is a single streaming scan of the
  source.

```sql
INSERT INTO game_data_filtered.<t>
SELECT t.*
FROM game_data.<t> AS t
SEMI JOIN game_data_filtered.valid_game_ids AS v ON t.matchid = v.matchid;
```

## Payload tables

### Current state

Two `game_data.*` tables carry an Object-typed dynamic `JSON` payload column:

| Table | Payload column | Current handling |
| --- | --- | --- |
| `participant_challenges` | `payload JSON` (flat dict of ~100 numeric challenge scores) | Copied, but only by splitting the source into 4 `cityHash64(matchid) % 4` buckets with `max_threads=1` — the dynamic-JSON column blows the 4.5 GiB container cap in a single streaming read |
| `tl_payload_event` | `payload JSON` (14 timeline event types lumped together) | **Skipped entirely** — `INSERT ... SELECT` trips `LOGICAL_ERROR (Code 49)` on Object-column serialisation. `game_data_filtered.tl_payload_event` exists but is never populated |

Both issues trace to the same root cause: ClickHouse's dynamic `JSON` /
`Object` column type. It's not designed to be streamed through
`INSERT ... SELECT`, it compresses poorly, and its memory footprint during
copy is unpredictable.

### Recommendation

Eliminate the dynamic JSON columns. After this change, `5003_filtered_tables_build.sql`
becomes a uniform loop — one `TRUNCATE + SEMI JOIN` per table, no hash
chunking, no skip.

#### 1. Split `tl_payload_event` into per-event-type tables

All 14 event types already have fixed, known payload shapes from the Riot
timeline API. Model them the same way we already model `tl_building_kill`,
`tl_champion_kill`, `tl_turret_plate_destroyed`, etc. — one table per event
type with explicit typed columns.

Proposed tables (fields drawn from the Riot schema):

| Table | Columns (beyond `run_id, matchid, frame_timestamp, timestamp`) |
| --- | --- |
| `tl_ward_placed` | `creatorid UInt8, wardtype LowCardinality(String)` |
| `tl_ward_kill` | `killerid UInt8, wardtype LowCardinality(String)` |
| `tl_item_purchased` | `participantid UInt8, itemid UInt32` |
| `tl_item_sold` | `participantid UInt8, itemid UInt32` |
| `tl_item_destroyed` | `participantid UInt8, itemid UInt32` |
| `tl_item_undo` | `participantid UInt8, beforeid UInt32, afterid UInt32, goldgain Int32` |
| `tl_level_up` | `participantid UInt8, level UInt8` |
| `tl_skill_level_up` | `participantid UInt8, skillslot UInt8, leveluptype LowCardinality(String)` |
| `tl_pause_end` | `realtimestamp UInt64` |
| `tl_game_end` | `winningteam UInt8, gameid UInt64, realtimestamp UInt64` |
| `tl_objective_bounty_prestart` | `teamid UInt8, actualstarttime UInt64` |
| `tl_objective_bounty_finish` | `teamid UInt8` |
| `tl_feat_update` | `teamid UInt8, feattype LowCardinality(String), featvalue Int32` |
| `tl_champion_transform` | `participantid UInt8, transformtype LowCardinality(String)` |

Why this is strictly better than the current `tl_payload_event`:

| Axis | `tl_payload_event` (JSON) | Per-event hot tables |
| --- | --- | --- |
| Filterable by `valid_game_ids`? | No — copy crashes | Yes — plain SEMI JOIN, identical to every other table |
| Storage | JSON stores every key name on every row; poor compression | Columnar; each field's name stored once |
| CPU to filter/copy | Fails; or if forced, full dynamic-schema reserialisation | Cheap: source is already partitioned in practice by event type, per-table copies stream in fixed-schema blocks |
| Memory during copy | Unbounded (the reason `participant_challenges` needs 4-way chunking) | Bounded — we can remove the memory-safe SETTINGS |
| Query ergonomics | `payload.itemid`, `payload.killerid` — typeless, no pushdown on payload fields | Direct typed columns with primary-key pruning |
| Trade-off | None in its favour at this scale | More tables (14 instead of 1); schema migration required; ingest orchestrator gains 14 table-specs |

#### 2. Replace `participant_challenges.payload JSON` with a typed container

Two options, pick by how stable the challenge set is:

- **Option A (recommended if the key set is stable):** materialise the ~100
  challenges as explicit `Float32` columns. Best compression, best pushdown,
  trivially filterable, identical SEMI-JOIN copy as everything else.
- **Option B (if keys drift across patches):** change the column to
  `payload Map(String, Float32)`. Still a typed column — no Object serialiser —
  so `INSERT ... SELECT` streams normally. Filtering/copy becomes a plain
  SEMI JOIN and the `cityHash64 % 4` chunking in `5003` disappears.

Either way, the 4-way hash chunking loop in `5003` collapses into a single
`INSERT ... SELECT`.

### Summary of payload decisions

| Table                    | Stay as `JSON`? | Split?                              | Column form                                                  |
| ------------------------ | --------------- | ----------------------------------- | ------------------------------------------------------------ |
| `tl_payload_event`       | No              | **Yes — one table per event type**  | Explicit typed columns                                       |
| `participant_challenges` | No              | No (single table per participant)   | Explicit columns if stable, otherwise `Map(String, Float32)` |

## Bitmask (`rule_mask`)

- `player_rule_mask` — flags for this specific participant.
- `team_rule_mask` — team-level flags (same for all participants on a team).
- `game_rule_mask` — game-level flags (same for all participants in a game).
- `rule_mask` — game-level aggregate; fast `WHERE rule_mask = 0` for is-valid.

## Rule semantics

### Player-level (bit → column)

| Bit | Weight | Column | Rule |
|-----|--------|--------|------|
| 0 | 1 | `player_low_kda` | `(kills + assists) * 6 < deaths` |
| 1 | 2 | `player_gold_spent` | `goldspent * 100 < goldearned * 50` (spent < 50% earned) |
| 2 | 4 | `no_contribution_kda` | `kills + assists = 0 AND deaths > 4` |
| 3 | 8 | `bad_summoner_usage` | `summoner1casts = 0 OR summoner2casts = 0` |
| 4 | 16 | `player_high_winrate` | `wins + losses > 40 AND wins > 70%` |
| 6 | 64 | `solo_carried` | `kills > 75% of team kills` |
| 7 | 128 | `too_little_damage` | non-UTILITY player damage share < 5% |
| 8 | 256 | `low_minions_killed` | non-UTILITY CS/min < 4 |
| 15 | 32768 | `rare_build_label` | `(championid, teamposition, build_label)` has < 8 games in stage-2 pool |

### Team-level

| Bit | Weight | Column | Rule |
|-----|--------|--------|------|
| 5 | 32 | `team_kills_to_deaths` | `team_kills * 3 < team_deaths` |
| 9 | 512 | `team_non_utility_avg_cs_per_min_gt_2_5_below_enemy` | team non-UTILITY CS/min trails enemy by > 2.5 |
| 10 | 1024 | `team_non_utility_damage_to_champions_ratio_lt_1_3_vs_enemy` | team non-UTILITY damage < 1/3 of enemy |

### Game-level

| Bit | Weight | Column | Rule |
|-----|--------|--------|------|
| 13 | 8192 | `game_time_lte_16_5` | `info.gameduration <= 16 * 60 + 30` |
| 14 | 16384 | `has_rare_role` | `(championid, teamposition)` < 0.4% of champion picks AND player has < 30 picks on it, both measured on stage-1 pool |

## Rebuild procedure

```bash
CH="docker exec -i clickhouse clickhouse-client --multiquery --receive_timeout=3600 --send_timeout=3600"

# 1. filter_stg_* + filter_result schema
$CH < database/clickhouse/schema/4000_filter_schema.sql

# 2. filter_stg_* + filter_result data
$CH < database/clickhouse/schema/4001_filter_build.sql

# 3. valid_game_ids
$CH < database/clickhouse/schema/5002_valid_game_ids_build.sql

# 4. Persistent game_data_filtered.* copies
$CH < database/clickhouse/schema/5003_filtered_tables_build.sql

# 5. Derived analytical tables
$CH < database/clickhouse/schema/5131_tl_participant_per_minute_stats_build.sql
$CH < database/clickhouse/schema/5133_participant_item_value_totals_build.sql
```

To rebuild from scratch (drops everything first), run `5000` before the
sequence above. Run `5001` only if `valid_game_ids` itself was dropped —
`5000` deliberately does not touch it.

Memory note: `5003` currently carries `max_threads=1, max_block_size=8192,
max_insert_block_size=32768` and a 4-way hash split on `participant_challenges`
to stay under the 4.5 GiB container cap. Both workarounds can be removed once
the payload-table changes above land.

## Item-value dictionary refresh

The item-value dictionary (`game_data.item_value_map_dict`, from
`database/clickhouse/support/item_value_map.jsonl`) feeds stage 3 (`4001`)
and the downstream label totals (`5133`). The file is bind-mounted at
`/var/lib/clickhouse/user_files/clickhouse_support/item_value_map.jsonl`, so
host edits are immediately visible.

Both consumers score the visible item slots (`item0`–`item6`) and the optional
`roleBoundItem` field. If `roleBoundItem` is absent, the original item-slot-only
flow is preserved. If a present role-bound item has no dictionary row, it
contributes a zero vector instead of failing the rebuild.

**Build-related rebuild** — after changing the rare-build calculation or the
item-value totals build, rebuild in dependency order. `4001` refreshes the
filter stages and rare-build labels, `5002` refreshes valid matchids, `5003`
copies the filtered source tables, and `5133` rebuilds the derived item-value
totals from the refreshed `game_data_filtered.participant_stats`.

```bash
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/4001_filter_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5002_valid_game_ids_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5003_filtered_tables_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5133_participant_item_value_totals_build.sql
```

Post-run, `participant_item_value_totals` should have exactly one row per
filtered participant:

```sql
SELECT
    count() AS participant_stats_rows,
    (
        SELECT count()
        FROM game_data_filtered.participant_item_value_totals
    ) AS totals_rows,
    participant_stats_rows = totals_rows AS aligned
FROM game_data_filtered.participant_stats;
```

**Value-only edits** (numeric weights changed, keys unchanged):

```bash
docker exec clickhouse clickhouse-client \
  -q "SYSTEM RELOAD DICTIONARY 'game_data.item_value_map_dict'"
# then re-run the rebuild sequence above from step 1.
```

**Structural edits** (keys added / removed / renamed) — every dictionary
consumer must be updated to match the new key set:

- `7000_item_value_map_dictionary_schema.sql` — dictionary column list.
- `5000_create_filtered_db_schema.sql` + `5132_participant_item_value_totals_schema.sql` — totals column list.
- `4001_filter_build.sql` — `dictGet` tuples, `greatest(...)`, `multiIf(...)`.
- `5133_participant_item_value_totals_build.sql` — column list, `dictGet` tuples, `greatest(...)`, `multiIf(...)`.

The `multiIf` tie-break order in `4001` and `5133` must match exactly so the
same participant receives the same label in both tables. The ordering inside
`greatest(...)` is irrelevant to the result.

After code changes, recreate the dictionary and the totals table, then run the
standard rebuild:

```bash
$CH < database/clickhouse/schema/7000_item_value_map_dictionary_schema.sql
docker exec clickhouse clickhouse-client \
  -q "SYSTEM RELOAD DICTIONARY 'game_data.item_value_map_dict'"
$CH < database/clickhouse/schema/5132_participant_item_value_totals_schema.sql
# then the standard rebuild sequence.
```

**Verification** — no `(championid, teamposition, highest_value_label)` bucket
should fall below the rare-build threshold of 8 after convergence:

```sql
WITH bucket_counts AS (
    SELECT championid, teamposition, highest_value_label, count() AS n
    FROM game_data_filtered.participant_item_value_totals
    GROUP BY championid, teamposition, highest_value_label
)
SELECT count() AS total_buckets, countIf(n < 8) AS sub_8, min(n) AS min_n
FROM bucket_counts;
```

`sub_8 = 0` and `min_n = 8` means the 3-pass rare-build filter converged
under the new mapping. See `analytics_builds/8000_build_label_distribution.sql`
for the per-champion breakdown.

## Sample figures (1,858,577 total games, post item-filter removal)

| Filter | Games | Total games | % |
|---|---:|---:|---:|
| `17-any-filter-triggered` | 461,482 | 1,858,577 | 24.83 |
| `stage3-survivors` | 1,397,095 | 1,858,577 | 75.17 |
| `stage1-survivors` | 1,429,130 | 1,858,577 | 76.89 |
| `stage2-survivors` | 1,402,278 | 1,858,577 | 75.45 |
| `01-kda-lt-1/6` | 160,687 | 1,858,577 | 8.65 |
| `02-spent-lt-50%-earned` | 27,587 | 1,858,577 | 1.48 |
| `03-kills+assists-is-0-and-deaths-gt-4` | 49,281 | 1,858,577 | 2.65 |
| `04-either-summoner-not-cast` | 106,069 | 1,858,577 | 5.71 |
| `05-player-games-gr-40-winrate-gt-70%` | 30,786 | 1,858,577 | 1.66 |
| `06-team-kd-ratio-lt-0.33-vs-enemy` | 164,607 | 1,858,577 | 8.86 |
| `07-player-kills-gt-75%-team-kills` | 24,590 | 1,858,577 | 1.32 |
| `08-non-utility-dmg-share-lt-5%` | 40,908 | 1,858,577 | 2.20 |
| `09-non-utility-cs-per-min-lt-4` | 88,053 | 1,858,577 | 4.74 |
| `10-team-non-utility-avg-cs-per-min-gt-2.5-below-enemy` | 18,287 | 1,858,577 | 0.98 |
| `11-team-non-utility-dmg-to-champs-ratio-lt-1/3-vs-enemy` | 4,335 | 1,858,577 | 0.23 |
| `14-game-time-lte-16.5` | 190,145 | 1,858,577 | 10.23 |
| `15-rare-role-champion-position-lt-0.4-pct-lt-30-games` | 26,852 | 1,858,577 | 1.44 |
| `16-rare-build-label-lt-8-games` | 5,183 | 1,858,577 | 0.28 |
| `17-off-role-low-experience` | 0 | 1,858,577 | 0.00 |

## Future considerations

- **Default runes filter**: flag players using untouched default rune pages.
  Data source: `game_data.participant_perk_ids` (+ optionally
  `participant_perk_values`); reference data: Riot default runes versioned by
  patch. Roll out report-only first, then fold into `any_filter_triggered`.
