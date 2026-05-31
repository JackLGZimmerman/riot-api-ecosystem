# Filter & Filtered-DB Refresh Commands

Refresh procedures for the filter pipeline and the downstream
the ML-facing `game_data_filtered.*` copies. This file is operational only — for rule
semantics, win-rate analysis, threshold history, and bitmask layout, see
[`filter_evidence.md`](filter_evidence.md).

## Pipeline at a glance

```text
game_data.participant_stats
  │
  ├─► 3139 — participant_stats_corrected          (remove end-of-game stat padding)
  │
  ├─► STAGE 1 — filter_stg_participant_flags         (cheap per-participant flags, 1 scan)
  │     └─► filter_stg_stage1_valid_matchids
  │
  ├─► STAGE 2 — filter_stg_participant_labels        (build label + highest_value + low_build_value)
  │
  ├─► filter_stg_game_flags                          (rollup: stage1 + low_build_value)
  ├─► filter_result                                  (bitmask, one row per participant)
  │
  └─► game_data_filtered.valid_game_ids
        ├─► game_data_filtered.participant_stats     (SEMI JOIN copy for ML feature builds)
        └─► game_data_filtered.tl_participant_stats  (classification final-state features)
```

## Files

| File | Role |
| --- | --- |
| `3139_participant_stats_corrected_schema.sql` | DROP + CREATE for `game_data.participant_stats_corrected` |
| `3139_participant_stats_corrected_build.sql` | Populate `participant_stats_corrected` (run before filter) |
| `3140_migrate_raw_tables_to_replacing_merge_tree.sql` | One-shot migration for existing raw `3xxx` tables to `ReplacingMergeTree` natural keys |
| `4000_filter_schema.sql` | DROP + CREATE for all `filter_stg_*` + `filter_result` tables |
| `4000_filter_build.sql` | Populate all three stages, rollups, and `filter_result` |
| `5000_create_filtered_db_schema.sql` | `game_data_filtered` database + cleanup of retired copies + DROP/CREATE for ML persistent tables |
| `5001_valid_game_ids_schema.sql` | `game_data_filtered.valid_game_ids` (not dropped by 5000) |
| `5001_valid_game_ids_build.sql` | Populate `valid_game_ids` from `filter_stg_game_flags` |
| `5003_filtered_tables_build.sql` | Populate `participant_stats` and `tl_participant_stats` via `SEMI JOIN valid_game_ids` |
| `5003_participant_stats_only_build.sql` | Fast iteration copy for the same two 5003 tables |
| `5132` | Build labels in `game_data_filtered.participant_item_value_totals` |
| `5133` | Temporal scaling weights in `game_data_filtered.participant_scaling_weights` |
| `5900_ml_game_split_schema.sql` | Persistent chronological train/validation/test label table |
| `5900_ml_game_split_build.sql` | Populate the 80/10/10 split labels used by 6000+ aggregate builds |
| `analytics_builds/8xxx_*.sql` | Human-facing inspection / reporting queries |

## Raw table dedupe migration

Raw `game_data` matchdata tables use `ReplacingMergeTree` with
`ORDER BY` set to the row's natural identity. `run_id` stays as a normal
column for rollback/delete operations, but it is no longer part of the
dedupe key. Consumers that materialize clean snapshots from raw tables should
read them with `FINAL`; the `3139` and `5003` builds do this explicitly.

For an existing database created with the old `MergeTree`/`run_id` sort keys,
stop the matchdata pipeline first, then run:

```bash
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/3140_migrate_raw_tables_to_replacing_merge_tree.sql
```

The migration creates `__rmt_new` tables, copies the old data, runs
`OPTIMIZE ... FINAL`, atomically exchanges the rebuilt tables into place, and
keeps the previous physical tables as `__pre_rmt` backups. Drop the
`__pre_rmt` tables only after validating counts and rebuilding downstream
filtered/analytics tables.

## Standard rebuild

After a fresh ingest, or after editing any rule in `4000_filter_build.sql`:

```bash
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/3139_participant_stats_corrected_schema.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/3139_participant_stats_corrected_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/4000_filter_schema.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/4000_filter_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5001_valid_game_ids_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5003_filtered_tables_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5132_participant_item_value_totals_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5133_participant_scaling_weights_schema.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5133_participant_scaling_weights_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5900_ml_game_split_schema.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5900_ml_game_split_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6900_ml_game_player_pivot_schema.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6900_ml_game_player_pivot_build.sql

# Then rebuild the active ML priors:
#   6003, 6000, 6004, 6020, 6021, 6022, 6023
# and reload dictionaries:
#   7004, 7005, 7006, 7010, 7011, 7012, 7013
```

To rebuild from scratch (drops `game_data_filtered.*` first), run `5000`
before the sequence above. Run `5001` only if `valid_game_ids` itself was
dropped — `5000` deliberately does not touch it.

After the SQL path finishes, rebuild the Python ML cache:

```bash
uv run python -m app.ml.build_dataset
```

Memory note: `5003` now copies only `participant_stats` and
`tl_participant_stats`. The wider filtered timeline snapshots and non-ML
matchdata copies were retired to keep `game_data_filtered` focused on active
feature paths.

## Fast Filter Iteration

Use when validating filter changes and checks need `filter_stg_*`,
`valid_game_ids`, `game_data_filtered.participant_stats`, and
`game_data_filtered.tl_participant_stats`. Other timeline event tables are no
longer mirrored into `game_data_filtered`; use raw `game_data` tables directly
for one-off profiling.

```bash
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/3139_participant_stats_corrected_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/4000_filter_schema.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/4000_filter_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5001_valid_game_ids_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5003_participant_stats_only_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5132_participant_item_value_totals_build.sql
```

For stage survivor counts after this path:

```bash
docker exec -i clickhouse clickhouse-client \
  < database/clickhouse/schema/analytics_builds/8003_filter_statistics.sql
```

Run the standard rebuild later when ML aggregate tables need to reflect the new
valid-game pool.

## Rule-only re-measurement

When iterating on a threshold and re-running win-rate queries from
[`filter_evidence.md`](filter_evidence.md), stop after the filter stages:

```bash
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/4000_filter_schema.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/4000_filter_build.sql
```

## Named collection for dictionary reloads

`SOURCE(CLICKHOUSE(...))` dictionaries that pull from this server (the ML
priors in `7004`/`7005`/`7006`) need credentials because the `default`
user no longer exists. Credentials live in the `ch_internal` named
collection so they stay out of the schema files.

Create it once per environment (and re-run if the password rotates):

```bash
docker exec clickhouse clickhouse-client -q "
CREATE NAMED COLLECTION IF NOT EXISTS ch_internal AS
    host = 'localhost',
    port = 9000,
    user = '<ml_loader_user>',
    password = '<ml_loader_password>',
    db = 'game_data_filtered'
;"
```

The dictionary `SOURCE` clauses reference it by `NAME 'ch_internal'`; only
the `QUERY` is per-dictionary.

## ML prior dictionary refresh

After rebuilding the aggregations in `6003`/`6000`/`6004`, reload each
matching prior dictionary so `build_dataset.py` sees fresh values:

```bash
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/7004_synergy_1vx_dict_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/7005_matchup_1v1_dict_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/7006_synergy_2vx_dict_build.sql
```

If a dictionary was renamed or its column list changed, re-run the
matching `*_schema.sql` first, then the build above.

### Interaction backoff levels (nested EB pooling)

`build_dataset.py` shrinks each build-conditioned 1v1/2vx prior toward denser
no-build and champion-only parents. Build the backoff aggregations (run each
`*_schema.sql` once, then its `*_build.sql`) and reload their dictionaries:

```bash
# 1v1 backoff: no-build pair (6020) then champion pair (6021)
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6020_1v1_nobuild_aggregations_build.sql
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6021_1v1_champ_aggregations_build.sql

# 2vx backoff: no-build pair (6022) then champion pair (6023)
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6022_2vx_nobuild_aggregations_build.sql
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6023_2vx_champ_aggregations_build.sql

# Reload the four backoff dictionaries
for d in 7010_matchup_1v1_nobuild 7011_matchup_1v1_champ \
         7012_synergy_2vx_nobuild 7013_synergy_2vx_champ; do
  docker exec clickhouse clickhouse-client --multiquery \
    --queries-file /docker-entrypoint-initdb.d/${d}_dict_build.sql
done
```

## Item-value dictionary refresh

The item-value dictionary (`game_data.item_value_map_dict`, from
`database/clickhouse/support/item_value_map.jsonl`) feeds stage 2 (`4000`)
and the downstream label totals (`5132`). The file is bind-mounted at
`/var/lib/clickhouse/user_files/clickhouse_support/item_value_map.jsonl`, so
host edits are immediately visible.

**Value-only edits** (numeric weights changed, keys unchanged):

```bash
docker exec clickhouse clickhouse-client \
  -q "SYSTEM RELOAD DICTIONARY 'game_data.item_value_map_dict'"
# then re-run the standard rebuild from 4000.
```

**Reload both item dictionaries** — after editing support JSONL files directly:

```bash
docker exec clickhouse clickhouse-client --multiquery \
  -q "SYSTEM RELOAD DICTIONARY 'game_data.item_info_dict';
      SYSTEM RELOAD DICTIONARY 'game_data.item_value_map_dict';"
```

**Structural edits** (keys added / removed / renamed) — every dictionary
consumer must be updated to match the new key set:

- `7000_item_value_map_dictionary_schema.sql` — dictionary column list.
- `5000_create_filtered_db_schema.sql` + `5132_participant_item_value_totals_schema.sql` — totals column list.
- `4000_filter_build.sql` — `dictGet` tuples, `greatest(...)`, `multiIf(...)`.
- `5132_participant_item_value_totals_build.sql` — column list, `dictGet` tuples, `greatest(...)`, `multiIf(...)`.

The `multiIf` tie-break order in `4000` and `5132` must match exactly so
the same participant receives the same label in both tables.

After code changes, recreate the dictionary and totals schema, then run
the standard rebuild:

```bash
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/7000_item_value_map_dictionary_schema.sql

docker exec clickhouse clickhouse-client \
  -q "SYSTEM RELOAD DICTIONARY 'game_data.item_value_map_dict'"

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5132_participant_item_value_totals_schema.sql
# then the standard rebuild sequence.
```
