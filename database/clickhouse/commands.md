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
        └─► game_data_filtered.participant_stats     (SEMI JOIN copy for ML feature builds)
```

## Files

| File | Role |
| --- | --- |
| `3139_participant_stats_corrected_schema.sql` | DROP + CREATE for `game_data.participant_stats_corrected` |
| `3139_participant_stats_corrected_build.sql` | Populate `participant_stats_corrected` (run before filter) |
| `4000_filter_schema.sql` | DROP + CREATE for all `filter_stg_*` + `filter_result` tables |
| `4000_filter_build.sql` | Populate all three stages, rollups, and `filter_result` |
| `5000_create_filtered_db_schema.sql` | `game_data_filtered` database + cleanup of retired copies + DROP/CREATE for ML persistent tables |
| `5001_valid_game_ids_schema.sql` | `game_data_filtered.valid_game_ids` (not dropped by 5000) |
| `5001_valid_game_ids_build.sql` | Populate `valid_game_ids` from `filter_stg_game_flags` |
| `5003_filtered_tables_build.sql` | Populate `participant_stats` via `SEMI JOIN valid_game_ids` |
| `5003_participant_stats_only_build.sql` | Fast iteration copy for `participant_stats` |
| `5132_participant_item_value_totals_schema.sql` | DROP + CREATE for `game_data_filtered.participant_item_value_totals` |
| `5132_participant_item_value_totals_build.sql` | Populate build labels in `participant_item_value_totals` |
| `5900_ml_game_split_schema.sql` | Persistent per-patch chronological train/test label table |
| `5900_ml_game_split_build.sql` | Populate the per-patch chronological 80/20 train/test labels used by 6000+ aggregate builds |
| `6000`/`6003`/`6004` | Active build-conditioned 1v1, 1vx, and 2vx aggregate tables |
| `6020`-`6024` | Active no-build, champion, and build-group backoff aggregate tables |
| `6900_ml_game_player_pivot_schema.sql` | Persistent per-game player tuple table for aggregate and cache builds |
| `6900_ml_game_player_pivot_build.sql` | Populate `ml_game_player_pivot` from filtered participant rows |
| `7004`-`7006`, `7010`-`7014` | Active ML prior dictionary schemas and reload scripts |
| `analytics_builds/8xxx_*.sql` | Human-facing inspection / reporting queries |

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
  --queries-file /docker-entrypoint-initdb.d/5900_ml_game_split_schema.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/5900_ml_game_split_build.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6900_ml_game_player_pivot_schema.sql

docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6900_ml_game_player_pivot_build.sql

# Then rebuild the active ML priors:
#   6003, 6000, 6004, 6020, 6021, 6024, 6022, 6023
# and reload dictionaries:
#   7004, 7005, 7006, 7010, 7011, 7014, 7012, 7013
```

To rebuild from scratch (drops `game_data_filtered.*` first), run `5000`
before the sequence above. Run `5001` only if `valid_game_ids` itself was
dropped — `5000` deliberately does not touch it.

After the SQL path finishes, rebuild the Python ML cache:

```bash
uv run python -m app.ml.build_dataset
```

Memory note: `5003` now copies only `participant_stats`. Filtered timeline
snapshots and non-ML matchdata copies were retired to keep
`game_data_filtered` focused on active feature paths.

## Fast Filter Iteration

Use when validating filter changes and checks need `filter_stg_*`,
`valid_game_ids` and `game_data_filtered.participant_stats`. Timeline tables are
no longer mirrored into `game_data_filtered`; use raw `game_data` tables
directly for one-off profiling.

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

`SOURCE(CLICKHOUSE(...))` dictionaries that pull from this server need
credentials because the `default` user is not available in this deployment.
Credentials live in the `ch_internal` named collection so they stay out of the
schema files.

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

After rebuilding the build-conditioned aggregations in `6003`/`6000`/`6004`,
reload each matching prior dictionary so `build_dataset.py` sees fresh values:

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

# 2vx backoff: build-group sibling (6024), no-build pair (6022), and
# champion pair (6023, kept for compatibility with existing cache metadata)
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6024_2vx_build_group_aggregations_build.sql
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6022_2vx_nobuild_aggregations_build.sql
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/6023_2vx_champ_aggregations_build.sql

# Reload the backoff dictionaries
for d in 7010_matchup_1v1_nobuild 7011_matchup_1v1_champ \
         7014_synergy_2vx_build_group 7012_synergy_2vx_nobuild \
         7013_synergy_2vx_champ; do
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
