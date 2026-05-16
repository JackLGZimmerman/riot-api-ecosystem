# ML Dataset And Cache

Maintenance: update this file when the ML ClickHouse table build path, cache files, or cache semantics change. Keep current model/training defaults in [README.md](README.md), metric-field detail in [DIAGNOSTICS.md](DIAGNOSTICS.md), and experiment evidence in [OPTIMISATIONS.md](OPTIMISATIONS.md).

## Rebuild Order

Run each schema/build file with:

```bash
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/<file>.sql
```

After filter-rule changes, rebuild `game_data_filtered.valid_game_ids` and filtered participant/build-label inputs before ML tables. Current filter evidence leaves about `1,678,311` valid games; the Python cache should be within the strict 10-participant eligibility delta.

Required path:

```text
5900_ml_game_split_schema.sql
5900_ml_game_split_build.sql
6900_ml_game_player_pivot_schema.sql
6900_ml_game_player_pivot_build.sql
6000_1v1_aggregations_schema.sql
6000_1v1_aggregations_build.sql
6002_1vx_aggregations_schema.sql
6002_1vx_aggregations_build.sql
6002_2vx_aggregations_schema.sql
6002_2vx_aggregations_build.sql
6002_3vx_aggregations_schema.sql
6002_3vx_aggregations_build.sql
6003_2v1_aggregations_schema.sql
6003_2v1_aggregations_build.sql
6901_ml_interaction_counts_schema.sql
6901_ml_interaction_counts_build.sql
```

Optional support-analysis aggregates: `6001_2v2`, `6004_3v1`, `6005_3v2`, `6001_3v3`, and `6002_4vx`.

## Cache Layout

| Array | Shape | Cache dtype | Training dtype | Status |
| --- | --- | --- | --- | --- |
| `player_champion_build_idx.npy` | `(games, 10)` | `uint16` | decoded to `int32` | model input |
| player role ids | implied by slot | not stored | `int32` | model input |
| `blue_win.npy` | `(games,)` | `uint8` | `float32` | label |
| `interaction_score.npy` | `(games, 10)` | `float16` | not loaded into model tensors | cached only |

Champion/build ids are packed as `champion_idx * n_builds + build_idx`. Role ids are implied by fixed slot order: blue `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`, then red `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`.

Interaction counts use train leave-one-out for train rows and train-only aggregates for validation/test rows. `matchups >= 5` is the model-support threshold when interaction features are reintroduced. `DatasetConfig.smooth_interaction_scores` controls optional Wilson smoothing in the cache and is off by default.
