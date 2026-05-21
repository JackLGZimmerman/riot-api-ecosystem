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
```

## Cache Layout

| Array | Shape | Cache dtype | Training dtype | Status |
| --- | --- | --- | --- | --- |
| `player_champion_build_idx.npy` | `(games, 10)` | `uint16` | decoded to `int32` | model input |
| player role ids | implied by slot | not stored | `int32` | model input |
| `blue_win.npy` | `(games,)` | `uint8` | `float32` | label |

Champion/build ids are packed as `champion_idx * n_builds + build_idx`. Role ids are implied by fixed slot order: blue `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`, then red `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`.

`build_dataset` also writes `profile_standardization` to `cache_meta.json`: per-feature mean/std over present (non-zero-filled) train profile rows, used to standardize the 22 profile fields before the model's profile encoder. A cache format bump (`npy-memmap-v9`) forces a rebuild for caches built before this metadata existed.
