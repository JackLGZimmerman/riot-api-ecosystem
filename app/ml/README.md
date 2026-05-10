# ML Win Prediction

Predicts `blue_win` from 10 player tokens plus 175 interaction tokens. The active feature scope is deliberately small: `1vX`, `1v1`, `2vX`, `2v1`, and `3vX` only. `2v2`, `3v1`, `3v2`, `3v3`, and `4vX` aggregates may exist for support analysis, but they are not model inputs.

## Flow

1. Build `game_data_filtered.ml_game_split` with the `5900` SQL.
2. Build `game_data_filtered.ml_game_player_pivot` with the `6900` SQL.
3. Build the model aggregate tables: `6000_1v1`, `6002_1vx`, `6002_2vx`, `6002_3vx`, and `6003_2v1`.
4. Build `game_data_filtered.ml_interaction_counts` with the `6901` SQL.
5. Cache with `CLICKHOUSE_HOST=localhost python -m app.ml.build_dataset`.
6. Train with `CLICKHOUSE_HOST=localhost python -m app.ml.train`.

`6901_ml_interaction_counts_build.sql` materialises sparse `(matchid, token_idx, matchups, primary_wins)` rows. It joins only train-split aggregate rows with `matchups >= 5`. The Python cache builder applies the same minimum after train leave-one-out, then smooths and normalizes interaction scores.

## Matchup Support Filter

Minimum support is `matchups >= 5` for every aggregate consumed by the model. The `6xxx` aggregate tables keep lower-support rows for analysis; `6901` filters its right-hand joins so large low-signal rows do not enter the model cache.

Train-split row counts from the current local build:

| Table | no filter | `>= 2` | `>= 3` | `>= 4` | `>= 5` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `matchup_1v1` | 673,871 | 425,058 | 333,994 | 283,142 | 249,245 |
| `matchup_2v1` | 29,046,790 | 10,362,158 | 6,218,148 | 4,379,744 | 3,339,648 |
| `matchup_2v2` | 71,101,078 | 6,116,807 | 1,703,305 | 694,851 | 347,092 |
| `matchup_3v1` | 71,088,318 | 6,143,029 | 1,708,103 | 697,997 | 347,977 |
| `matchup_3v2` | 160,337,204 | 831,509 | 48,165 | 7,732 | 2,052 |
| `matchup_3v3` | 80,600,752 | 13,177 | 67 | 4 | 0 |
| `synergy_1vx` | 3,166 | 3,149 | 3,114 | 3,046 | 2,959 |
| `synergy_2vx` | 534,938 | 336,887 | 264,444 | 224,043 | 197,593 |
| `synergy_3vx` | 5,762,314 | 2,071,656 | 1,246,433 | 880,213 | 672,039 |
| `synergy_4vx` | 7,105,186 | 622,665 | 172,588 | 69,571 | 34,340 |

Re-run `database/clickhouse/schema/analytics_builds/8005_matchup_threshold_counts.sql` after rebuilding aggregate tables.

## Rebuild Order

Use this command shape for each file:

```bash
docker exec clickhouse clickhouse-client --multiquery \
  --queries-file /docker-entrypoint-initdb.d/<file>.sql
```

Required model path:

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

Optional support-analysis aggregates:

```text
6001_2v2_aggregations_schema.sql
6001_2v2_aggregations_build.sql
6004_3v1_aggregations_schema.sql
6004_3v1_aggregations_build.sql
6005_3v2_aggregations_schema.sql
6005_3v2_aggregations_build.sql
6001_3v3_aggregations_schema.sql
6001_3v3_aggregations_build.sql
6002_4vx_aggregations_schema.sql
6002_4vx_aggregations_build.sql
```

## Model Inputs

| Array | Shape | Meaning |
| --- | --- | --- |
| `champion_idx.npy` | `(games, 10)` | champion embedding ids |
| `role_idx.npy` | `(games, 10)` | role embedding ids |
| `build_idx.npy` | `(games, 10)` | item-value build label ids |
| `interaction_score.npy` | `(games, 175)` | normalized centered win-rate prior |
| `interaction_reliability.npy` | `(games, 175)` | support saturation in `[0, 1]` |
| `blue_win.npy` | `(games,)` | target label |

Player slots are blue `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`, then red `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`.

## Token Layout

| Token idx | Count | Side | Source |
| --- | ---: | --- | --- |
| `0..4` | 5 | blue | `synergy_1vx` |
| `5..9` | 5 | red | `synergy_1vx` |
| `10..34` | 25 | cross | `matchup_1v1` |
| `35..44` | 10 | blue | `synergy_2vx` |
| `45..54` | 10 | red | `synergy_2vx` |
| `55..104` | 50 | blue | `matchup_2v1` blue pair vs red single |
| `105..154` | 50 | red | `matchup_2v1` red pair vs blue single |
| `155..164` | 10 | blue | `synergy_3vx` |
| `165..174` | 10 | red | `synergy_3vx` |

`HybridTokenModel` embeds 10 player tokens, 175 interaction tokens, and one learnable `[CLS]` token. Training writes checkpoints and metrics under `app/ml/data/checkpoints/`; caching writes arrays and metadata under `app/ml/data/cache/`.
