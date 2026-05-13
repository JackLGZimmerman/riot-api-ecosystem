# ML Win Prediction

Predicts `blue_win` from 10 player tokens plus 10 interaction tokens while
tuning. This is a temporary high-throughput tuning mode that keeps only the
`1vX` single-synergy tokens active. The wider 55-token layout (`1vX`, `1v1`,
and `2vX`) remains the intended feature scope, but `1v1` and `2vX` are disabled
as model inputs during this pass. Larger matchup aggregates (`2v1`, `3vX`,
`4vX`, etc.) may exist for support analysis but are not model inputs.

## Flow

After changing the filter rules, rebuild `game_data_filtered.valid_game_ids`
and the filtered participant/build-label inputs before rebuilding the ML tables.
The current filter evidence leaves `1,678,311` valid games; `ml_game_split` and
the Python cache should be within the 10-participant eligibility delta of that
number, not the stale `1.28M` filtered-info overlap.

The remaining 88 game delta is the strict ML eligibility gate for exactly 10 usable role/build participants.

1. Build `game_data_filtered.ml_game_split` with the `5900` SQL.
2. Build `game_data_filtered.ml_game_player_pivot` with the `6900` SQL.
3. Build the model aggregate tables: `6000_1v1`, `6002_1vx`, `6002_2vx`, `6002_3vx`, and `6003_2v1`.
4. Build `game_data_filtered.ml_interaction_counts` with the `6901` SQL.
5. Cache with `CLICKHOUSE_HOST=localhost python -m app.ml.build_dataset`.
6. Train with `CLICKHOUSE_HOST=localhost python -m app.ml.train`.

## GPU Software Stack

The 5070 Ti is a Blackwell GPU (compute capability `12.0`), so the project pins current CUDA-era packages rather than the old GTX 1080-friendly stack:

| Component | Version |
| --- | --- |
| PyTorch | `2.11.0` |
| TensorFlow | `2.21.0` with `tensorflow[and-cuda]` |
| CUDA runtime family | CUDA `13` Python wheels, satisfying the CUDA `12.5+` requirement |
| cuDNN | `nvidia-cudnn-cu13==9.21.0.82` |
| TensorRT | `tensorrt-cu13[numpy]==10.16.1.11` |

`HybridTokenModel` trains through PyTorch, so TensorFlow is available for compatibility/experiments but is not in the training path. TensorRT is included for optimized inference/export work after a PyTorch checkpoint exists; it does not change training loss or model architecture.

`6901_ml_interaction_counts_build.sql` materialises sparse `(matchid, token_idx, matchups, primary_wins)` rows. It joins only train-split aggregate rows with `matchups >= 5`. The Python cache builder applies the same minimum after train leave-one-out, then computes and normalizes interaction scores. Wilson score smoothing is controlled by `DatasetConfig.smooth_interaction_scores` and is off by default.

Train leave-one-out is applied only to train rows because the interaction
aggregates are built from train games. For each train game/token, the cache
builder subtracts that game's own matchup and win from the aggregate before
computing the score, preventing the row's label from leaking into its own
features. Validation and test rows use the train aggregate as-is because they
were never part of it.

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

| Array | Shape | Cache dtype | Training dtype | Meaning |
| --- | --- | --- | --- | --- |
| `player_champion_build_idx.npy` | `(games, 10)` | `uint16` | decoded to `int32` | packed champion/build embedding ids |
| player role ids | implied by slot | not stored | `int32` | role embedding ids |
| `interaction_score.npy` | `(games, 10)` | `float16` | `float32` | normalized centered interaction win-rate score |
| `blue_win.npy` | `(games,)` | `uint8` | `float32` | target label |

The compact cache stores only the precision/range needed on disk, then decodes
to PyTorch-friendly training tensors when loaded. Champion and build ids are
packed as `champion_idx * n_builds + build_idx`; role ids are implied by the
fixed player slot order. With the temporary 10-token tuning layout this keeps
the active array cache near `41 MB` per million games before filesystem
metadata, without int8 quantization.

Player slots are blue `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`, then red `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`.

## Token Sparsity

The cache records how often model inputs fall back to `0`, split by the 10 player tokens and 10 interaction tokens.

For player tokens, logical `champion_idx`, `role_idx`, and `build_idx` reserve
`0` as `UNK_INDEX`. The cache packs champion/build ids and implies roles from
slot order, but `cache_meta.json` still records the source coverage rates under
`player_token_sparsity`:

| Field | Meaning |
| --- | --- |
| `champion_unknown_slots` / `champion_unknown_frac` | player slots where `champion_idx == 0` |
| `role_unknown_slots` / `role_unknown_frac` | player slots where `role_idx == 0` |
| `build_unknown_slots` / `build_unknown_frac` | player slots where `build_idx == 0` |
| `any_unknown_slots` / `any_unknown_frac` | player slots where any of champion, role, or build is unknown |
| `by_token` | the same counts/rates per player slot, with side and role label |

For interaction tokens, every game starts with a dense `(10,)` interaction vector initialized to `0.0`; only `(matchid, token_idx)` rows found in `ml_interaction_counts` are filled. Missing source rows therefore become raw zero scores before train-only normalization. `cache_meta.json` records pre-normalization interaction sparsity under `interaction_sparsity`:

| Field | Meaning |
| --- | --- |
| `source_missing_slots` / `source_missing_frac` | token slots absent from `ml_interaction_counts`; these are the direct "missing became zero" cases |
| `support_filtered_slots` / `support_filtered_frac` | source rows that fail `min_matchup_count` after train leave-one-out, mostly train rows with exactly the minimum source support |
| `zero_score_slots` / `zero_score_frac` | all raw zero scores, including missing slots, support-filtered slots, exact 50% win-rate slots, and Wilson-smoothed zeros |
| `by_token` | the same counts/rates per interaction token idx, with token type, side, and role slots |

Both sections include a top-level `overall` summary plus `splits.train`, `splits.val`, and `splits.test`. Use player-token `any_unknown_frac` to interpret categorical coverage, interaction `source_missing_frac` to interpret lookup coverage, and interaction `nonzero_score_frac` to estimate how much of a proposed token family will carry an actual signal into the model.

## Target Label Smoothing

`blue_win.npy` stays as the observed hard game outcome: `0` for a red win and
`1` for a blue win. The cache should not rewrite those labels. During training
only, `TrainConfig.target_min` and `TrainConfig.target_max` smooth the BCE
target with:

```text
smoothed_target = blue_win * (target_max - target_min) + target_min
```

With the default bounds, hard labels become:

| Hard label | Training target |
| ---: | ---: |
| `0` | `0.15` |
| `1` | `0.85` |

This is a training prior, not a different definition of the outcome. Draft
features can carry useful signal, but they cannot make a game certain before it
is played. Smoothing the training target discourages the model from spending
capacity on extreme logits for draft-only evidence and usually improves
calibration by making the maximum/minimum rewarded probabilities more realistic.

Validation and test metrics still use the original hard `blue_win` labels.
`AUC` measures how well predictions rank actual wins above actual losses;
`accuracy` measures thresholded correctness against real outcomes; `Brier` and
`ECE` measure probability calibration against observed events; and displayed
win rates/positive rates should remain the empirical game result rate. Scoring
those against smoothed labels would evaluate the regularizer instead of the
thing the model is meant to predict, and would hide real overconfidence or
underconfidence.

## Token Layout

10 interaction tokens are active for temporary tuning. They are the blue and
red `1vX` single-synergy tokens only. The full 55-token layout is still the
target feature set, but the 25 `1v1` matchup tokens and 20 `2vX` pair-synergy
tokens are currently disabled to speed optimizer and batch-size tests.

| Token idx | Count | Side | Source |
| --- | ---: | --- | --- |
| `0..4` | 5 | blue | `synergy_1vx` |
| `5..9` | 5 | red | `synergy_1vx` |

The 10 player tokens identify the champion, role, and build label for each slot:
blue `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`, then red `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`.
Each interaction token adds `interaction_score`: a centered win-rate signal,
then train-normalized. If `DatasetConfig.smooth_interaction_scores` is enabled,
Wilson score smoothing first shrinks uncertain signals toward `0`.

Example: token `0` is blue top `1vX` synergy. If a blue top champion/build has
`70` train matchups and the blue side won `39`, the score before normalization
is `39 / 70 - 0.5 = 0.057`. At `50 / 70` wins, the score before normalization is
`0.214`.

`HybridTokenModel` embeds 10 player tokens, 10 interaction tokens, and one learnable `[CLS]` token. Training writes checkpoints and metrics under `app/ml/data/checkpoints/`; caching writes arrays and metadata under `app/ml/data/cache/`.

## Throughput

The 5070 Ti throughput probe uses short real optimizer-step sessions and chooses
the largest batch that stays on the samples/s plateau before allocator pressure
dominates. The current default physical batch is `10240`, which measured about
`42.9k` samples/s over 10 timed steps and peaked at about `12.4 GiB` allocated
(`14.1 GiB` reserved). `11264` fell to about `18.1k` samples/s, so the default
backs off to the faster point rather than the memory cliff.

## Attention Diagnostics

Training samples concise attention diagnostics every `TrainConfig.attention_diagnostics_interval` epochs on one active training batch while limiting retained summaries to `attention_diagnostics_batch_size` examples (default `256`). Validation evaluation collects `attention_diagnostics_eval_samples` examples (default `1024`) on the same epoch cadence, and final test evaluation still collects diagnostics once. Live samples are appended to `metrics.jsonl` on `train_step` / `attention_step` events and mirrored to `metrics_latest.json`.

Diagnostic epoch summaries add `train_attention_*`, `val_attention_*`, and
`val_pred_*` fields to `epoch_end` only on the configured diagnostic cadence;
non-diagnostic epochs keep the core loss/accuracy/AUC/Brier/ECE fields. Final
evaluation adds `test_attention_*` and `test_pred_*` fields to the `test` event.
When `torch.utils.tensorboard` is available, scalar metrics are also written
under `app/ml/data/checkpoints/tensorboard/<metrics-file-stem>/`.

Every validation and final test evaluation logs a prediction bucket table using
the `0.35`, `0.40`, `0.45`, `0.50`, `0.55`, `0.60`, and `0.65` probability
edges. It also logs central-band AUC/logloss/Brier for `0.35-0.65`,
`0.40-0.60`, and `0.45-0.55`. The structured diagnostic fields use
`pred_bucket_*` and `pred_central_*` names and follow the 5-epoch diagnostic
cadence for validation JSONL rows.

Logged attention fields include:

| Metric family | Purpose |
| --- | --- |
| `attention_entropy_mean`, `attention_entropy_std`, `attention_effective_tokens_mean` | Detect collapse (very low entropy) or over-smoothing (near-uniform entropy). |
| `attention_max_prob_mean`, `attention_max_prob_p95`, `attention_top5_mass_mean` | Track excessive sharpness or flatness. |
| `attention_head_similarity_mean`, `attention_head_diversity_mean` | Surface redundant heads and weak head specialisation. |
| `attention_token_utilization`, `attention_ignored_token_frac` | Show whether source tokens are being ignored. |
| `attention_first_layer_*`, `attention_last_layer_*`, `attention_layer_*_range` | Compare early and final layer behavior, especially final-layer head redundancy. |
| `attention_drift_l2`, `attention_drift_cosine`, `*_temporal_std` | Monitor instability between diagnostic samples. |
| `pred_std`, `pred_p01`...`pred_p99`, `pred_gt_*`, `pred_lt_*` | Track prediction spread and threshold count/accuracy. |
| `pred_bucket_*` | Track count, data share, mean prediction, actual rate, gap, accuracy, and logloss for p35/40/45/50/55/60/65 buckets. |
| `pred_central_*` | Track central-band count, data share, AUC, logloss, and Brier for 35-65, 40-60, and 45-55. |

## Architecture Experiments

`ModelConfig` exposes the requested architecture sweep without changing data,
labels, features, sampling, or preprocessing:

| Question | Config knobs |
| --- | --- |
| 6 vs 4 vs 3 vs 2 layers | `ModelConfig(n_layers=...)` |
| Fewer heads | `ModelConfig(n_heads=4)` |
| Attention regularisation | `attention_dropout=0.05..0.15`, optional `head_dropout=0.05..0.10` |
| Wider FFN | `dim_feedforward=4 * d_model` or `6 * d_model` |
| Pooling head | `pooling="cls"`, `"mean"`, `"attention"`, `"concat_cls_mean"`, or `"gated"` |

The default candidate is shallow/wide: 4 transformer layers, 4 heads, 6x FFN,
separate 0.10 attention dropout, no head dropout, and gated pooling. This
spends capacity in the FFN and pooling head before adding more attention depth.

## Training Defaults

| Parameter | Value |
| --- | --- |
| `batch_size` | 10240 |
| `gradient_accumulation_steps` | 1 |
| effective batch size | 10240 |
| `optimizer` | `lion` |
| `lr` | 1e-5 |
| `lion_betas` | `(0.9, 0.99)` |
| `warmup_steps` | 125 |
| `log_interval` | 40 |
| `epochs` | 100 (early stopping patience 8) |
| `d_model` | 256 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `dim_feedforward` | 1536 |
| `dropout` | 0.15 |
| `attention_dropout` | 0.10 |
| `head_dropout` | 0.0 |
| `pooling` | `gated` |
| `weight_decay` | 2.5e-2 |
| `target_min` | 0.15 |
| `target_max` | 0.85 |
| `attention_diagnostics_interval` | 5 epochs |
| `attention_diagnostics_batch_size` | 256 |
| `attention_diagnostics_eval_samples` | 1024 |
| `use_amp` | true |
| `amp_dtype` | `bfloat16` |

### VRAM note

The 5070 Ti target now uses a physical batch of `10240` with no gradient
accumulation. BF16 autocast is enabled by default for Blackwell-class hardware.
The `log_interval` and `warmup_steps` defaults are scaled to stay close to the
previous sample cadence after the larger physical batch.

Lion replaces the previous AdamW optimizer via `lion-pytorch==0.2.4`. Its
sign-based update usually wants a smaller learning rate and proportionally
larger decoupled weight decay than AdamW, so the defaults move from `lr=5e-5`,
`weight_decay=5e-3` to `lr=1e-5`, `weight_decay=2.5e-2`.
