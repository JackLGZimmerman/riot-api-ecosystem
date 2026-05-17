# ML Win Prediction

Maintenance: keep this README as the live operating contract. Put dataset/cache mechanics in [DATASET.md](DATASET.md), experiment evidence in [OPTIMISATIONS.md](OPTIMISATIONS.md), repeatable sweep procedure in [TESTING.md](TESTING.md), and metric-field detail in [DIAGNOSTICS.md](DIAGNOSTICS.md).

Predicts `blue_win` from the 10 fixed player slots. `HybridTokenModel` consumes champion, role, build, and side embeddings per player.

championId, teamPosition, and build define what was picked. Historical profile features describe how that pick usually performs based on past games. So the first group is the lookup identity, while the second group is the expected behaviour attached to that identity.

## Flow

See [DATASET.md](DATASET.md) for the exact ClickHouse rebuild order and cache layout.

1. Build `game_data_filtered.ml_game_split` with `5900`.
2. Build `game_data_filtered.ml_game_player_pivot` with `6900`.
3. Cache: `CLICKHOUSE_HOST=localhost python -m app.ml.build_dataset`.
4. Train: `CLICKHOUSE_HOST=localhost python -m app.ml.train`.
5. Curves: `uv run tensorboard --logdir app/ml/data/tensorboard`.

## Model

Current default `ModelConfig`:

| Parameter | Value |
| --- | ---: |
| `d_model` | 256 |
| `n_heads` | 4 |
| `n_layers` | 3 |
| `dim_feedforward` | 1024 |
| `dropout` | 0.15 |
| `attention_dropout` | 0.10 |
| `head_dropout` | 0.0 |
| `pooling` | `team_mean_symmetric` |
| `head_hidden` | 256 |

This is about `2.62M` parameters with the current cache vocabulary. Keep architecture rationale and sweep evidence in `OPTIMISATIONS.md`.

## Target Smoothing

`blue_win.npy` remains the hard outcome: red win `0`, blue win `1`. Training smooths BCE targets only:

```text
smoothed_target = blue_win * (target_max - target_min) + target_min
```

Defaults are `0 -> 0.05` and `1 -> 0.95`. Validation/test metrics always use hard labels, so `val_loss`, `test_loss`, AUC, Brier, ECE, positive rates, and bucket diagnostics remain comparable to real outcomes.

## Throughput

Current 5070 Ti warm-path training uses `batch_size=16384`, BF16 AMP, fused AdamW, `torch.compile(mode="reduce-overhead")`, and `grad_clip=0.0`. Recent `train_step` rows sit around `190k` samples/s after compile/warmup. `16384` reserves ~8 GB, leaving headroom for larger configs in sweeps.

## Training Defaults

| Parameter | Value |
| --- | --- |
| `batch_size` | 16384 |
| `optimizer` | `adamw` |
| `lr` | 5e-5 |
| `weight_decay` | 5e-3 |
| `adamw_betas` | `(0.9, 0.999)` |
| `compile_mode` | `reduce-overhead` |
| `warmup_steps` | 125 |
| `grad_clip` | 0.0 |
| `log_interval` | 40 |
| `epochs` | 80 |
| `target_min` / `target_max` | `0.05` / `0.95` |
| `attention_diagnostics_interval` | 40 epochs |
| `attention_diagnostics_batch_size` | 256 |
| `attention_diagnostics_eval_samples` | 1024 |
| `train_monitor_samples` | 50000 |
| `tensorboard_dir` | `tensorboard` |
| `tensorboard_raw_mirror` | false |
| `use_amp` | true |
| `amp_dtype` | `bfloat16` |

Live training writes `best.pt`, `metrics.jsonl`, and `metrics_latest.json` to `app/ml/data/`. Preserved sweep runs live under `app/ml/data/checkpoints/`.

## Central Prediction Bands

The headline central band is `0.475-0.525`, emitted every epoch for train-monitor, validation, TensorBoard, and generalization-gap tracking. Full prediction diagnostics also include `0.45-0.55` and `0.40-0.60` so the summary shows the tight requirement first, then progressively wider decision regions.
