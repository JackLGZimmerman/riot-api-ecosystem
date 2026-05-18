# ML Win Prediction

Maintenance: keep this README as the live operating contract. Put dataset/cache mechanics in [DATASET.md](DATASET.md), experiment evidence in [OPTIMISATIONS.md](OPTIMISATIONS.md), repeatable sweep procedure in [TESTING.md](TESTING.md), and metric-field detail in [DIAGNOSTICS.md](DIAGNOSTICS.md).

Predicts `blue_win` from the 10 fixed player slots. `HybridTokenModel` consumes champion, role, build, and side embeddings per player.

championId, teamPosition, and build define what was picked. Historical profile features describe how that pick usually performs based on past games. So the first group is the lookup identity, while the second group is the expected behaviour attached to that identity.

## Token Layout

Sequence length is `10`, fixed per game by the `6900` pivot:

- Tokens `0-4` = blue side (`teamid = 100`), ordered TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY.
- Tokens `5-9` = red side (`teamid = 200`), same role order.

`blue_win` is `anyIf(win, teamid = 100)`, so the label is always relative to the first 5 tokens. The implied role index in `dataset.py` is `[0,1,2,3,4, 0,1,2,3,4]`; side is inferred from token position, not stored.

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
| `pooling` | `team_mean` |
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
| `lr` / base LR | 2e-4 |
| `weight_decay` | 5e-3 |
| `adamw_betas` | `(0.9, 0.999)` |
| `compile_mode` | `reduce-overhead` |
| `warmup_steps` | 125 |
| `lr_center_epoch` | 10 |
| `lr_sharpness` | 8.0 |
| `lr_tail_strength` | 0.5 |
| `lr_eta_min_ratio` | 0.01 |
| `grad_clip` | 0.0 |
| `log_interval` | 10 |
| `epochs` | 300 |
| `target_min` / `target_max` | `0.05` / `0.95` |
| `attention_diagnostics_interval` | 10 epochs |
| `attention_diagnostics_batch_size` | 256 |
| `attention_diagnostics_eval_samples` | 1024 |
| `train_monitor_samples` | 50000 |
| `tensorboard_dir` | `tensorboard` |
| `tensorboard_raw_mirror` | false |
| `use_amp` | true |
| `amp_dtype` | `bfloat16` |

Live training writes `best.pt`, `metrics.jsonl`, and `metrics_latest.json` to `app/ml/data/`. Preserved sweep runs live under `app/ml/data/checkpoints/`.

## Learning-Rate Schedule

Training uses a single `LambdaLR` stepped once after each optimizer step: a linear warmup ramps to the base LR, then a smooth heavy-tail decay holds the LR near peak early, falls off smoothly around `lr_center_epoch`, and decays slowly through a long tail to `lr_eta_min_ratio * base_lr`. The floor is reached exactly at the final step; the curve is continuous and smooth after warmup.

```python
import torch

base_lr = 2e-4
warmup_steps = 125
num_epochs = 300
batches_per_epoch = len(train_loader)
total_steps = max(1, batches_per_epoch * num_epochs)
decay_steps = max(1, total_steps - warmup_steps)
center_step = batches_per_epoch * 10  # smooth fall-off around epoch 10
center_progress = center_step / decay_steps
sharpness = 8.0
tail_strength = 0.5
eta_min_ratio = 0.01

raw_end = (1.0 + (1.0 / center_progress) ** sharpness) ** (-tail_strength)

def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        start = 1.0 / warmup_steps
        return start + (1.0 - start) * (step / warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    raw = (1.0 + (progress / center_progress) ** sharpness) ** (-tail_strength)
    remaining = (raw - raw_end) / (1.0 - raw_end)
    return eta_min_ratio + (1.0 - eta_min_ratio) * remaining

optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = model_loss(model, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

    validate(model, val_loader)
```

Tuning parameters:

- `lr` is the base (peak) learning rate after warmup.
- `warmup_steps` controls how many optimizer steps ramp up to the base LR.
- `lr_center_epoch` is the epoch around which the main fall-off happens.
- `lr_sharpness` controls how steep the fall-off transition is (higher = sharper).
- `lr_tail_strength` controls how slowly the post-centre tail decays (higher = faster decay).
- `lr_eta_min_ratio` sets the final floor as a fraction of `lr`; the last step hits it exactly.
- `epochs` controls the schedule length together with the number of batches per epoch.

## Central Prediction Bands

The headline central band is `0.475-0.525`, emitted every epoch for train-monitor, validation, TensorBoard, and generalization-gap tracking. Full prediction diagnostics also include `0.45-0.55` and `0.40-0.60` so the summary shows the tight requirement first, then progressively wider decision regions.

## Future Considerations

Removed during the most recent simplification pass; reinstate only if the listed need re-emerges.

- `head_dropout` (`ModelConfig`): per-head attention drop applied inside `_DiagnosticEncoderLayer`. Removed because the default `0.0` was never swept. To reinstate: add the field back, restore `_apply_head_dropout` on `_DiagnosticEncoderLayer`, and gate `_sa_block`'s manual-attention path on `self.training and self.head_dropout > 0.0` so non-diagnostic forward passes still take the SDPA fast path when the knob is off.
- `train_matchids_hash` (`cache_meta.json`): 12-char SHA1 over the ordered train matchids. Removed because no loader consumed it. Reinstate as a cheap integrity check that the cache still matches the current `ml_game_split` rows.
