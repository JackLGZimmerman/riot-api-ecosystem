# ML Diagnostics

Training writes the current run to `app/ml/data/metrics.jsonl` and mirrors the latest row to `app/ml/data/metrics_latest.json`.

For cross-run comparisons, use the compact summary utility:

```bash
uv run python -m app.ml.utils.diagnostics_summary app/ml/data/metrics.jsonl app/ml/data/checkpoints/moe_hparam_20260519/e07_dense_final --sort best_val_loss --details
```

## Cadence

- Every `train_step` at `TrainConfig.log_interval`: `train_loss`, `batch_loss`, `lr`, `grad_norm`, `samples_per_s`.
- Every epoch (`epoch_end`): `train_loss`, `val_loss`, `val_accuracy`, `val_auc`, `val_brier`, `val_ece`, `lr`, `epoch_s`.
- Final test (`test` + `prediction_bands` + `matched_moe_diagnostics`): the same five `test_*` scalars from the best validation-loss checkpoint, plus the graduated band table and matched dense-head-vs-MoE diagnostics when `use_moe=true`.

## Prediction Bands

The final-test `prediction_bands` event emits one row per bin defined by `_BAND_EDGES` in `app/ml/utils/prediction_diagnostics.py`: 5% slices over 5-25% and 75-90%, 2.5% over 25-40% and 60-75%, 0.5% over 40-60%, and one 90-100% tail bin. Each row carries `band`, `count`, and `accuracy_pct`.

## Matched MoE Diagnostics

The final-test `matched_moe_diagnostics` event compares each test example's dense baseline logit with the final MoE logit from the same checkpoint. It records the P0 guardrails: folded-confidence baseline -> final transition counts, 40-60% baseline central metrics for baseline/final/delta, and central baseline-band delta rows with logit/probability delta, absolute logit-delta quantiles, target-sign agreement, expert utilization, and average selected expert weights.

Historical full run, 2026-05-18 P1 match-level MoE head:

| central 40-60% metric | dense baseline | final MoE | delta |
| --- | ---: | ---: | ---: |
| count | 133975 | 133975 | 0 |
| BCE | 0.6858 | 0.6853 | -0.0005 |
| Brier | 0.2463 | 0.2461 | -0.0002 |
| ECE | 0.0196 | 0.0172 | -0.0024 |
| AUC | 0.5689 | 0.5694 | +0.0005 |
| hard accuracy | 0.5525 | 0.5531 | +0.0005 |

Route note: the latest central-band rows route almost entirely through experts `E5` and `E7`, so route balance is the main next diagnostic.

## TensorBoard

TensorBoard writes to `app/ml/data/tensorboard/<metrics-file-stem>_<YYYYMMDD_HHMMSS>/`:

```bash
uv run tensorboard --logdir app/ml/data/tensorboard
```

Curated families: `loss/*`, `quality/*`, `calibration/*`, `optimization/*`, `throughput/*`, `best/*`. Set `TrainConfig.tensorboard_raw_mirror=True` to additionally mirror every JSONL scalar under `raw/<event>/<field>`. Nested dict/list diagnostics are recursively mirrored, so final-test tables are visible under tags such as `raw/prediction_bands/rows/...` and `raw/matched_moe_diagnostics/central/delta/...`.

The 2026-05-19 one-by-one optimization runs are already written and replayed under:

```text
app/ml/data/tensorboard/moe_hparam_20260519/
```

## Reading The Run

Use validation loss and Brier for probability quality, AUC for ranking quality, and ECE for probability trust. The final-test `prediction_bands` table shows accuracy distribution across confidence bins.

## Local Training Note

In this workspace, plain `python` resolves to `/usr/bin/python` and may not have the ML dependencies. Use `CLICKHOUSE_HOST=localhost .venv/bin/python -m app.ml.train` or `CLICKHOUSE_HOST=localhost uv run python -m app.ml.train`. The default training config also uses `device="cuda"`, so a host without an NVIDIA driver needs a CPU-specific `TrainConfig` override for smoke tests.
