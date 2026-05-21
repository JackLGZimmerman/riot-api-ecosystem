# ML Diagnostics

Training writes the current run to `app/ml/data/metrics.jsonl` and mirrors the latest row to `app/ml/data/metrics_latest.json`.

For cross-run comparisons, use the compact summary utility:

```bash
uv run python -m app.ml.utils.diagnostics_summary app/ml/data/metrics.jsonl --sort best_val_loss --details
```

## Cadence

- Every `train_step` at `TrainConfig.log_interval`: `train_loss`, `batch_loss`, `lr`, `grad_norm`, `samples_per_s`.
- Every epoch (`epoch_end`): `train_loss`, `val_loss`, `val_accuracy`, `val_auc`, `val_brier`, `val_ece`, `lr`, `epoch_s`.
- Final test (`test` + `prediction_bands`): the same five `test_*` scalars from the best validation-loss checkpoint, plus the graduated band table.

## Prediction Bands

The final-test `prediction_bands` event emits one row per bin defined by `_BAND_EDGES` in `app/ml/utils/prediction_diagnostics.py`: 5% slices over 5-25% and 75-90%, 2.5% over 25-40% and 60-75%, 0.5% over 40-60%, and one 90-100% tail bin. Each row carries `band`, `count`, and `accuracy_pct`.

## TensorBoard

TensorBoard writes to `app/ml/data/tensorboard/<metrics-file-stem>_<YYYYMMDD_HHMMSS>/`:

```bash
uv run tensorboard --logdir app/ml/data/tensorboard
```

Curated families: `loss/*`, `quality/*`, `calibration/*`, `optimization/*`, `throughput/*`, `best/*`. Set `TrainConfig.tensorboard_raw_mirror=True` to additionally mirror every JSONL scalar under `raw/<event>/<field>`. Nested dict/list diagnostics are recursively mirrored, so final-test tables are visible under tags such as `raw/prediction_bands/rows/...`.

## Reading The Run

Use validation loss and Brier for probability quality, AUC for ranking quality, and ECE for probability trust. The final-test `prediction_bands` table shows accuracy distribution across confidence bins.

## Local Training Note

In this workspace, plain `python` resolves to `/usr/bin/python` and may not have the ML dependencies. Use `CLICKHOUSE_HOST=localhost .venv/bin/python -m app.ml.train` or `CLICKHOUSE_HOST=localhost uv run python -m app.ml.train`. The default training config also uses `device="cuda"`, so a host without an NVIDIA driver needs a CPU-specific `TrainConfig` override for smoke tests.
