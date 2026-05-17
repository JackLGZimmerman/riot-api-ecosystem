# ML Diagnostics

Training runs for the full configured `TrainConfig.epochs`; there is no early stopping gate. Live rows append to `app/ml/data/metrics.jsonl` and mirror to `app/ml/data/metrics_latest.json`.

## Cadence

- Every `train_step`: `train_loss`, `batch_loss`, `lr`, `grad_norm`, `samples_per_s` at `TrainConfig.log_interval`.
- Every epoch (`epoch_end`): `train_loss`, `train_monitor_loss`, `val_loss`, `val_accuracy`, `val_auc`, `val_brier`, `val_ece`, `val_central_475_525_{auc,logloss,calibration}`, `gen_{loss,accuracy,auc}_gap`, `gen_central_475_525_auc_gap`, `lr`, `epoch_s`.
- Every `TrainConfig.attention_diagnostics_interval` epochs: adds the curated train/val attention scalars (entropy, head diversity, ignored-token fraction, player mass; train also `attention_drift_cosine` epoch-over-epoch). The graduated band table is emitted as a dedicated `prediction_bands` event.
- Final test: full metrics from the best validation-loss checkpoint plus a final `prediction_bands` table.

## Train Monitor

`TrainConfig.train_monitor_samples` evaluates a fixed held-in train slice through the validation path every epoch. Only `train_monitor_loss` is logged directly; the other train-vs-val deltas are surfaced via `gen_*` fields. Signs on `gen_*` are normalized so positive means the train split is ahead - the overfitting direction.

## TensorBoard

TensorBoard writes to `app/ml/data/tensorboard/<metrics-file-stem>_<YYYYMMDD_HHMMSS>/`:

```bash
uv run tensorboard --logdir app/ml/data/tensorboard
```

Curated families: `loss/*`, `quality/*`, `central_475_525/*`, `generalization/*`, `calibration/*`, `optimization/*`, `throughput/*`, `attention/*`, `best/*`. Set `TrainConfig.tensorboard_raw_mirror=True` to additionally mirror every JSONL scalar under `raw/<event>/<field>`.

## Field Guide

The graduated prediction-band table (5% / 2.5% / 0.5% slices across 0-100%) is emitted on heavy epochs and final test as its own `prediction_bands` event. The 0.475-0.525 central band is the decision-critical region and is logged every epoch.

| Metric family | Purpose |
| --- | --- |
| `gen_*` | Train-vs-held-out gaps; positive means overfitting direction. |
| `val_central_475_525_*` | Decision-band AUC, logloss, calibration. |
| `grad_norm` | Optimization stability; spikes flag instability, collapse flags weak signal. |
| `attention_entropy_mean` | Attention collapse detector (lower = sharper). |
| `attention_head_diversity_mean` | Head-degeneration detector (lower = redundant heads). |
| `attention_ignored_token_frac` | Tokens receiving near-zero attention mass. |
| `attention_player_mass` | Attention mass on player tokens vs `[CLS]`. |
| `attention_drift_cosine` | Train-side attention reconfiguration between heavy epochs. |

## Reading The Run

Use validation loss and Brier for probability quality, AUC for ranking quality, and ECE / `val_central_475_525_calibration` for probability trust. Use the central-band fields when the main question is "does this help close calls?" but do not promote a change that improves central-band AUC while worsening overall validation loss, Brier, or calibration.
