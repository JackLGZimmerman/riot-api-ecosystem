# ML Diagnostics

Training runs for the full configured `TrainConfig.epochs`; there is no early stopping gate. Live rows append to `app/ml/data/metrics.jsonl` and mirror to `app/ml/data/metrics_latest.json`.

## Cadence

- Every `train_step`: `train_loss`, `batch_loss`, `lr`, `grad_norm`, and `samples_per_s` at `TrainConfig.log_interval`.
- Every epoch: core train-monitor and validation loss/accuracy/AUC/Brier/ECE, `0.475-0.525` central-band metrics, and `gen_*` gaps.
- Every `TrainConfig.attention_diagnostics_interval` epochs: sampled attention fields plus full prediction bucket/distribution tables.
- Final test: full metrics from the best validation-loss checkpoint.

## Train Monitor

`TrainConfig.train_monitor_samples` evaluates a fixed held-in train slice through the validation path every epoch. These metrics are directly comparable with validation because both use hard labels.

`gen_*` fields are train-minus-held-out gaps with signs normalized so positive means the train split is ahead, which is the overfitting direction. Headline fields include `gen_loss_gap`, `gen_accuracy_gap`, `gen_auc_gap`, `gen_brier_gap`, `gen_ece_gap`, `gen_central_475_525_logloss_gap`, and `gen_central_475_525_auc_gap`.

## TensorBoard

TensorBoard writes to `app/ml/data/tensorboard/<metrics-file-stem>_<YYYYMMDD_HHMMSS>/`:

```bash
uv run tensorboard --logdir app/ml/data/tensorboard
```

Curated families are written by default: `loss/*`, `quality/*`, `central_475_525/*`, `generalization/*`, `calibration/*`, `optimization/*`, `throughput/*`, `predictions/*`, `attention/*`, and `best/*`. Set `TrainConfig.tensorboard_raw_mirror=True` to additionally mirror every JSONL scalar under `raw/<event>/<field>`.

## Session Summary

`app/ml/utils/session_summary.py` compresses a completed or in-progress `metrics.jsonl` into compact JSON intended for LLM review:

```bash
uv run python -m app.ml.utils.session_summary app/ml/data/metrics.jsonl \
  --out app/ml/data/session_summary.json
```

The summary includes run/config/data context, model evaluation snapshots (`validation_last`, validation bests, train-monitor, checkpoint, and final test when present), trend stats over the whole session and recent window, a downsampled epoch timeline, prediction diagnostics, attention diagnostics, and derived signals such as overfit gaps or calibration/prediction-bias warnings. Prediction buckets and central-band outputs are folded into split-level rollups; attention fields are reduced to the decision-relevant layer/head/token-utilization metrics plus top movers between heavy-diagnostic epochs.

Use the compact default output for LLM input. Add `--pretty` only for human inspection.

## Field Guide

Prediction buckets use probability edges `0.35`, `0.40`, `0.45`, `0.50`, `0.55`, `0.60`, `0.65`. Central bands cover `0.475-0.525`, `0.45-0.55`, and `0.40-0.60`.

| Metric family | Purpose |
| --- | --- |
| `gen_*` | Train-vs-held-out gaps; positive means overfitting direction. |
| `train_monitor_*`, `train_central_475_525_*` | Held-in train metrics comparable to validation. |
| `*_central_475_525_*` | Critical decision-band logloss/AUC/Brier/calibration/accuracy. |
| `grad_norm` | Optimization stability; spikes flag instability, collapse flags weak signal. |
| `attention_cls_mass`, `attention_player_mass` | Attention mass on `[CLS]` vs player tokens. |
| `attention_entropy_mean`, `attention_effective_tokens_mean` | Flat vs sharp attention. |
| `attention_head_similarity_mean`, `attention_head_diversity_mean` | Redundant vs specialized heads. |
| `attention_token_utilization`, `attention_ignored_token_frac` | Whether tokens are being ignored. |
| `attention_first_layer_*`, `attention_last_layer_*`, `attention_layer_*_range` | Layer-to-layer attention behavior. |
| `attention_drift_l2`, `attention_drift_cosine`, `*_temporal_std` | Attention instability over time or samples. |
| `pred_std`, `pred_p01`...`pred_p99` | Prediction spread. |
| `pred_bucket_*` | Count/share/mean prediction/actual rate/gap/accuracy/logloss per bucket. |
| `pred_central_*` | Central-band AUC/logloss/Brier/calibration/accuracy. |

## Reading The Run

Use validation loss and Brier for probability quality, AUC for ranking quality, and ECE/calibration fields for probability trust. Use central-band metrics when the main question is "does this help close calls?" but do not promote a change that improves central-band AUC while worsening overall validation loss, Brier, or calibration.
