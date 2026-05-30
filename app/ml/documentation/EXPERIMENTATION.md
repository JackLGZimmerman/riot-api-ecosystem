# Structured Model Experimentation

This is the practical map for tuning and running the ML win-rate model without
having to rediscover the important files.

## Important Files

| File | Why it matters |
| --- | --- |
| `app/ml/config.py` | Main tuning knobs in `TrainConfig`: batch size, epochs, patience, learning rate, weight decay, delta baseline mode, artifact paths, and device policy. |
| `app/ml/structured_model.py` | Model architecture, feature construction, delta formulas, save/load helpers, and shared tensor/device helpers. |
| `app/ml/train.py` | Production training loop: loads count-required splits, trains with BCE-with-logits, early-stops on validation NLL, writes model and metrics. |
| `app/ml/dataset.py` | Cache loading and split slicing. Structured training uses `require_counts=True`. |
| `app/ml/cache_layout.py` | Cache file names, shapes, and cache format version. |
| `app/ml/build_dataset.py` | Builds the extended cache from ClickHouse, including support-count arrays. |
| `app/ml/predictor.py` | Production runtime inference path used by RL callers. |
| `app/ml/data/metrics_latest.json` | Latest production training metrics and epoch history. |
| `app/ml/data/structured_model_latest.json` | Historical structured audit output, if present in the workspace. |

## Production Training

Default production training:

```bash
uv run python -m app.ml.train
```

This writes:

```text
app/ml/data/structured_winrate_model.pt
app/ml/data/metrics_latest.json
```

The command requires an extended cache containing:

```text
p1_cnt.npy
m1v1_cnt.npy
s2vx_cnt.npy
```

If those files are missing, rebuild the cache:

```bash
uv run python -m app.ml.build_dataset
```

## Safe Experiment Runs

Avoid overwriting the production artifact while experimenting. Run training
programmatically with alternate output paths:

```bash
uv run python - <<'PY'
from pathlib import Path

from app.ml.config import DatasetConfig, TrainConfig
from app.ml.train import train

train(
    DatasetConfig(),
    TrainConfig(
        model_path=Path("app/ml/data/experiments/structured_trial.pt"),
        metrics_path=Path("app/ml/data/experiments/structured_trial_metrics.json"),
        batch_size=16384,
        max_epochs=8,
        patience=2,
        learning_rate=5e-4,
        weight_decay=1e-3,
        delta_baseline_mode="logit",
        device="auto",
    ),
)
PY
```

For a quick mechanical check, lower `max_epochs` and write to a scratch path.
Do not use a shortened run as evidence of model quality.

## Tuning Knobs

Start with the knobs in `TrainConfig`:

- `learning_rate`: first place to try if validation NLL spikes after epoch 1.
- `weight_decay`: raise this to fight the interaction-branch overfit seen in
  the latest full-cache run.
- `patience`: keep low during broad sweeps; use higher values only after a
  candidate is stable.
- `batch_size`: mostly throughput; keep large enough for GPU efficiency.
- `delta_baseline_mode`: compare `"logit"` and `"probability"` only when the
  rest of the run is otherwise controlled.
- `device`: `"auto"` for normal runs, `"cpu"` for deterministic smoke checks.

Architecture changes live in `StructuredModelConfig` in `structured_model.py`.
Treat changes there as model changes and record them in the metrics filename or
experiment notes.

Leakage-robustness knobs (production defaults in parentheses); see
`documentation/README.md` for the rationale:

- `interaction_loo` (`True`, in `DatasetConfig`): leave-one-out encodes train
  priors so the `expected`/`delta` columns no longer leak the label. Requires a
  cache rebuild.
- `object_feature_mode` (`"full"`): keeps the 6-feature objects (incl.
  `expected`/`delta`), which carry the matchup/synergy signal once
  `interaction_loo` makes them honest. `"raw"` drops them (the pre-LOO workaround).
- `confidence_gate` (`True`): scale interaction embeddings by support confidence.
- `pooling_ops` (`("weighted",)`): which pooling ops feed the heads; the old
  default `("mean","max","min","weighted")` re-introduces the leakage.

Current structured feature shapes:

| Tensor | Shape |
| --- | --- |
| `base_features` | `[games, 15]` |
| `synergy_objects` | `[games, 2, 10, 6]` |
| `matchup_objects` | `[games, 25, 6]` |
| `role_pair_type_ids` | `[10]` |
| `confidence_summaries` | `[games, 7]` |

`synergy_objects` and `matchup_objects` expose both the observed interaction
prior and the expected `1vx` baseline. The default `delta_baseline_mode` remains
`"logit"`, so deltas are interaction logits minus those logit-space baselines.

## What To Watch

The train/validation divergence is the headline signal. LOO encoding closed it
at the source:

```text
pre-fix:  train_nll ~= 0.51  val_nll ~= 0.80   (train AUC 0.83, val AUC 0.54)
current:  train_nll ~= 0.68  val_nll ~= 0.68   (train AUC 0.597, val AUC 0.599)
```

With LOO the train AUC sits at or *below* val. If a change re-opens the gap
(train AUC >> val, val_nll >> train_nll, best epoch 1), it is re-introducing
interaction leakage — check that `interaction_loo` is on and the cache is rebuilt.

For long experimentation, keep the scoreboard narrow. The only outputs that
matter for run-to-run comparison are:

- `val_accuracy`
- `val_ECE`
- `val_loss`
- `train_accuracy`
- `train_loss`
- `train_ECE`

In `metrics_latest.json`, `val_loss` and `train_loss` are currently stored as
`val.nll` and `train.nll`; `val_ECE` and `train_ECE` are stored as `val.ece`
and `train.ece`.

Useful checks after every non-trivial run:

```bash
uv run pyright app/ml
uv run ruff check app/ml
```

When tests are present for the ML package, run:

```bash
uv run pytest tests/ml -q
```

## Comparing Runs

For each long run, compare only the six experiment outputs above. Treat test
metrics and AUC as diagnostic details, not selection criteria.

Historical logistic benchmark (code removed; numbers kept for comparison only):

```text
test NLL ~= 0.6860   test accuracy ~= 0.5701   test AUC ~= 0.5945
```

## Notes For Future Refactors

Keep feature construction, tensor conversion, and artifact save/load centralized
in `structured_model.py`. `train.py` and `predictor.py` should stay thin around
their respective workflows so training and inference cannot quietly diverge.
