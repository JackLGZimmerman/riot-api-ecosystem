# ML Win-Rate Model

This documents the production win prediction model in `app/ml`.

## Production Model

As of 2026-05-29, production uses the Structured Interaction Model With Cross
Layer.

Production training:

```bash
uv run python -m app.ml.train
```

Training writes:

| File | Meaning |
| --- | --- |
| `app/ml/data/structured_winrate_model.pt` | Structured PyTorch state dict plus model config |
| `app/ml/data/metrics_latest.json` | Training config, history, and split metrics |

`app/ml/predictor.py` loads the structured artifact by default through
`load_predictor()`. Runtime callers still pass teams, roles, and builds and get
one `P(blue wins)` float.

## Data Contract

An identity is `(champion_id, team_position, build)`.

The cache is built from train-only priors and chronological splits:

| Array | Shape | Meaning |
| --- | --- | --- |
| `win_rate.npy` | `[games, 10]` | Smoothed `1vx` identity priors |
| `matchup_1v1.npy` | `[games, 25]` | Smoothed ordered blue-vs-red `1v1` priors |
| `synergy_2vx.npy` | `[games, 20]` | Smoothed blue and red same-team `2vx` priors |
| `p1_cnt.npy` | `[games, 10]` | Support counts for `1vx` |
| `m1v1_cnt.npy` | `[games, 25]` | Support counts for `1v1` |
| `s2vx_cnt.npy` | `[games, 20]` | Support counts for `2vx` |
| `blue_win.npy` | `[games]` | Target label |

Structured training requires the count arrays. The cache order is train,
validation, then test; `app/ml/dataset.py` slices those arrays back into split
objects.

## Model Structure

The model path is:

```text
ClickHouse train-only priors
  -> app.ml.build_dataset cache arrays
  -> app.ml.dataset split loader
  -> structured feature builders
  -> StructuredWinModel
  -> structured_winrate_model.pt
  -> app.ml.predictor runtime P(blue wins)
```

`StructuredWinModel` combines three branch logits and confidence summaries:

| Branch | Input | Role |
| --- | --- | --- |
| Base identity | 10 identity logits plus 5 blue-minus-red role diffs | Main single-identity signal |
| Synergy | Blue and red `2vx` pair objects | Same-team pair effects |
| Matchup | 25 ordered `1v1` objects | Blue-vs-red matchup effects |
| Cross layer | Synergy contexts plus matchup embeddings | Lets team synergy adjust matchup interpretation |
| Final head | Branch logits plus confidence summaries | Calibrates the final win probability |

Pair and matchup objects carry observed prior, relevant solo priors, expected
baseline, support confidence, and observed-minus-expected delta. The final head
does not see raw champion IDs.

### Interaction config (current production)

**Why LOO + full delta:** train `1v1`/`2vx` priors are in-sample — a game's own
outcome sits in its own priors, leaking the label onto the `joint - expected` delta
at low support. The old config dropped that delta (`"raw"`), discarding real signal.
`interaction_loo` subtracts each train game's own outcome first, so `full` (delta-on)
now uses honest matchup and synergy signal. Test AUC 0.5928 → 0.5972.

| Knob | Production value | Why |
| --- | --- | --- |
| `interaction_loo` (dataset) | `True` | Leave-one-out encodes train priors (symmetrically across solo/`1v1`/`2vx`) so the delta no longer leaks. Val/test are train-derived and already leak-free, so untouched. |
| `object_feature_mode` | `"full"` | Keeps the `expected`/`delta` columns; with LOO the delta beats delta-off on val *and* test instead of overfitting. |
| `confidence_gate` | `True` | Scales each interaction embedding by support confidence, muting the noisiest low-support pairs. |
| `pooling_ops` | `("weighted",)` | Confidence-weighted mean only; `max`/`min` selected the lowest-support interactions. |
| `smoothing_mode` | `"cascade"` | Uses the raw contextual rate once support clears `prior_confidence_matchups`, else shrinks toward the broad/composite fallback. Avoids oversmoothing confident identities. |
| `confidence_gate_strength` | `30` | Confidence column `n/(n+s)` strength. Sweep over 5–150 peaks on a flat 28–35 plateau; below ~20 tail calibration degrades. Requires retrain. |

For tuning guidance and safe non-production output paths, see
[EXPERIMENTATION.md](EXPERIMENTATION.md).

## Runs To Keep

| Model | Test Acc | Test AUC | Test NLL | Test ECE |
| --- | ---: | ---: | ---: | ---: |
| L2 logistic reference (55 flat priors) | 0.5701 | 0.5945 | 0.6860 | — |
| Structured pre-fix (delta + max/min pool) | 0.5303 | 0.5384 | 0.8031 | 0.1699 |
| Structured leakage-robust raw (pre-LOO) | 0.5673 | 0.5928 | 0.6788 | 0.0204 |
| **Structured LOO + full delta + cascade (current)** | **0.5712** | **0.5972** | **0.6770** | **0.0201** |

The pre-fix structured model overfit the in-sample train leakage (train AUC 0.83,
val AUC 0.54, best epoch 1) and was strictly worse than the logistic baseline.
The pre-LOO `raw` config dropped the delta to survive that leakage; it matched the
logistic baseline but could not use the interaction signal. LOO encoding removes
the leak at the source, so `full` (delta-on) now beats both the pre-LOO config and
the logistic reference on AUC and NLL, while staying ~9x better calibrated than the
pre-fix model. Training is healthy — train AUC 0.5968 sits *below* val 0.5990
(no leakage gap), running 29 epochs.

The L2 logistic row is a historical benchmark only; its training code has been
removed from the package.
