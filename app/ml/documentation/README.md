# ML Win-Rate Model

This documents the production win prediction model in `app/ml`.

## Production Model

As of 2026-05-29, production uses the Structured Interaction Model With Cross
Layer. The earlier L2-regularised logistic regression remains in
`app/ml/model.py` only as a historical baseline.

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
| `matchup_1v1.npy` | `[games, 25]` | Ordered blue-vs-red `1v1` priors |
| `synergy_2vx.npy` | `[games, 20]` | Blue and red same-team `2vx` priors |
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

### Leakage-robust interaction config (current production)

Train interaction priors are **in-sample**: a train game's own outcome is folded
into its `1v1`/`2vx` priors. `1vx` identities have high support so this is
negligible, but low-support `1v1`/`2vx` pairs leak the label heavily. The
interaction branches memorise that leakage, which generalises to nothing. Three
`StructuredModelConfig` knobs make them leakage-robust:

| Knob | Production value | Why |
| --- | --- | --- |
| `object_feature_mode` | `"raw"` | Drops the `expected` and `delta` columns. The `joint - expected` delta isolates exactly the low-support leakage (linear val AUC 0.55 on deltas vs 0.59 on raw logit priors). |
| `confidence_gate` | `True` | Scales each interaction embedding by its support confidence, so low-support (leaky) pairs contribute ~0. |
| `pooling_ops` | `("weighted",)` | Confidence-weighted mean only. `max`/`min` pooling selected the most extreme = lowest-support = leakiest interactions. |

Large `batch_size` (32768) is also load-bearing: it acts as implicit
regularization, slowing the first-epoch fit of the residual leakage so the model
holds the honest ceiling instead of collapsing.

For tuning guidance and safe non-production output paths, see
[EXPERIMENTATION.md](EXPERIMENTATION.md).

## Runs To Keep

| Model | Test Acc | Test AUC | Test NLL | Test ECE |
| --- | ---: | ---: | ---: | ---: |
| L2 logistic reference (55 flat priors) | 0.5701 | 0.5945 | 0.6860 | — |
| Structured pre-fix (delta + max/min pool) | 0.5303 | 0.5384 | 0.8031 | 0.1699 |
| **Structured leakage-robust (current)** | **0.5665** | **0.5918** | **0.6792** | **0.0219** |

The pre-fix structured model overfit the in-sample train leakage (train AUC 0.83,
val AUC 0.54, best epoch 1) and was strictly worse than the logistic baseline.
The leakage-robust config matches the logistic baseline on accuracy/AUC, beats it
on NLL, and is ~8x better calibrated, with a healthy train/val gap (train AUC
0.624 vs val 0.594, best epoch 2).

The logistic model remains the rollback/reference point:

```text
app/ml/data/linear_winrate_model.npz
```

## Findings (puzzle investigation)

Diagnosis from controlled ablations (`app/ml/experiments/`):

1. **Where the signal lives.** The `base` identity branch alone reaches val AUC
   0.594 / test acc 0.570 with no overfitting. Adding the synergy/matchup
   branches *collapsed* val AUC to ~0.54.
2. **Why.** Train interaction priors are in-sample (leaky). Negative controls
   confirmed it: real interactions (test AUC 0.538) were **worse than noise** —
   shuffling all interactions recovered base level (0.593), because the model
   then ignores them. The `delta` representation and `max`/`min` pooling
   amplified the low-support leakage.
3. **The interaction signal is real but tiny.** Trained on *honest* priors (the
   out-of-sample val split), the full model does not overfit and beats base by
   only +0.003 AUC. A learning curve shows that gap plateauing near +0.003.
4. **Is ~60% reachable?** No, not from these smoothed priors. The honest signal
   ceiling is ~0.59–0.60 AUC / ~0.57–0.58 accuracy. 60% accuracy needs ≈0.62
   AUC. The fix recovers the model to that honest ceiling and removes the
   collapse, but the priors do not contain enough extractable interaction signal
   for 60%.

### Remaining bottleneck / next adjustment

The blocker to more interaction signal is **train-side in-sample prior leakage**,
not the model. To extract the (small) remaining interaction signal:

- Rebuild the train cache with **out-of-sample / leave-one-out** interaction
  priors so training sees honest `1v1`/`2vx` features (`build_dataset.py` /
  ClickHouse dictionaries). The honest-training experiments prove the model is
  then well-behaved.
- Test **lower smoothing** (`smoothing_prior_strength` < 20) on high-support
  interactions once training is honest, to stop washing out their signal.

Reproduce the investigation:

```bash
uv run python -m app.ml.experiments.stage0_diagnose       # ablations + logistic
uv run python -m app.ml.experiments.stage2_controls       # negative controls + buckets
uv run python -m app.ml.experiments.stage3_linear_decomp  # raw vs delta signal ceiling
uv run python -m app.ml.experiments.stage4_honest_training # honest-prior training
uv run python -m app.ml.experiments.stage6_robust         # leakage-robust config sweep
```
