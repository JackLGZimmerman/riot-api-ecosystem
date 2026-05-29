# ML Win-Rate Model

This documents the current win prediction baseline in `app/ml`.

## Identity

> An identity is the tuple `(champion_id, team_position, build)`.

Every prior in this project is keyed on identities (or, for interactions, on
pairs of identities). A `1vx` prior is the generic win rate of a single
identity; a `1v1` prior is one identity against another; a `2vx` prior is two
identities on the same team.

## Baseline and the Current Regression

The original model trained the win-rate logistic regression on the `1vx` synergy
prior **only** — the generic win rate of each identity. That baseline reached
**~57% accuracy** on validation/test.

Adding interaction features (`1v1` matchups and `2vx` synergies) pushed
**training** accuracy to ~70% but left **validation/test** accuracy at ~53% —
*below* the 57% `1vx`-only baseline. Investigation (see
[EXPERIMENTS evidence](../../classification/EXPERIMENTS.md)) found two causes:
the 45 interaction features overfit because the regression was **unregularised**,
and under-sampled pairs were smoothed to a flat `0.5` (no signal). The fixes are
an L2 penalty (recovers val/test to ~0.57 on its own) and the per-side fallback
prior (a small further ranking/AUC gain), both described below.

The benchmarks and the result after the fix (full 1.95M-game cache):

| Configuration | Val Acc | Test Acc | Val AUC | Val tail-ECE |
| --- | --- | --- | --- | --- |
| `1vx`-only baseline (target floor) | ~0.57 | ~0.57 | ~0.59 | — |
| `1vx` + interactions, no L2 (old) | 0.534 | 0.534 | 0.544 | 0.317 |
| `1vx` + interactions, **L2 + per-side fallback** | **0.569** | **0.570** | **0.596** | **0.112** |

Two changes recovered the model above the baseline: an L2 penalty on the
logistic weights (the dominant fix — alone it lifts val/test to ~0.568) and the
per-side fallback prior (a small further AUC/ranking gain). The train/val gap
fell from 0.167 to 0.065, and calibration improved sharply (tail-ECE 0.317 →
0.112). Diagnosis evidence is in
[../../classification/EXPERIMENTS.md](../../classification/EXPERIMENTS.md).

### Why the fix is validated in classification first

Before changing this model, the adaptive prior/smoothing strategy is prototyped
and evaluated in `app/classification` (see
[../../classification/README.md](../../classification/README.md)). That project
smooths the same `1vx` identity metrics with the same `9000-9040` prior
hierarchy, and it has a fast, self-contained grouping-quality rubric
([EXPERIMENTS.md](../../classification/EXPERIMENTS.md) /
[SPECIALISATIONS.md](../../classification/SPECIALISATIONS.md)) that does not
require retraining the win-rate model. If the smoothing change improves
classification grouping outcomes, that is evidence it is worth carrying into the
ML pipeline; if it does not, it is rejected before touching ML.

## Model Type

The implementation (`model.py`) is an L2-regularised logistic regression fit
with L-BFGS:

```text
predicted_blue_win = sigmoid(intercept + sum(feature_weight[i] * feature[i]))
```

The features are the 55 raw smoothed priors (no hand-engineered diffs/moments).
The intercept is unpenalised; weights carry an L2 penalty (`TrainConfig.l2`,
default `0.01`). Without the penalty the 45 interaction features overfit (train
~0.70, val/test ~0.53); the penalty keeps val/test near the ~0.57 baseline.

## Input

Each cached game has 55 prior features (`build_dataset.py`), all looked up from
ClickHouse dictionaries built on `split = 'train'` only:

```text
win_rate      10  blue then red player solo 1vx win rates (TOP..UTILITY)
matchup_1v1   25  each blue player vs each red player (blue perspective)
synergy_2vx   20  10 blue same-team pairs + 10 red same-team pairs
```

Each prior is Bayesian-smoothed (`prior_strength = 20`):

```text
smoothed = prior_mean + (raw_rate - prior_mean) * matchups / (matchups + strength)
```

Solo `win_rate` priors shrink toward `0.5`. Interaction priors (`matchup_1v1`,
`synergy_2vx`) shrink toward a **per-side composite** of the two sides' smoothed
solo priors rather than a flat `0.5` (`DatasetConfig.interaction_per_side_fallback`,
default on):

```text
1v1 prior (blue beats red) = 0.5 + (wr_blue - wr_red) / 2
2vx prior (same-team pair) = (wr_a + wr_b) / 2
```

So an under-sampled pair falls back to a real signal (its members' solo
strength) instead of the no-information `0.5`. This is the ML analogue of the
classification cascade: shrink toward the most specific prior with enough
support, then fall back. A genuinely well-sampled pair still dominates its own
smoothed value.

The target is `blue_win`, where `1` means the blue side won and `0` means the
red side won. The 55 priors are fed to the model directly; there are no
hand-engineered difference/moment features.

## Cache

`uv run python -m app.ml.build_dataset` writes:

| File | Shape | Meaning |
| --- | --- | --- |
| `app/ml/data/cache/win_rate.npy` | `[games, 10]` | Smoothed solo win-rate priors |
| `app/ml/data/cache/matchup_1v1.npy` | `[games, 25]` | Smoothed 1v1 matchup priors |
| `app/ml/data/cache/synergy_2vx.npy` | `[games, 20]` | Smoothed 2vx synergy priors |
| `app/ml/data/cache/blue_win.npy` | `[games]` | Hard blue-win labels |
| `app/ml/data/cache/cache_meta.json` | object | Cache format, counts, smoothing |

The cache order is train, validation, then test. `dataset.py` reads
`cache_meta.json` and slices the arrays back into those splits. The cache format
string bumps whenever the stored smoothing semantics change, so a stale cache is
rejected rather than silently mixed.

## Training

`uv run python -m app.ml.train` loads the train split and fits the regularised
logistic regression over the 55 priors. `metrics_latest.json` includes
`feature_names` next to `weights`, the `l2` used, and per-split metrics.

Training writes:

| File | Meaning |
| --- | --- |
| `app/ml/data/linear_winrate_model.npz` | Saved intercept and weights |
| `app/ml/data/metrics_latest.json` | Config, `l2`, coefficients, split metrics |

## Evaluation

The saved model is evaluated on train, validation, and test with:

- `accuracy` using `prediction >= 0.5` and `auc` using prediction ranking.
- `nll`, `brier`, `entropy`, and adaptive/tail `ece` for calibration.

Validation and test rows use priors built from the train split only, so their
labels are not used to create their input priors. Train metrics are in-sample:
train rows can be reflected in the train aggregate priors used as features.

## Adaptive Hierarchical Bayesian Smoothing

The adaptive prior-fallback strategy under evaluation picks the single
highest-priority prior whose sample size clears a confidence threshold instead
of pooling every prior level at once, so a well-sampled specific prior is not
contaminated by broad ones. For interactions (`1v1`, `2vx`), an additional
higher-priority fallback is the average of each side's individual identity prior
when the pair itself is under-sampled. This is prototyped and measured in
`app/classification` before being applied here.
