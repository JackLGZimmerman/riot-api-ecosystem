# ML Win-Rate Model

This documents the current win prediction baseline in `app/ml`.

## Model Type

The implementation is a linear probability model, not true logistic regression.
It fits the blue-side win label with least squares:

```text
predicted_blue_win = intercept + sum(feature_weight[i] * engineered_feature[i])
```

`predict()` clips the result into `[0.0, 1.0]`. A true logistic regression would
apply a sigmoid/logit link and train with a logistic loss; this code uses
`np.linalg.lstsq`.

## Input

Each cached game starts with 10 player win-rate priors:

```text
0-4 = blue TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
5-9 = red  TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
```

For each slot, `build_dataset.py` looks up the historical `win_rate` for that
player's `(championid, teamposition, build)` from
`game_data_filtered.synergy_1vx` using only rows where `split = 'train'`.
It then applies Bayesian smoothing toward a `0.5` prior:

```text
smoothed_win_rate = (wins + 0.5 * prior_strength) / (matchups + prior_strength)
```

The default `prior_strength` is `20`, equivalent to adding 10 wins and 10
losses to each aggregate. This mainly affects low-sample champion/role/build
combinations; high-sample priors stay close to their empirical win rate.
Missing priors have `matchups = 0`, so they remain `0.5`.

The target is `blue_win`, where `1` means the blue side won and `0` means the
red side won.

## Engineered Features

`model.py` turns the 10 slot priors into 17 linear-model features:

| Group | Features |
| --- | --- |
| Role differences | `top_diff`, `jungle_diff`, `middle_diff`, `bottom_diff`, `utility_diff` |
| Blue team stats | `blue_mean`, `blue_min`, `blue_max`, `blue_variance` |
| Red team stats | `red_mean`, `red_min`, `red_max`, `red_variance` |
| Team stat differences | `mean_diff`, `min_diff`, `max_diff`, `variance_diff` |

Each role difference is `blue_role - red_role`. Each stat difference is the
blue team statistic minus the red team statistic.

Skew and kurtosis are left as comments in `model.py`. They can be added later,
but each team has only five values, so those higher moments are noisy first-pass
features.

## Cache

`uv run python -m app.ml.build_dataset` writes:

| File | Shape | Meaning |
| --- | --- | --- |
| `app/ml/data/cache/win_rate.npy` | `[games, 10]` | Slot smoothed historical win-rate priors |
| `app/ml/data/cache/blue_win.npy` | `[games]` | Hard blue-win labels |
| `app/ml/data/cache/cache_meta.json` | object | Cache format, total rows, split counts |

The cache order is train, validation, then test. `dataset.py` reads
`cache_meta.json` and slices the arrays back into those splits.

## Training

`uv run python -m app.ml.train` loads the train split and solves:

```text
[1, engineered_feature_0, ..., engineered_feature_16] @ coefficients ~= blue_win
```

The first coefficient is the intercept. The remaining coefficients are one
weight per engineered feature. `metrics_latest.json` includes `feature_names`
next to `weights` so the coefficients can be inspected by name.

Training writes:

| File | Meaning |
| --- | --- |
| `app/ml/data/linear_winrate_model.npz` | Saved intercept and weights |
| `app/ml/data/metrics_latest.json` | Config, coefficients, and split metrics |

## Evaluation

The saved model is evaluated on train, validation, and test with:

- `mse` and `rmse` for probability error.
- `accuracy` using `prediction >= 0.5`.
- `auc` using prediction ranking.

Validation and test rows use priors built from the train split only, so their
labels are not used to create their input priors. Train metrics are in-sample:
train rows can be reflected in the train aggregate priors used as features.

## Future Considerations

- Bayesian Smoothing

We want to address the sparsity issue by applying bayesian smoothing with the most appropriate priors, to do this effectively we need to find similar classes to group together, that way the priors we use will be the most contextually relevant and adjust the win rate of the data accordingly.

- Neural Embeddings with Cosine Similarity
