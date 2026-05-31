# ML Win-Rate Model

This documents the production win prediction path in `app/ml`.

## Production Model

As of 2026-05-31, production trains the Match-Outcome HGNN from
`app/ml/hgnn_model.py` via `app/ml/train.py`.

Rebuild the cache before training if the cache format or priors changed:

```bash
uv run python -m app.ml.build_dataset
uv run python -m app.ml.train
```

Training writes:

| File | Meaning |
| --- | --- |
| `app/ml/data/structured_winrate_model.pt` | HGNN state dict plus `HGNNConfig` |
| `app/ml/data/metrics_latest.json` | Training config, epoch history, and split metrics |

`app/ml/predictor.py` loads the artifact through `load_predictor()`. Runtime
callers pass teams, roles, and builds and receive one `P(blue wins)` float.

## Cache Contract

An identity is `(champion_id, team_position, build)`. The cache format is
`npy-memmap-v18`, ordered as train, validation, then test.

| Array | Shape | Meaning |
| --- | --- | --- |
| `win_rate.npy` | `[games, 10]` | Smoothed `1vx` identity priors |
| `matchup_1v1.npy` | `[games, 25]` | Nested-pooled ordered blue-vs-red `1v1` priors |
| `synergy_2vx.npy` | `[games, 20]` | Nested-pooled blue and red same-team `2vx` priors |
| `p1_cnt.npy` | `[games, 10]` | Support counts for `1vx` |
| `m1v1_cnt.npy` | `[games, 25]` | Build-level support counts for `1v1` |
| `s2vx_cnt.npy` | `[games, 20]` | Build-level support counts for `2vx` |
| `m1v1_eff_n.npy` | `[games, 25]` | Effective `1v1` sample size after nested pooling |
| `s2vx_eff_n.npy` | `[games, 20]` | Effective `2vx` sample size after nested pooling |
| `champion_id.npy` | `[games, 10]` | Per-slot champion id embedding index |
| `build_id.npy` | `[games, 10]` | Per-slot build index into `build_vocab` |
| `blue_win.npy` | `[games]` | Target label |

`cache_meta.json` records split sizes, smoothing settings, and
`identity = {n_champions, n_builds, build_vocab}`. `n_champions` is
`max(champion_id)+1` because champion ids are raw embedding indices.

The current HGNN consumes raw support: `p1_cnt` feeds the `1vx` posterior
variance, while `m1v1_cnt` and `s2vx_cnt` feed direct relationship confidence and
missing flags. The effective-support arrays remain in the cache for the
nested-pooled relationship priors and historical artifacts, but they are not live
model inputs.

## Smoothing

Build-conditioned interactions are nested-pooled from finest to coarsest:

```text
L0 (champ,role,build) x (champ,role,build)
L1 (champ,role)       x (champ,role)
L2  champ             x  champ
L3 per-side composite prior
```

Each level shrinks toward the next coarser level with empirical-Bayes strengths
recorded in `cache_meta.json` under `smoothing.interaction_level_strengths`.
Runtime prediction uses those stored strengths so training and inference match.

Set `DatasetConfig(use_final_build_labels=False)` only with no-build priors that
contain the configured `draft_unknown_build_label`; the cache builder fails if
those priors are missing.

## Model Shape

`HGNNWinModel` has one production path: identity + `1vx` node init, blue/red team
readout, direct `1v1`/`2vx` residual head, and direct prior shortcut. See
`HGNN_CURRENT.md` for the full mechanics.

Each of the 10 player slots starts from a multiplicative champion, role, and
build identity embedding. The node initializer then fuses that identity with the
slot's `1vx` posterior through the uncertainty-gated `PhiEncoder`.

The relationship feature contract is shared by `1v1` matchup and `2vx` synergy
priors:

```text
joint    = logit(mu_edge)
expected = logit(generic 1vx baseline)
delta    = joint - expected
```

Those relationship priors enter the final prediction through two direct paths:

- `residual_head`: an MLP over signed `1v1`/`2vx` deltas, raw confidence,
  confidence-weighted deltas, and missing flags.
- `prior_shortcut`: a linear shortcut over `1vx` identity logits plus signed
  deltas and confidence-weighted deltas.

`2vx` features are signed so blue-side synergies are positive and red-side
synergies are negated before they are concatenated with blue-vs-red `1v1`
features.

The default forward pass is:

```text
identity + 1vx posterior -> node init
blue/red team readouts + residual head -> neural logit
1vx logits + deltas + weighted deltas  -> prior shortcut logit
neural logit + shortcut logit          -> final logit -> sigmoid
```

Training uses team-swap augmentation through `swap_hgnn_inputs`: each match is
also trained mirrored with a flipped label.

## Reference Config

| Setting | Value |
| --- | ---: |
| `node_dim` | 96 |
| `edge_hidden` | 64 |
| champion / role / build embed | `n_champions+1` / 5 / `n_builds+1` x 96 |
| readout hidden | 256 |
| residual head hidden | 128 |
| dropout | 0.1 |
| logit clip | 5.0 |
| explicit count features | enabled |
| residual head / prior shortcut | enabled / enabled |
| params (approx) | ~0.33M |

## Benchmark Notes

The current source shape is the relationship-direct HGNN. Rerun training after
model-shape changes so `structured_winrate_model.pt` and `metrics_latest.json`
match the code.

| Model | Test Acc | Test AUC | Test NLL | Test ECE |
| --- | ---: | ---: | ---: | ---: |
| Relationship-direct HGNN | 0.5713 | 0.5985 | 0.6771 | 0.0255 |
| Previous structured production | 0.5712 | 0.5972 | 0.6770 | 0.0201 |
| L2 logistic reference (55 flat priors) | 0.5701 | 0.5945 | 0.6860 | - |

The structured and logistic rows are retained as historical benchmarks; their
training and test code has been removed from the package.
