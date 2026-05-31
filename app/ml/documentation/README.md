# ML Win-Rate Model

This documents the production win prediction path in `app/ml`.

## Production Model

As of 2026-05-30, production uses the Match-Outcome HGNN in
`app/ml/hgnn_model.py`.

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

The HGNN consumes both raw support and nested-pooling effective support:
`*_cnt` feeds explicit confidence features, while `m1v1_eff_n` and `s2vx_eff_n`
feed the posterior variance so backed-off interactions can still be treated as
confident.

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

`HGNNWinModel` treats the 10 players as nodes and `1v1` / `2vx` priors as
hyperedges. For each receiving node, the shared message function partitions edge
members into allies and enemies via `TEAM_OF`, so cross-team priors are read from
the receiver's perspective.

Each player node starts from a factored champion, role, and build identity, then
fuses the `1vx` posterior. Relationship edges encode:

```text
joint    = logit(mu_edge)
expected = logit(generic 1vx baseline)
delta    = joint - expected
```

The forward pass is:

```text
identity + 1vx posterior -> node init
2vx hypergraph rounds    -> synergy-aware nodes
1v1 hypergraph rounds    -> matchup-aware nodes
team readout + residual/prior shortcut -> final logit -> sigmoid
```

Training uses team-swap augmentation through `swap_hgnn_inputs`: each match is
also trained mirrored with a flipped label.

## Reference Config

| Setting | Value |
| --- | ---: |
| `node_dim` / `msg_dim` | 96 / 96 |
| `edge_hidden` | 64 |
| champion / role / build embed | `n_champions+1` / 5 / `n_builds+1` x 96 |
| intra / cross rounds | 2 / 2 |
| readout hidden | 256 |
| residual head hidden | 128 |
| dropout | 0.1 |
| logit clip | 5.0 |
| explicit count features | enabled |
| explicit edge residual features | enabled |
| direct residual head / prior shortcut | enabled / enabled |
| params (approx) | ~0.69M |

## Metrics To Retain

| Model | Test Acc | Test AUC | Test NLL | Test ECE |
| --- | ---: | ---: | ---: | ---: |
| L2 logistic reference (55 flat priors) | 0.5701 | 0.5945 | 0.6860 | - |
| Structured pre-fix (delta + max/min pool) | 0.5303 | 0.5384 | 0.8031 | 0.1699 |
| Structured leakage-robust raw (pre-LOO) | 0.5673 | 0.5928 | 0.6788 | 0.0204 |
| Structured LOO + full delta + cascade (prev. production) | 0.5712 | 0.5972 | 0.6770 | 0.0201 |
| Match-Outcome HGNN (priors-only node init) | 0.5704 | 0.5947 | 0.6776 | 0.0167 |
| Match-Outcome HGNN + identity (current) | 0.5709 | 0.5984 | 0.6775 | 0.0297 |
| HGNN + count/residual/head shortcut | 0.5707 | 0.5993 | 0.6772 | 0.0294 |
| HGNN + v18 effective support + residual/head shortcut | 0.5718 | 0.5992 | 0.6771 | 0.0263 |

The structured and logistic rows are retained as historical benchmarks; their
training and test code has been removed from the package.
