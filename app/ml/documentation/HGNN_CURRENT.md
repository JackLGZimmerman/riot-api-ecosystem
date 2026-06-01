# Current HGNN Mechanics

As of 2026-05-31, `HGNNWinModel` has one production path:

```text
cache/prior arrays
-> posterior and support features
-> champion/role/build identity + 1vX node prior
-> blue/red team readout
-> direct 1v1/2vX residual head
-> direct prior shortcut
-> final logit
-> sigmoid = P(blue wins)
```

The direct path was selected over the typed relation encoder because the full
same-split run showed it matched relation AUC and produced better NLL, Brier,
and ECE.

## Input Contract

Each row is one match with 10 ordered slots:

```text
0..4 = blue TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
5..9 = red  TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
```

The model consumes these `npy-memmap-v18` cache arrays:

| Array | Shape | Used as |
| --- | --- | --- |
| `win_rate.npy` | `[games, 10]` | smoothed `1vX` player priors |
| `matchup_1v1.npy` | `[games, 25]` | ordered blue-vs-red matchup priors |
| `synergy_2vx.npy` | `[games, 20]` | 10 blue and 10 red same-team pair priors |
| `p1_cnt.npy` | `[games, 10]` | raw `1vX` support |
| `m1v1_cnt.npy` | `[games, 25]` | raw build-level `1v1` support |
| `s2vx_cnt.npy` | `[games, 20]` | raw build-level `2vX` support |
| `champion_id.npy` | `[games, 10]` | champion embedding index |
| `build_id.npy` | `[games, 10]` | build embedding index |
| `blue_win.npy` | `[games]` | target label |

`m1v1_eff_n.npy` and `s2vx_eff_n.npy` may exist in the cache as nested-pooling
metadata, but the production direct model does not consume them.

## Posterior And Support Features

`1vX` rates are treated as posterior means for node initialization:

```text
mu = clamp(rate, 0, 1)
1vX variance = mu * (1 - mu) / (p1_cnt + confidence_strength + 1)
```

`confidence_strength` is saved with the artifact and is currently `30.0`.
Relationship means already include nested-pooling backoff before they reach the
model.

Relationship support enters the direct heads through raw build-level features:

```text
confidence = raw_count / (raw_count + confidence_strength)
log_count = log1p(raw_count)
missing = raw_count <= 0
```

## Relationship Deltas

The model builds logit-space residuals for each relationship prior:

```text
joint = logit(relationship_mu)
expected = logit(generic baseline from 1vX priors)
delta = joint - expected
```

For `1v1`, expected probability is:

```text
0.5 + (blue_1vX - red_1vX) / 2
```

For `2vX`, expected probability is the average of the two same-team players'
`1vX` priors.

The flattened direct relationship order is:

```text
25 blue-vs-red 1v1 deltas
10 blue 2vX deltas
10 negated red 2vX deltas
```

Red `2vX` deltas are negated so every direct relationship feature points in the
blue-win direction.

## Node Initialization

Each player starts from multiplicative identity embeddings:

```text
identity =
  champion_embedding(champion_id)
  * (1 + W_role(role_embedding(slot_role)))
  * (1 + W_build(build_embedding(build_id)))
```

The identity vector is layer-normalized, concatenated with the uncertainty-gated
`1vX` posterior embedding, then projected to the node dimension:

```text
identity:      [B, 10, 96]
1vX phi:       [B, 10, 64]
concat:        [B, 10, 160]
node_init MLP: 160 -> 96 -> 96
LayerNorm:     [B, 10, 96]
```

`PhiEncoder` for `1vX` uses:

```text
value inputs = [logit(mu), variance, confidence, log_count, missing]
gate inputs  = [1 / (1 + variance), confidence, log_count, missing]
phi = sigmoid(gate_mlp(gate inputs)) * value_mlp(value inputs)
```

Logits are clipped to `[-5, 5]`.

## Team Readout And Heads

Each team is read out from its five nodes:

```text
mean pool       [B, 96]
max pool        [B, 96]
attention pool  [B, 96]
concat          [B, 288]
team_proj       288 -> 96
```

This yields `a` for blue and `b` for red.

The residual head receives direct relationship features:

```text
delta                 45
confidence            45
delta * confidence    45
missing               45
total                180

residual_head: 180 -> 128 -> 96
```

The main head receives:

```text
[a, b, a - b, a * b, residual_head_output]
480 -> 256 -> 1
```

The prior shortcut is a direct linear logit path:

```text
blue 1vX logits          5
red 1vX logits           5
blue - red role logits   5
relationship deltas     45
deltas * confidence     45
total                  105

prior_shortcut: 105 -> 1
```

Final prediction:

```text
final_logit = main_head_logit + prior_shortcut_logit + detail_logit + profile_logit
P(blue wins) = sigmoid(final_logit)
```

Default source model size with `n_champions=951` and `n_builds=11` is `330,521`
parameters.

## Classification-Feature Integration

The classification pipeline can supply three extra per-game arrays:
`identity_semantic` (`[10, 64]` dense identity descriptors), `m1v1_detail`
(`[25, 16]` historic 1v1 pressure) and `s2vx_detail` (`[20, 16]` historic 2vX
synergy pressure). A game-level residual study (detail vs win-rate prior,
bucketed by prior centrality) showed:

- **1v1 matchup detail is redundant** with the matchup win-rate prior — zero
  residual correlation with the outcome, even in the central band. The
  win-rate prior already encodes how a matchup resolves.
- **The dense semantic identity is redundant** with the learned champion
  embedding.
- **2vX synergy detail carries a small residual signal**, concentrated where
  the win-rate prior is a coin-flip (the central band). Out-of-sample it lifts
  ranking by `+0.004` AUC at `|p-0.5|<0.03` and `+0.008` at `|p-0.5|<0.02`.

Feeding these as wide dense heads let the model memorise compositions
(train AUC `0.60 -> 0.63`, test `0.5925 -> 0.5875`). The production model
therefore consumes **only the 2vX synergy detail**, through a minimal,
overfitting-resistant term:

```text
team_diff = mean(blue synergy detail) - mean(red synergy detail)   # [B, 16]
detail_logit_raw = Linear(16 -> 1, no bias)(team_diff)             # antisymmetric
centrality = 4 * p * (1 - p)        # p = sigmoid(pre-detail logit), detached
detail_logit = detail_gate * centrality * detail_logit_raw         # gate init 0
```

`team_diff` flips sign under team swap, so the term is exactly antisymmetric
(no capacity spent learning the mirror). `centrality` restricts the nudge to
coin-flip matchups where the signal lives; the zero-init `detail_gate` makes the
whole term opt-in on top of the win-rate model. `_hgnn_config_from_meta` sets
`identity_semantic_dim=0` and `m1v1_detail_dim=0`; flip them to re-enable a
pathway for experiments.

## Training

`app/ml/train.py` builds `HGNNConfig` from cache identity metadata and trains
with:

| Setting | Value |
| --- | ---: |
| optimizer | AdamW |
| batch size request | 32768 |
| train batch cap | 7424 |
| max epochs | 40 |
| patience | 3 |
| learning rate | 0.001 |
| weight decay | 0.001 |
| gradient clip | 1.0 |
| checkpoint metric | `val_auc` |
| checkpoint min delta | 0.0005 |

Each batch is trained twice: original blue/red order with label `y`, and a
team-swapped mirror with label `1 - y`.

```text
loss = 0.5 * BCE(original_logit, y)
     + 0.5 * BCE(swapped_logit, 1 - y)
```

This encourages `P(A beats B) ~= 1 - P(B beats A)`.

Training saves `structured_winrate_model.pt` with `model_type`, `model_config`,
`confidence_strength`, and `state_dict`. Legacy artifacts may contain removed
config/state keys; `load_hgnn_model()` keeps only current config keys and loads
matching weights.

The current production artifact (synergy-detail term, current cache) has test
accuracy `0.5678`, threshold accuracy `0.5708`, AUC `0.5921`, NLL `0.6779`,
Brier `0.2425`, and ECE `0.0233`, with a train-test AUC gap of `0.008` (no
overfitting). Absolute numbers are not comparable to runs on earlier caches:
the current filtered dataset is smaller (1.15M vs 1.56M train games).

## Runtime Prediction

`load_predictor()` loads the saved HGNN artifact, prior tables, and nested-pooling
strengths from `cache_meta.json`.

For each draft state:

```text
champions + roles + build ids
-> blue/red (champion, role, build) tuples
-> prior table lookups
-> same smoothing/nested pooling as training
-> raw arrays matching the model contract
-> champion/build embedding ids
-> build_hgnn_inputs()
-> HGNNWinModel.forward()
-> sigmoid(final_logit)
```

Unknown champion/build ids map to the final embedding row. If
`use_final_build_labels=False`, every build lookup is forced to
`draft_unknown_build_label`.
