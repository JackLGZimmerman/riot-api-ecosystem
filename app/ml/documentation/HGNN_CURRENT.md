# Current HGNN Mechanics

Updated: 2026-06-02.

`HGNNWinModel` uses a direct relationship path plus one optional context
residual. Production enables the raw identity-conditioned residual and selects
the checkpoint by validation threshold accuracy.

```text
cache/prior arrays
-> build_hgnn_inputs()
-> champion/role/build identity + 1vX posterior features
-> blue/red team readout
-> direct 1v1/2vX residual head
-> direct prior shortcut
-> context residual: raw identity-conditioned atlas (production) OR shared atlas
-> final logit
-> sigmoid = P(blue wins)
```

The direct path replaced the retired typed relation encoder because it matched
relation AUC and improved NLL, Brier, and ECE on the same split.

Context docs:

| Doc | Owns |
| --- | --- |
| [HGNN_CONTEXT_ATLAS.md](HGNN_CONTEXT_ATLAS.md) | Shared 24-dim atlas design and its historical limitation. |
| [HGNN_IDENTITY_CONDITIONED_CONTEXT.md](HGNN_IDENTITY_CONDITIONED_CONTEXT.md) | Production low-rank raw-atlas conditioned head and measured gains. |
| [HGNN_CONTEXT_WR_VALIDATION.md](HGNN_CONTEXT_WR_VALIDATION.md) | Global context-ceiling and calibration validation. |
| [HGNN_CONTEXT_EXAMPLES_AUDIT.md](HGNN_CONTEXT_EXAMPLES_AUDIT.md) | Concrete identity/context slices. |
| [../context_examples_audit.py](../context_examples_audit.py) | Reproducer for the examples-audit values. |

## Input Contract

Each row is one match with 10 ordered slots:

```text
0..4 = blue TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
5..9 = red  TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
```

The current cache format is `npy-memmap-v26`.

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
| `identity_context.npy` | `[games, 10, 24]` | shared atlas descriptor; dense tail also available to conditioned `raw_plus_dense` |
| `identity_context_support.npy` | `[games, 10]` | per-player historical support for the context gate |
| `identity_context_raw.npy` | `[games, 10, 62]` | production wide raw atlas for semantic context |
| `blue_win.npy` | `[games]` | target label |

Retained inspection/back-compat arrays are not consumed by the production direct
config: `identity_semantic.npy`, `identity_profile.npy`, `m1v1_detail.npy`,
`m1v1_eff_n.npy`, and `s2vx_eff_n.npy`. Experimental config flags can still wire
some of those paths back in.

## Posterior And Support Features

`1vX` rates are treated as posterior means for node initialization:

```text
mu = clamp(rate, 0, 1)
variance = mu * (1 - mu) / (p1_cnt + confidence_strength + 1)
```

`confidence_strength` is saved with the artifact and is currently `30.0`.

Raw support features enter the node encoder and direct relationship heads:

```text
confidence = raw_count / (raw_count + confidence_strength)
log_count = log1p(raw_count)
missing = raw_count <= 0
```

Relationship means already include upstream nested-pooling backoff before they
reach the model.

## Relationship Deltas

The model converts cached priors to logit-space residuals:

```text
joint = logit(relationship_mu)
expected = logit(generic baseline from 1vX priors)
delta = joint - expected
```

For `1v1`, the expected probability is:

```text
0.5 + (blue_1vX - red_1vX) / 2
```

For `2vX`, the expected probability is the average of the two same-team
players' `1vX` priors.

The direct relationship order is:

```text
25 blue-vs-red 1v1 deltas
10 blue 2vX deltas
10 negated red 2vX deltas
```

Red `2vX` deltas are negated so every direct feature points in the blue-win
direction.

## Node And Team Readout

Each player starts from multiplicative identity embeddings:

```text
identity =
  champion_embedding(champion_id)
  * (1 + W_role(role_embedding(slot_role)))
  * (1 + W_build(build_embedding(build_id)))
```

The identity vector is layer-normalized, concatenated with the uncertainty-gated
`1vX` posterior embedding, and projected to the node dimension:

| Tensor | Shape |
| --- | --- |
| identity | `[B, 10, 96]` |
| `1vX` phi | `[B, 10, 64]` |
| concat | `[B, 10, 160]` |
| node output | `[B, 10, 96]` |

`PhiEncoder` uses:

```text
value inputs = [logit(mu), variance, confidence, log_count, missing]
gate inputs  = [1 / (1 + variance), confidence, log_count, missing]
phi = sigmoid(gate_mlp(gate inputs)) * value_mlp(value inputs)
```

Logits are clipped to `[-5, 5]`.

Each team is read out from five nodes:

```text
mean pool       [B, 96]
max pool        [B, 96]
attention pool  [B, 96]
concat          [B, 288]
team_proj       [B, 96]
```

## Heads

The residual head receives relationship deltas and support:

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
[blue, red, blue - red, blue * red, residual_head_output]
480 -> 256 -> 1
```

The prior shortcut is a linear calibrated prior path:

```text
blue 1vX logits          5
red 1vX logits           5
blue - red role logits   5
relationship deltas     45
deltas * confidence     45
total                  105

prior_shortcut: 105 -> 1
```

## Context Residual

Only one context residual contributes in `forward()`:

| Condition | Context term |
| --- | --- |
| `use_identity_conditioned_context=true` and `identity_context_raw` is present | `IdentityConditionedContext` |
| otherwise, if `identity_context_dim > 0` and `identity_context` is present | shared `_context_logit` |
| otherwise | no context residual |

Shared atlas head:

```text
feat_p = [self, enemy_mean, enemy_damage_weighted_mean, lane_opp,
          ally_mean, products(7)]
conf_p = support_p / (support_p + context_support_strength)
context_logit = sum_blue conf_p * head(feat_p) - sum_red conf_p * head(feat_p)
```

Identity-conditioned head:

```text
context_feat_p  = [self, enemy_mean, enemy_weighted, lane_opp, ally_mean]
identity_cond_p = [champion_emb, role_emb, build_emb, self_raw]
z_id_p          = identity_conditioner(identity_cond_p)
z_ctx_p         = context_projector(context_feat_p)
raw_context_p   = init_scale * dot(z_id_p, z_ctx_p)
context_logit   = sum_blue conf_p * raw_context_p - sum_red conf_p * raw_context_p
```

Production uses `identity_context_source=raw`, rank `16`, hidden dim `64`, and
validation threshold-accuracy checkpointing. This is intentionally the naive
semantic context implementation: the model receives a wide historical identity
descriptor and a narrow learned context interaction, without champion-specific
rules.

Both terms are support-gated, antisymmetric under team swap, zero-initialized as
opt-in residuals, and draft-safe. A missing identity with zero context/support
contributes zero.

Final prediction:

```text
final_logit = main_head_logit + prior_shortcut_logit + context_logit
P(blue wins) = sigmoid(final_logit)
```

## Classification Inputs

The classification pipeline supplies pre-game identity descriptors keyed by
`(championid, teamposition, build)`.

| Output | Cache file | HGNN array | Shape |
| --- | --- | --- | --- |
| Context atlas | `identity_context_embedding.npz` | `identity_context` + `identity_context_support` | `[games, 10, 24]` + `[games, 10]` |
| Raw atlas | `identity_context_embedding.npz` (`raw_embeddings`) | `identity_context_raw` | `[games, 10, 62]` |

The 24-dim atlas is `[14 interpretable axes || 10 PCA axes]`. The 62-dim raw
atlas is `[same 14 interpretable axes || 48 median/MAD-standardized metrics]`.
No `participant_challenges` or `challenge_*` metrics are allowed.

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
| checkpoint metric | `val_threshold_accuracy` |
| checkpoint min delta | 0.0005 |

Each batch is trained twice: original blue/red order with label `y`, and a
team-swapped mirror with label `1 - y`.

```text
loss = 0.5 * BCE(original_logit, y)
     + 0.5 * BCE(swapped_logit, 1 - y)
```

Training saves `structured_winrate_model.pt` with `model_type`, `model_config`,
`confidence_strength`, and `state_dict`. `load_hgnn_model()` ignores removed
config keys and loads matching weights, so older artifacts can still be read
when their current-shape weights match.

## Current Results

Metrics are from the current filtered dataset (`1.15M` train games).

| Artifact | Context term | Test Acc | Test Thr Acc | Test AUC | Test NLL | Test Brier | Test ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `structured_winrate_model.pt` | raw identity-conditioned atlas, threshold-tuned | 0.5717 | 0.5743 | 0.5972 | 0.6763 | 0.2418 | 0.0222 |
| `experiments/identity_conditioned/cond_raw.pt` | raw identity-conditioned atlas, AUC-selected reference | 0.5714 | 0.5751 | 0.5979 | 0.6762 | 0.2417 | 0.0244 |
| previous `structured_winrate_model.pt` | shared 24-dim atlas | 0.5703 | 0.5735 | 0.5953 | 0.6770 | 0.2421 | 0.0244 |

The threshold-tuned identity-conditioned checkpoint is the production artifact.
The shared atlas remains available for baseline runs with `--shared-context`.

## Runtime Prediction

`load_predictor()` loads the HGNN artifact, prior tables, context lookups, and
nested-pooling strengths from `cache_meta.json`.

For each draft state:

```text
champions + roles + build ids
-> blue/red (champion, role, build) tuples
-> prior table lookups and smoothing
-> identity_context/support lookup
-> identity_context_raw lookup when the artifact needs it
-> champion/build embedding ids
-> build_hgnn_inputs()
-> HGNNWinModel.forward()
-> sigmoid(final_logit)
```

Unknown champion/build ids map to the final embedding row. If
`use_final_build_labels=False`, every build lookup is forced to
`draft_unknown_build_label`.
