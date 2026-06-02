# ML Win-Rate Model

As of 2026-06-02, production uses the direct relationship HGNN path plus the
threshold-tuned raw identity-conditioned context head. This is the naive
semantic-context implementation: a wide draft-safe identity atlas, a low-rank
identity-specific ally/enemy interaction, and no manual champion-specific rules.
The shared 24-dim context-atlas head remains available as `--shared-context` for
baseline runs. Context docs:
[HGNN_CONTEXT_ATLAS.md](HGNN_CONTEXT_ATLAS.md) and
[HGNN_IDENTITY_CONDITIONED_CONTEXT.md](HGNN_IDENTITY_CONDITIONED_CONTEXT.md).

## Production Path

```text
cache/prior arrays
-> posterior and support features
-> champion/role/build identity + 1vX node prior
-> blue/red team readout
-> direct 1v1/2vX residual head
-> direct prior shortcut
-> raw identity-conditioned context interaction (support-gated, antisymmetric)
-> final logit
-> sigmoid = P(blue wins)
```

Train the model with:

```bash
uv run python -m app.ml.train
```

Training writes:

| File | Meaning |
| --- | --- |
| `app/ml/data/structured_winrate_model.pt` | HGNN config, confidence strength, and state dict |
| `app/ml/data/metrics_latest.json` | Train/val/test metrics and epoch history |

Runtime prediction uses `load_predictor()` from `app/ml/predictor.py`.

## Cache Contract

The model consumes `npy-memmap-v26` cache arrays with 10 ordered slots:

```text
0..4 = blue TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
5..9 = red  TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
```

| Array | Shape | Live model use |
| --- | --- | --- |
| `win_rate.npy` | `[games, 10]` | smoothed `1vX` player priors |
| `matchup_1v1.npy` | `[games, 25]` | ordered blue-vs-red `1v1` priors |
| `synergy_2vx.npy` | `[games, 20]` | 10 blue and 10 red same-team `2vX` priors |
| `p1_cnt.npy` | `[games, 10]` | raw `1vX` support for node confidence |
| `m1v1_cnt.npy` | `[games, 25]` | raw `1v1` support for direct confidence/missing features |
| `s2vx_cnt.npy` | `[games, 20]` | raw `2vX` support for direct confidence/missing features |
| `champion_id.npy` | `[games, 10]` | champion embedding index |
| `build_id.npy` | `[games, 10]` | build embedding index |
| `identity_context.npy` | `[games, 10, 24]` | shared descriptor; dense tail available to `raw_plus_dense` experiments |
| `identity_context_support.npy` | `[games, 10]` | per-player historical support (context head gate) |
| `identity_context_raw.npy` | `[games, 10, 62]` | production raw semantic atlas for the identity-conditioned head |
| `blue_win.npy` | `[games]` | target label |

`identity_semantic.npy` `[games, 10, 64]` and `identity_profile.npy`
`[games, 10, 9]` are retained for inspection/back-compat but unused by the
production model. The cache may also contain `m1v1_eff_n.npy` / `s2vx_eff_n.npy`
from nested pooling; the production direct HGNN does not consume them either.

## Relationship Features

The direct path keeps every relationship feature in blue-win direction:

```text
1v1 delta        = logit(blue beats red prior) - logit(generic 1vX baseline)
blue 2vX delta   = +team-local synergy delta
red 2vX delta    = -team-local synergy delta
confidence       = raw_count / (raw_count + confidence_strength)
missing          = raw_count <= 0
```

The residual head receives:

```text
delta                 45
confidence            45
delta * confidence    45
missing               45
total                180
```

The prior shortcut receives:

```text
blue 1vX logits          5
red 1vX logits           5
blue - red role logits   5
relationship deltas     45
deltas * confidence     45
total                  105
```

Final prediction:

```text
final_logit = main_head_logit + prior_shortcut_logit + context_logit
P(blue wins) = sigmoid(final_logit)
```

## Training

`app/ml/train.py` trains one production model shape with AdamW, team-swap
augmentation, and validation threshold-accuracy checkpointing.

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

The swap flips the `1v1` matrix into the new blue perspective and negates signed
relationship logits where needed.

## Current Result

The same-split ablation selected the direct model:

| Variant | Test AUC | Test NLL | Test Brier | Test ECE |
| --- | ---: | ---: | ---: | ---: |
| onevX only | 0.5938 | 0.6788 | 0.2430 | 0.0291 |
| direct 1v1 + 2vX | 0.5998 | 0.6765 | 0.2418 | 0.0251 |
| relation encoder | 0.5998 | 0.6767 | 0.2419 | 0.0266 |

Production therefore keeps the calibrated direct priors and direct sparse
residual corrections, removes the typed relation transfer layer, and adds the
threshold-tuned raw semantic-context term for enemy/ally-composition context.

The current production semantic-context artifact saved
`app/ml/data/structured_winrate_model.pt` with:

| Split | Accuracy | Threshold Acc | AUC | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.5735 | 0.5710 | 0.6038 | 0.6747 | 0.2410 | 0.0037 |
| val | 0.5750 | 0.5779 | 0.6014 | 0.6749 | 0.2411 | 0.0252 |
| test | 0.5717 | 0.5743 | 0.5972 | 0.6763 | 0.2418 | 0.0222 |

This promotes the audited identity-conditioned raw atlas into production and
selects the checkpoint by threshold accuracy. Concrete context slices are in
[HGNN_CONTEXT_EXAMPLES_AUDIT.md](HGNN_CONTEXT_EXAMPLES_AUDIT.md).
