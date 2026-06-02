# ML Win-Rate Model

As of 2026-06-02, production uses the threshold-tuned raw
identity-conditioned context head on top of the 1vX player prior. Direct 1v1 and
2vX integrations are disabled by default after an accuracy-neutral removal audit;
the loader still keeps those arrays for future research or legacy artifacts. The
shared 24-dim context-atlas head remains available as `--shared-context` for
baseline runs. Maintained iteration surfaces are
`app/ml/experiments/context_ablation.py` and `app/ml/context_examples_audit.py`.

## Production Path

```text
cache/prior arrays
-> posterior and support features
-> champion/role/build identity + 1vX node prior
-> blue/red mean + attention team readout
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
| `matchup_1v1.npy` | `[games, 25]` | loader-retained; not used by default training/serving |
| `synergy_2vx.npy` | `[games, 20]` | loader-retained; not used by default training/serving |
| `p1_cnt.npy` | `[games, 10]` | raw `1vX` support for node confidence |
| `m1v1_cnt.npy` | `[games, 25]` | loader-retained relationship support |
| `s2vx_cnt.npy` | `[games, 20]` | loader-retained relationship support |
| `champion_id.npy` | `[games, 10]` | champion embedding index |
| `build_id.npy` | `[games, 10]` | build embedding index |
| `identity_context.npy` | `[games, 10, 24]` | shared descriptor; dense tail available to `raw_plus_dense` experiments |
| `identity_context_support.npy` | `[games, 10]` | per-player historical support (context head gate) |
| `identity_context_raw.npy` | `[games, 10, 62]` | production raw semantic atlas for the identity-conditioned head |
| `blue_win.npy` | `[games]` | target label |

`identity_semantic.npy` `[games, 10, 64]`, `identity_profile.npy`
`[games, 10, 9]`, and `m1v1_eff_n.npy` / `s2vx_eff_n.npy` are retained for
inspection, back-compat, and future experiments but unused by the default model.

## Relationship Features

Direct relationship features are retained only behind
`HGNNConfig.use_relationship_integrations=True` for legacy/research runs:

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

Final prediction:

```text
final_logit = main_head_logit + context_logit
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

The swap mirrors blue/red slots and, for explicit legacy relationship runs,
flips the `1v1` matrix into the new blue perspective and negates signed
relationship logits where needed.

## Current Result

The active post-removal `val_nll_ece` verification run selected the default
no-relationship model path:

| Variant | Val Threshold Acc | Val AUC | Test Threshold Acc | Test AUC | Test ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| no direct 1v1/2vX, `val_nll_ece` | 0.57746 | 0.60074 | 0.57319 | 0.59539 | 0.03131 |

The measured accuracy/AUC movement versus the prior relationship-enabled
calibration-aware run stayed below `0.005`, while calibration worsened. The next
iteration should focus on recovering calibration without restoring direct
1v1/2vX data to the default pipeline.

The removal verification artifact saved
`app/ml/data/experiments/context_ablation_relationship_removed_iter/low_rank_checkpoint_nll_ece/model.pt`
with:

| Split | Accuracy | Threshold Acc | AUC | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.5731 | 0.5723 | 0.6047 | 0.6745 | 0.2409 | 0.0108 |
| val | 0.5717 | 0.5775 | 0.6007 | 0.6761 | 0.2417 | 0.0341 |
| test | 0.5674 | 0.5732 | 0.5954 | 0.6778 | 0.2425 | 0.0313 |

Concrete context slices are in
[HGNN_CONTEXT_EXAMPLES_AUDIT.md](HGNN_CONTEXT_EXAMPLES_AUDIT.md).
