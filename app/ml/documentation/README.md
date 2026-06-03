# ML Win-Rate Model

As of 2026-06-03, production uses champion/build identity embeddings on top of
the smoothed 1vX player prior. The old classification-derived semantic,
profile, and context inputs have been removed from the HGNN contract.

Direct 1v1 and 2vX relationship integrations remain as explicit research
capacity behind `HGNNConfig.use_relationship_integrations=True`; default
training and serving leave them disabled. Identity-encoder sidecars
(static / full-game / temporal) are the current identity-signal research
surface, and the semantic context head can aggregate those latents into
ally/enemy context logits. Both are disabled by default — see
[HGNN_CURRENT.md](HGNN_CURRENT.md#identity-encoder-sidecars).

## Production Path

```text
cache/prior arrays
-> posterior and support features
-> champion/role/build identity + 1vX node prior
-> optional static/full-game/temporal sidecars and semantic context head
-> blue/red mean + attention team readout
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

The model consumes `npy-memmap-v27` cache arrays with 10 ordered slots:

```text
0..4 = blue TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
5..9 = red  TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
```

| Array | Shape | Live model use |
| --- | --- | --- |
| `win_rate.npy` | `[games, 10]` | smoothed `1vX` player priors |
| `p1_cnt.npy` | `[games, 10]` | raw `1vX` support for node confidence |
| `champion_id.npy` | `[games, 10]` | champion embedding index |
| `build_id.npy` | `[games, 10]` | build embedding index |
| `blue_win.npy` | `[games]` | target label |

Relationship arrays are still written for opt-in research runs:

| Array | Shape | Default use |
| --- | --- | --- |
| `matchup_1v1.npy` | `[games, 25]` | ignored |
| `synergy_2vx.npy` | `[games, 20]` | ignored |
| `m1v1_cnt.npy` | `[games, 25]` | ignored |
| `s2vx_cnt.npy` | `[games, 20]` | ignored |
| `m1v1_eff_n.npy` | `[games, 25]` | ignored |
| `s2vx_eff_n.npy` | `[games, 20]` | ignored |

## Relationship Features

When relationship integrations are enabled, the residual features are:

```text
1v1 delta        = logit(blue beats red prior) - logit(generic 1vX baseline)
blue 2vX delta   = +team-local synergy delta
red 2vX delta    = -team-local synergy delta
confidence       = raw_count / (raw_count + confidence_strength)
missing          = raw_count <= 0
```

The default model path does not feed those tensors into the head.

## Identity Semantic Context

`HGNNConfig.use_identity_semantic_context_head=True` enables an opt-in
side-logit over the frozen static, full-game, and temporal identity sidecars.
It projects the three latent blocks into a shared semantic vector, builds
support-weighted ally and enemy summaries for each slot, scores the interaction,
and returns `base_logit`, `context_logit`, and `final_logit`. The final context
layer is zero-initialised, so enabling the flag starts as a no-op.

The recommended experiment variant is `all_three_plus_semantic_context` in
`app/ml/experiments/context_ablation.py`.

## Training

`app/ml/train.py` trains one production model shape with AdamW, team-swap
augmentation, validation temperature diagnostics, and validation
threshold-accuracy checkpointing.

Each batch is trained twice: original blue/red order with label `y`, and a
team-swapped mirror with label `1 - y`.

```text
loss = 0.5 * BCE(original_logit, y)
     + 0.5 * BCE(swapped_logit, 1 - y)
```
