# ML Win-Rate Model

As of 2026-06-10, production uses champion/build identity embeddings on top of
the smoothed 1vX player prior, the production Loadout head, the bounded
patch-only Temporal head, and the promoted all-encoder learned semantic MoE over
frozen static, full-game, and temporal identity latents. The old
classification-derived semantic, profile, context, and direct relationship
inputs have been removed from the HGNN contract.

The current identity-signal surface is the frozen identity-encoder sidecar
artifact at `app/ml/data/semantic_identity_sidecar_compact.npz`. Production
consumes all three sidecar blocks through
`semantic_moe_architecture="convex_encoder_mix"`; the older node-init sidecar
MLPs remain disabled by default. See
[HGNN_CURRENT.md](HGNN_CURRENT.md#identity-encoder-sidecars).

Build-intent work is tracked separately in
[HGNN_BUILD_INTENT.md](HGNN_BUILD_INTENT.md). That plan keeps observed final
build labels as oracle diagnostics only and defines the accepted path as
train-only historical build-profile prior marginalisation.

## Production Path

```text
cache/prior arrays
-> posterior and support features
-> champion/role/build identity + 1vX node prior
-> Loadout + patch-only Temporal residual heads
-> static/full-game/temporal sidecars into convex semantic MoE
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
| `app/ml/data/hgnn_production_model.pt` | HGNN config, confidence strength, and state dict |
| `app/ml/data/metrics_latest.json` | Train/val/test metrics and epoch history |

Runtime prediction uses `load_predictor()` from `app/ml/predictor.py`.

## Cache Contract

The model consumes `npy-memmap-v29` cache arrays with 10 ordered slots:

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
| `loadout_features.npy` | `[games, 10]` | production Loadout residual features |
| `patch_features.npy` | `[games, 2]` | bounded patch-only Temporal residual features |
| `blue_win.npy` | `[games]` | target label |

The default model path now points at the promoted `convex_encoder_mix` semantic
MoE checkpoint copied into `app/ml/data/hgnn_production_model.pt`.

## Identity Semantic Context

`HGNNConfig.use_learned_semantic_moe=True` enables the production side-logit over
the frozen static, full-game, and temporal identity sidecars. The promoted
architecture is `convex_encoder_mix`, selected because it kept all three encoder
views present while producing the lowest validation group-EB semantic gap in the
architecture matrix. The older identity semantic context head remains a
model-level research path only; `train.py` production testing uses the learned
semantic MoE route.

## Training

`app/ml/train.py` trains one production model shape with AdamW, team-swap
augmentation, validation temperature diagnostics, and raw validation-accuracy
checkpointing.

Each batch is trained twice: original blue/red order with label `y`, and a
team-swapped mirror with label `1 - y`.

```text
loss = 0.5 * BCE(original_logit, y)
     + 0.5 * BCE(swapped_logit, 1 - y)
```
