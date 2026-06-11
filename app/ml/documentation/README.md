# ML Win-Rate Model

As of 2026-06-10, the production model uses champion/build identity embeddings on top of
the smoothed 1vX champion-role/build prior, the production Loadout head, the bounded
patch-only Temporal head, and the promoted all-encoder learned semantic MoE over
frozen static, full-game, and temporal identity latents. The old
classification-derived semantic, profile, context, and direct relationship
inputs have been removed from the HGNN contract.

The current identity-signal surface is the frozen identity-encoder sidecar
artifact at `app/ml/data/semantic_identity_sidecar_compact.npz`. Production
consumes all three sidecar blocks through the learned semantic MoE; the older
node-init sidecar MLPs were removed. See
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
-> static/full-game/temporal sidecars into learned semantic MoE
-> blue/red mean + attention team readout
-> final logit
-> sigmoid = P(blue wins)
```

Train the model with:

```bash
uv run python -m app.ml.train \
  --model-path app/ml/data/experiments/manual/model.pt \
  --metrics-path app/ml/data/experiments/manual/metrics.json
```

Training tensor-caches the train and test splits, evaluates test every epoch,
selects the best checkpoint by raw test accuracy, and writes lean accuracy/NLL
metrics. Test is the model-selection split, not a final untouched holdout. Add
`--allow-production-artifact-overwrite` only for an explicit promotion run.

Training writes:

| File | Meaning |
| --- | --- |
| Candidate `model.pt` | HGNN config, confidence strength, and state dict |
| Candidate `metrics.json` | Train/test metrics and epoch history with `selection_split: "test"` |

The promoted load paths remain `app/ml/data/hgnn_production_model.pt` and
`app/ml/data/metrics_latest.json`. Runtime prediction uses `load_predictor()`
from `app/ml/predictor.py`; it now fails fast for checkpoints that require
Loadout, patch, or player-prior tensors because the current
`app.rl.reward.Predictor` protocol only supplies champions, roles, and build
ids.

## Cache Contract

The model consumes `npy-memmap-v32` cache arrays with 10 ordered slots. Splits
are per-patch chronological 80/20 train/test (no validation range); older
caches with a validation split must be rebuilt:

```text
0..4 = blue TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
5..9 = red  TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
```

| Array | Shape | Live model use |
| --- | --- | --- |
| `win_rate.npy` | `[games, 10]` | smoothed `1vX` champion-role/build priors |
| `p1_cnt.npy` | `[games, 10]` | raw `1vX` support for node confidence |
| `champion_id.npy` | `[games, 10]` | champion embedding index |
| `build_id.npy` | `[games, 10]` | build embedding index |
| `loadout_features.npy` | `[games, 10]` | production Loadout residual features |
| `patch_features.npy` | `[games, 2]` | bounded patch-only Temporal residual features |
| `player_rate.npy`, `player_cnt.npy` | `[games, 10]` | optional per-player overall priors; disabled in the production model until serving supplies player features |
| `player_champ_rate.npy`, `player_champ_cnt.npy` | `[games, 10]` | optional per-player/champion priors; disabled in the production model until serving supplies player features |
| `blue_win.npy` | `[games]` | target label |

The default model path now points at the promoted learned semantic MoE
checkpoint copied into `app/ml/data/hgnn_production_model.pt`.

## Identity Semantic Context

`HGNNConfig.use_learned_semantic_moe=True` enables the production side-logit over
the frozen static, full-game, and temporal identity sidecars. The maintained path
keeps all three encoder views present through the learned semantic MoE and
compact group features; older direct context heads and node-init sidecar MLPs
have been removed from the production surface.

## Training

`app/ml/train.py` trains one production model shape with AdamW, team-swap
augmentation, raw test-accuracy checkpointing, and accuracy/NLL reporting.

Each batch is trained twice: original blue/red order with label `y`, and a
team-swapped mirror with label `1 - y`.

```text
loss = 0.5 * BCE(original_logit, y)
     + 0.5 * BCE(swapped_logit, 1 - y)
```
