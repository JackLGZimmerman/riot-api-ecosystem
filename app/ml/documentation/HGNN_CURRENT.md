# HGNN Current State

Last updated: 2026-06-11.

The model is draft-generic by hard constraint: no player information of any
kind (no puuids, no player priors, no rank). The admissible surface is
champions, positions, runes, summoners, bans, patch, and
(champion, role, build)-keyed historical profiles via the encoder sidecars.

## Split Protocol (v32)

`ml_game_split` labels are per-patch chronological 80/20 train/test — games
are partitioned by `(season, patch)`, ordered by game start, and each patch's
first 80% goes to train with the remainder to test. There is no validation
split; test is the model-selection split for checkpoint selection and
accuracy/NLL reporting, not a final untouched holdout. The cache format is
`npy-memmap-v32`; older validation-bearing caches must be rebuilt. This makes
same-patch history available to train-side priors and features before each
patch's scored tail.

Protocol validation (2026-06-11): the ClickHouse split, pivot, and
split-scoped aggregates were rebuilt (`1,318,329` train / `329,586` test, 11
patches all at 80/20) along with the v32 cache, and three default-recipe seeds
were trained. Single-seed test accuracy is `0.5784`–`0.5792` (mean `0.5788`,
stdev `0.0004`) with test NLL `0.6719`–`0.6727` — recovering the old
protocol's validation level and beating its frozen-tail test by `+0.50pp`
accuracy and `-0.0037` NLL. The old val-over-test gap was freshness, and the
per-patch split removes it: per-patch test accuracy spans only
`0.5742`–`0.5819` (at the binomial sampling floor per patch), and accuracy
across chronological quartiles of each patch's test tail is flat, so no
measurable within-patch freshness decay remains. Remaining headroom is
model/feature, not split mechanics; cross-patch train weighting is still an
open, untested lever.

Follow-up draft-only residual probes (see EXPERIMENTS.md, 2026-06-11) found
two bankable levers — a 3-seed ensemble (`+0.26pp` acc, `-0.0011` NLL over a
refit single seed) and a train-fitted side intercept the swap-augmented model
cannot express — and otherwise no linear or shallow-nonlinear residual in
bans, loadout, encoder latents, or role-aligned lane diffs; the stable hard
core is well-covered balanced drafts, not a data blind spot. Both levers are
now promoted (see Production Status).

## Production Path

Default training and evaluation use the 1vX champion-role/build prior, champion/build identity
embeddings, the production Loadout head, the production patch-only Temporal
head, the promoted learned semantic MoE over all three frozen identity encoders
(`static`, `full_game`, and `temporal`), and team-swap augmentation. Legacy
classification-derived semantic, profile, and context inputs are no longer part
of `build_hgnn_inputs()` or `HGNNWinModel.forward()`.

The promoted semantic path is the fixed learned MoE over static, full-game, and
temporal sidecar latents plus compact semantic group features. Production
capacity is 128 experts with `top_k=32`. See
[Identity Encoder Sidecars](#identity-encoder-sidecars).

Serving through `app.ml.predictor.load_predictor()` is intentionally narrower
than the batch validation surface: the current `app.rl.reward.Predictor`
protocol supplies champions, roles, and build ids, but not Loadout or patch
feature tensors. Checkpoints that require those residual heads fail fast at
load time instead of silently dropping trained production inputs.
`WinRatePredictor.predict_marginal` additionally serves the pregame path: it
takes no build ids, enumerates train-supported build worlds from the
`app.ml.build_catalog` priors, and averages output probabilities (see
`HGNN_BUILD_INTENT.md` and the marginalisation record in `EXPERIMENTS.md`).
Accepted marginal *metrics* come from the cache-side harness
`python -m app.ml.marginal_eval`, which serves loadout/patch heads as trained.

```text
cache 1vX priors + support
-> posterior node features
-> champion/build identity embeddings
-> production Loadout head + patch-only Temporal head
-> frozen static/full-game/temporal sidecars into learned semantic MoE
-> blue/red team readout
-> per-seed final logit, mean over 3 seeds
-> affine calibration (scale, bias)
-> sigmoid = P(blue wins)
```

Direct 1v1/2vX champion matchup and synergy relationship integrations, and the
experimental player-prior arrays, have been removed from the model contract,
cache layout, priors, and predictor. Older local cache directories may still
contain ignored relationship or player `.npy` files, but v32 production
loading does not declare or consume them. Loadout and patch-only Temporal are
part of the default production model when the v32 cache provides
`loadout_features.npy` and `patch_features.npy`.

## Production Status

Hard acceptance remains overall raw test accuracy `>=60%` on the per-patch
test split. The current promoted production artifact does not meet that gate
yet, but it banks every validated draft-only lever.

Promoted artifact: `app/ml/data/hgnn_production_model.pt` — a calibrated
3-seed ensemble written by `app/ml/promote.py`. Each member is a
default-recipe v32 checkpoint (seeds 4/5/6: lr `3e-4`, batch `16384`,
`max_epochs=40`, `patience=5`, raw test-accuracy checkpointing). Promotion
averages the per-seed logits and fits an affine logit calibration on the train
split; the bias restores the blue-side prior that team-swap augmentation
suppresses (model mean `p≈0.493` vs true blue winrate `0.482`).

| Promoted ensemble | Test accuracy | Test NLL | logit scale | logit bias |
| --- | ---: | ---: | ---: | ---: |
| 3-seed logit-mean + affine calibration | **58.2601%** | **0.671053** | `1.1686` | `-0.0432` |
| Single-seed reference (mean of seeds 4/5/6) | `57.88%` | `0.6723` | — | — |

Gate reachability (2026-06-10, reaffirmed 2026-06-11): every draft-safe input
axis has been audited — context head saturated at the draft-time ceiling,
relationship features dead, recency/level dead, role experience marginal,
player-skill priors blocked by aggregate staleness and now excluded outright
by the draft-generic constraint, champion-strength / meta-drift features
bounded out by a leakage-free future-knowledge oracle (`<=0.005pp` ceiling),
and `(champion, position)` semantic identity profiles shown fully redundant
with champion identity by shuffled-profile and one-hot controls (see
`EXPERIMENTS.md` for each record). Remaining headroom most plausibly requires
new draft-generic information rather than residual heads on current features.

Loadout uses train-only, leave-one-out-adjusted historical priors over
summoner spell pairs, broad rune setup, full rune page, secondary rune pair, and
stat shards. Rune rows are joined through `puuid` only to align the selected
rune page; no player identity is emitted into `loadout_features.npy`. Patch
Temporal is restricted to season/patch blue-side drift only and does not
include champion-role patch deltas; the older broad `T+L` diagnostic washed
out loadout because it combined champion-role patch deltas with the patch
blue-side intercept in one shared residual head.

Under the current leakage policy, observed final build-value/profile residuals
remain diagnostic only unless a draft-safe source or RL search supplies the
build intent (see `HGNN_BUILD_INTENT.md`).

## Architecture

```mermaid
flowchart TD
    cache["v32 production cache"] --> ids["champion/build ids"]
    cache --> prior["1vX prior + support"]
    cache --> loadout["Loadout features<br/>spells + rune page + stat shards"]
    cache --> patch["Patch-only Temporal<br/>season/patch blue-side drift"]

    ids --> hgnn_inputs["build_hgnn_inputs"]
    prior --> hgnn_inputs
    loadout --> hgnn_inputs
    patch --> hgnn_inputs
    sidecars["static + full-game + temporal sidecars"] -->|semantic MoE default| hgnn_inputs

    hgnn_inputs --> tensor_batch["HGNN tensor batch"]
    tensor_batch --> nodes["Identity embeddings + 1vX posterior node features"]
    tensor_batch --> loadout_logit["production loadout_logit"]
    tensor_batch --> patch_logit["production patch_logit<br/>bounded max abs 0.15"]
    tensor_batch -.-> moe["Learned semantic MoE head<br/>sidecar factors + top-k experts"]
    nodes --> readout["Mean + attention team readout"]
    readout --> base["base_logit"]
    moe -.->|use_learned_semantic_moe=True| context_logit
    base --> final["final_logit = base_logit + loadout_logit + patch_logit + context_logit"]
    context_logit --> final
    loadout_logit --> final
    patch_logit --> final
    final --> ensemble["mean over 3 seed members<br/>scale * logit + bias"]
    ensemble --> prob["sigmoid = P(blue wins)"]
    prob --> metrics["accuracy / NLL metrics"]
```

## Identity Encoder Sidecars

Three standalone identity autoencoders produce latents consumed by the learned
semantic MoE. The sidecar artifact is one row per
`(champion, role, build)` identity; the static block is champion-level and is
joined/repeated onto those rows, while full-game and temporal latents are native
to the full identity grain.

The latents are **not** materialised per game-slot. The cache (`v32`) records the
artifact path/dims only; `app/ml/train.py` builds an on-device gather table
(`EncoderSidecarLookup.gather_tables`) and gathers `(batch, 10, dim)` blocks per
batch from `champion_id`/`build_id` — the static block is keyed by champion. This
collapses the sidecar cache from tens of GB to the few-MB frozen artifact. The
draft-time predictor already gathered the same way. Legacy caches that still hold
per-game sidecar arrays continue to load and are used directly.

| Sidecar | Encoder module | Maintained consumer |
| --- | --- | --- |
| Static | [classification/static_identity_encoder.py](../../classification/static_identity_encoder.py) | Learned semantic MoE |
| Full-game | [classification/full_game_encoder.py](../../classification/full_game_encoder.py) | Learned semantic MoE |
| Temporal | [classification/temporal_autoencoder.py](../../classification/temporal_autoencoder.py) | Learned semantic MoE |

`HGNNConfig.use_learned_semantic_moe=True` enables the learned mixture-of-experts
context path over the same required sidecar inputs plus the champion, role,
build, and fused identity embeddings. This is the only maintained semantic
sidecar architecture. It builds support/log-support sidecar tokens, derives own /
ally / enemy / extremity factors, routes each slot through top-k experts
(production default 32 of 128), support-gates
zero-initialised slot deltas, and adds `semantic_moe_logit` into `context_logit`.
Production training consumes `semantic_moe_regularization_loss`; the maintained
train metrics stay focused on accuracy and NLL.

When `use_semantic_group_features=True`, the learned MoE also receives the
compact semantic group feature tensor from `app/ml/semantic_group_features.py`.
The relationship head builds slot-level own / ally / enemy group summaries
including mean, sum, max, ally-vs-enemy differences, and own-by-team interaction
blocks. A zero-initialised MLP turns those relationship blocks into support-gated
slot deltas, so the production prior is unchanged at init while identities can
slowly learn how their own semantic groups react to every allied and enemy group
composition.

Serving rebuilds the same compact group tensor from smoothed train identity
metrics plus static champion HP/range lookups, so melee/ranged and natural
tankiness remain available without a large per-game cache array.

### Retired Surfaces

The old node-init sidecar MLP flags, the separate identity semantic context
head, the dense/sparse MoE dispatch flag, the warm-start/freeze fine-tune
machinery, the player-prior cache arrays and model paths, and the
context-examples / group-EB audit tooling were all removed from the maintained
workspace after their conclusions were recorded (the closed-lever findings
live in `EXPERIMENTS.md`; the MoE capacity decision is below). Checkpoint
loading filters removed legacy config/state keys so current artifacts load on
the leaner model.

### Retired Expert-Grid Ablation Outcomes

Temporary MoE expert-count / `top_k` runners were removed on 2026-06-07 after
their outcomes were captured here; the production recipe was fixed at 128
experts and `top_k=32`. The seed-4 sweeps varied only
`semantic_moe_num_experts` and `semantic_moe_top_k` on the old 80/10/10
protocol; primary context ranking was the flagged support-weighted mean
absolute gap from the (since retired) context-examples audit; lower is better.

| Variant | Experts | `top_k` | Active fraction | Flagged MAE | Flagged MSE | Validation accuracy | Test accuracy | Validation NLL | Test NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `128x32` | 128 | 32 | 0.250 | **1.7027 pp** | **4.8201 pp^2** | 57.8547% | 57.3433% | **0.6729** | **0.6759** |
| `32x16` | 32 | 16 | 0.500 | 1.7616 pp | 4.9456 pp^2 | 57.8715% | 57.3593% | 0.6729 | 0.6760 |
| `32x8` | 32 | 8 | 0.250 | 1.7653 pp | 4.9552 pp^2 | 57.8701% | **57.3970%** | 0.6729 | 0.6760 |
| `16x8` | 16 | 8 | 0.500 | 1.7890 pp | 5.0316 pp^2 | 57.8589% | 57.3712% | 0.6729 | 0.6760 |
| `64x32` | 64 | 32 | 0.500 | 1.9191 pp | 5.8366 pp^2 | 57.8575% | 57.3579% | 0.6730 | 0.6760 |
| `64x16` | 64 | 16 | 0.250 | 1.9395 pp | 5.9308 pp^2 | 57.8547% | 57.3433% | 0.6731 | 0.6761 |
| `128x16` | 128 | 16 | 0.125 | 1.9599 pp | 6.2078 pp^2 | 57.8155% | 57.3241% | 0.6733 | 0.6762 |
| `32x4` | 32 | 4 | 0.125 | 1.9669 pp | 5.7403 pp^2 | **57.8994%** | 57.3712% | 0.6730 | 0.6760 |
| `64x8` | 64 | 8 | 0.125 | 1.9822 pp | 6.1229 pp^2 | 57.8673% | 57.3579% | 0.6731 | 0.6760 |
| `8x2` control | 8 | 2 | 0.250 | 1.9989 pp | 6.2255 pp^2 | 57.8575% | 57.3489% | 0.6730 | 0.6760 |

Against the `8x2` in-sweep control, `128x32` reduced flagged context MAE by
14.8% and flagged context MSE by 22.6% with effectively flat accuracy and
slightly better NLL. The larger-capacity signal was promising but not
monotonic (`64x*` and `128x16` underperformed); a `256x64` follow-up was
slower without beating `128x32` and was abandoned.

### Semantic MoE Plan

```mermaid
flowchart TD
    support["identity_encoder_support"] --> gate["support confidence + log support"]
    sidecars["static + full-game + temporal latents<br/>[game, 10, dim]"] --> sidecar_token["MoE sidecar token<br/>latents + confidence + log support"]
    support --> sidecar_token
    sidecar_token --> sidecar_factor["sidecar factor MLP<br/>token dropout"]
    sidecar_factor --> moe_context["MoE own / ally / enemy / max factors"]
    identity["champion / role / build / fused identity"] --> moe_token["MoE semantic factor token"]
    moe_context --> moe_token
    gate --> moe_token
    moe_token --> factor["semantic factor MLP"]
    factor --> router["router top-k experts<br/>production 32 of 128"]
    factor --> experts["zero-initialised expert deltas"]
    router --> moe_slots["support-gated slot deltas"]
    experts --> moe_slots
    moe_slots --> context["context_logit = semantic_moe_logit<br/>mean blue - mean red"]

    production["production logit<br/>base + loadout + patch"] --> final["final_logit = production_logit + context_logit"]
    context --> final
```

## Maintained Surfaces

| File | Purpose |
| --- | --- |
| [../hgnn_model.py](../hgnn_model.py) | HGNN model, ensemble wrapper, input builder, swap invariants, and maintained residual/semantic MoE heads. |
| [../encoder_sidecar.py](../encoder_sidecar.py) | Identity-encoder latent loading, per-game lookup, and dedup gather tables. |
| [../loadout_patch_features.py](../loadout_patch_features.py) | Production train-only loadout priors and patch-only temporal feature extraction. |
| [../build_dataset.py](../build_dataset.py) | Cache builder for 1vX identity inputs and sidecar metadata. |
| [../dataset.py](../dataset.py) | Cache loader and split dataclass. |
| [../train.py](../train.py) | Production training, test-split selection, and accuracy/NLL metrics. |
| [../promote.py](../promote.py) | Seed-checkpoint scoring, affine calibration fit, and production ensemble artifact writer. |
| [../predictor.py](../predictor.py) | Draft-time runtime bridge. |

## Throughput Default

Use `--batch-size 16384` for the current 128x32 HGNN recipe unless the
experiment is explicitly a throughput/allocator sweep. Batch
size is architecture-dependent: if parameter count or activation footprint
increases, retune downward by measured samples/s; if it decreases, retune upward
only after a fresh sweep. The 2026-06-10 local RTX 5070 Ti sweep found batch
`16384` as the fastest stable point at `51,505` team-swap-augmented samples/s
(`25,752` raw rows/s). Larger tested batches hit the allocator/throughput cliff:

Local experiment hardware is an NVIDIA GeForce RTX 5070 Ti with `16,303 MiB`
VRAM. Production-scale runs should use `--raw-tensor-cache-device cpu` so the
GPU holds the model and active minibatch rather than the full raw split cache.

| Batch size | Augmented samples/s | Raw rows/s |
| ---: | ---: | ---: |
| `12288` | `49,182` | `24,591` |
| `16384` | **`51,505`** | **`25,752`** |
| `20480` | `16,020` | `8,010` |
| `24576` | `5,126` | `2,563` |
| `28672` | `4,708` | `2,354` |

## Active Defaults

| Area | Default |
| --- | --- |
| Checkpoint selection | raw test accuracy (test is the model-selection split) |
| Training batch size / throughput | `16384`; `51,505` augmented samples/s on the 2026-06-10 local RTX 5070 Ti sweep for the current 128x32 recipe. |
| Learning rate / patience / weight decay | `3e-4` / `5` / `0.0` |
| Raw tensor cache device | `cpu`; model-device caching is an explicit throughput sweep option. |
| Test evaluation | Always on: test is tensor-cached, evaluated every epoch, and written to metrics with accuracy/NLL and `selection_split: "test"`. |
| Production artifact overwrite | Refused by default; `--allow-production-artifact-overwrite` is required for `train.py`. |
| Production promotion | `python -m app.ml.promote --checkpoints <3 seed checkpoints>`; logit-mean + train-fitted affine calibration. |
| Direct 1v1/2vX integrations, player priors | Removed from the model, cache, priors, and predictor. |
| Loadout head | Offline-training head with v32 cache metadata and `loadout_features.npy`; the default production serving path rejects checkpoints that require it. |
| Patch-only Temporal head | Offline-training head with v32 cache metadata and `patch_features.npy`; season/patch blue-side drift only; the default production serving path rejects checkpoints that require it. |
| Learned semantic MoE head over all three identity sidecars | Enabled by default. |
| Semantic group features and relationship head | Enabled by default for the learned semantic MoE. |

Invalid training config combinations fail early in `app/ml/train.py`. Under
the per-patch protocol the test split drives checkpoint selection and
accuracy/NLL reporting; it is a selection split, not a final untouched holdout.
