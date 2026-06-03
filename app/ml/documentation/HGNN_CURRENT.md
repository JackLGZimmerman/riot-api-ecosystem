# HGNN Current State

Last updated: 2026-06-03.

## Production Path

Default training and serving use the 1vX player prior, champion/build identity
embeddings, and team-swap augmentation. Legacy classification-derived semantic,
profile, and context inputs are no longer part of `build_hgnn_inputs()` or
`HGNNWinModel.forward()`.

Identity signal beyond the champion/build embeddings is now carried by the
disabled-by-default identity-encoder sidecars (static / full-game / temporal),
plus an opt-in semantic-context head over those same frozen latent blocks. See
[Identity Encoder Sidecars](#identity-encoder-sidecars).

```text
cache 1vX priors + support
-> posterior node features
-> champion/build identity embeddings
-> optional frozen identity sidecars / semantic context head
-> blue/red team readout
-> final logit
-> sigmoid = P(blue wins)
```

Direct 1v1 and 2vX integrations are disabled by default. The loader still
accepts existing relationship arrays, and `build_hgnn_inputs()` creates neutral
internal placeholders when they are absent. Older model artifacts without an
explicit `use_relationship_integrations` flag load with relationship
integrations preserved.

## Architecture

```mermaid
flowchart TD
    ids["champion/build ids"] --> hgnn_inputs["build_hgnn_inputs"]
    prior["1vX prior + support"] --> hgnn_inputs
    rel["1v1/2vX arrays"] -.->|opt-in research| hgnn_inputs
    sidecars["static + full-game + temporal sidecars"] -.->|opt-in| hgnn_inputs

    hgnn_inputs --> tensor_batch["HGNN tensor batch"]
    tensor_batch --> nodes["Identity embeddings + 1vX posterior node features"]
    tensor_batch -.-> context["Own / ally / enemy semantic context head"]
    nodes --> readout["Mean + attention team readout"]
    readout --> final["final_logit"]
    rel -.->|use_relationship_integrations=True| final
    context -.->|use_identity_semantic_context_head=True| final
    final --> prob["sigmoid = P(blue wins)"]
    final -.-> diagnostics["Calibration and support metrics"]
```

## Identity Encoder Sidecars

Three standalone identity autoencoders produce latents that can be injected as
node-level sidecars in `HGNNWinModel`. The sidecar artifact is one row per
`(champion, role, build)` identity; the static block is champion-level and is
joined/repeated onto those rows, while full-game and temporal latents are native
to the full identity grain:

| Sidecar | Encoder module | Config flag (default `False`) |
| --- | --- | --- |
| Static | [classification/static_identity_encoder.py](../../classification/static_identity_encoder.py) | `use_identity_static_sidecar` |
| Full-game | [classification/full_game_encoder.py](../../classification/full_game_encoder.py) | `use_identity_full_game_sidecar` |
| Temporal | [classification/temporal_autoencoder.py](../../classification/temporal_autoencoder.py) | `use_identity_temporal_sidecar` |

The sidecar MLP is support-gated and zero-initialised, so an unwired or
low-support latent is a no-op on the production node init. All three are off by
default; the ablation surface lives in
[../experiments/context_ablation.py](../experiments/context_ablation.py).

`HGNNConfig.use_identity_semantic_context_head=True` enables a separate
zero-initialised context logit over the frozen static, full-game, and temporal
blocks. For each slot it projects the concatenated latents, builds support-
weighted summaries of the other four allies and five enemies, scores the shared
`own / ally / enemy` interaction, and adds `context_logit` to `base_logit`.
The model returns all three columns: `base_logit`, `context_logit`, and
`final_logit`.

### Semantic Context Plan

```mermaid
flowchart TD
    sidecars["static + full-game + temporal latents<br/>[game, 10, dim]"] --> proj["shared latent projection<br/>LayerNorm -> 96-d semantic vector"]
    support["identity_encoder_support"] --> gate["support confidence + log support"]

    proj --> own["focus identity z_i"]
    proj --> allies["same-team latent summary<br/>excluding focus slot"]
    proj --> enemies["enemy-team latent summary"]

    gate --> allies
    gate --> enemies
    gate --> score

    own --> score["shared context interaction<br/>own / ally / enemy"]
    allies --> score
    enemies --> score

    score --> blue["blue slot context scores"]
    score --> red["red slot context scores"]
    blue --> diff["mean blue - mean red"]
    red --> diff
    diff --> context["context_logit"]
    base["existing HGNN base_logit"] --> final["final_logit = base_logit + context_logit"]
    context --> final

    final --> audit["HGNN_CONTEXT_EXAMPLES_AUDIT<br/>empirical WR vs predicted WR by threshold"]
```

## Maintained Surfaces

| File | Purpose |
| --- | --- |
| [../hgnn_model.py](../hgnn_model.py) | HGNN model, input builder, swap invariants, and relationship gate. |
| [../encoder_sidecar.py](../encoder_sidecar.py) | Identity-encoder latent loading and sidecar tensor assembly. |
| [../build_dataset.py](../build_dataset.py) | Cache builder for 1vX identity inputs and optional relationship arrays. |
| [../dataset.py](../dataset.py) | Cache loader and split dataclass. |
| [../train.py](../train.py) | Production training and validation/report-only calibration diagnostics. |
| [../predictor.py](../predictor.py) | Draft-time runtime bridge. |

## Active Defaults

| Area | Default |
| --- | --- |
| Checkpoint metric | `val_threshold_accuracy` |
| Report-only temperature scaling | Fit on validation logits only; never changes served probabilities. |
| Direct 1v1/2vX integrations | Disabled by default. |
| Relationship loader arrays | Retained for explicit research use. |
| Identity-encoder sidecars (static/full-game/temporal) | Disabled by default. |
| Semantic context head over all three identity sidecars | Disabled by default. |

Invalid calibration and training config combinations fail early in
`app/ml/train.py`. Test labels are not used for threshold selection,
temperature fitting, checkpoint selection, or model selection.
