# HGNN Semantic Grouping Handoff Prompt

Use this prompt with the attachment manifest below. Keep the review focused on
data/target construction and boundary-NLL movement, not another round of
capacity, learning-rate, cap, or calibration-weight sweeps.

## Attachment Manifest

Attach these files first:

- `app/ml/documentation/HGNN_CURRENT.md`
- `app/ml/documentation/EXPERIMENTS.md`
- `app/ml/documentation/HGNN_GROUP_CONTEXT_AUDIT.md`
- `app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md`
- `app/ml/documentation/HGNN_CENTRAL_BAND_REVIEW.md`
- `app/ml/documentation/README.md`
- `app/ml/data/metrics_latest.json`

Attach these code files for implementation context:

- `app/ml/hgnn_model.py`
- `app/ml/train.py`
- `app/ml/config.py`
- `app/ml/semantic_group_features.py`
- `app/ml/context_audit_specs.py`
- `app/ml/context_examples_audit.py`
- `app/ml/group_context_audit.py`
- `app/ml/encoder_sidecar.py`
- `app/ml/build_encoder_sidecar.py`
- `app/classification/static_identity_encoder.py`
- `app/classification/full_game_encoder.py`
- `app/classification/temporal_autoencoder.py`

Attach these tests if proposing code changes:

- `tests/ml/test_train_defaults.py`
- `tests/ml/test_train_calibration.py`
- `tests/ml/test_encoder_sidecar.py`
- `tests/ml/test_semantic_group_features.py`
- `tests/ml/test_train_sidecar_gather.py`

## Prompt

You are reviewing the HGNN win-rate model in `app/ml`. The over-arching goal is
to reach at least `60%` raw validation accuracy and `60%` raw test accuracy, with
NLL moving in the same direction. Accuracy-only gains are not enough; if NLL is
static, move on from that branch.

The specific semantic-grouping goal is to make semantic groups causally drive
near-boundary decisions. In the central band, the model should improve decisions
because the semantic group path is present, and lose that advantage when the
semantic group path is removed. The main diagnostic band is
`p_no_group in [0.45, 0.55]`; also report `[0.475, 0.525]`.

Current production state:

- Production model: `1vX + champion/build + Loadout + patch Temporal +
  all-encoder semantic MoE`.
- Semantic path: `convex_encoder_mix`, 128 experts, `top_k=32`, all three frozen
  identity sidecars, and compact semantic group features.
- Production checkpoint: `app/ml/data/hgnn_production_model.pt`.
- Current held-out production metrics are about `57.89%` validation accuracy,
  `57.38%` test accuracy, `0.672978` validation NLL, and `0.675965` test NLL.
- Direct 1v1/2vX relationship integrations are not part of the production model
  contract. Older local caches may contain ignored relationship arrays, but the
  maintained v29 path does not consume them.

The problem:

Semantic groups are useful for evaluation, but weak for final boundary
semantics. The group examples expose interpretable champion/build/context
failures. The group EB audit gives a lower-noise calibration guardrail. However,
attempts to turn those grouped contexts into per-game boundary decisions have
mostly produced threshold accuracy movement without enough NLL movement.

Observed failure pattern:

- Central-band accuracy often moved by roughly `+0.5pp` to `+0.8pp`.
- Central-band NLL stayed around `+0.001` to `+0.0015`, below the required
  `+0.003` validation and `+0.002` test central-band NLL lifts.
- Best fixed-feature relationship ceiling record: validation central
  `+0.779pp` to `53.29%` with NLL lift `0.001409`; test central `+0.767pp` to
  `52.75%` with NLL lift `0.001380`; global validation `+0.293pp` to `58.00%`;
  global test `+0.286pp` to `57.38%`.
- A label-aligned oracle at similar magnitude can produce roughly `+0.0099`
  central NLL lift, so the NLL gate is attainable in principle. The current
  target direction is the bottleneck.

Why the current architecture does not satisfy the goal:

- It contains semantic groups, but it does not make them reliable boundary
  decision variables.
- The current semantic group tensor is compact and interpretable, but it is a
  coarse summary. It can describe context existence and aggregate effects better
  than it can decide the exact side of a near-50/50 game.
- The group-relationship head converts own/ally/enemy group summaries into
  support-gated slot deltas, but the supervision has mostly been aggregate
  audit-bin calibration or EB residual matching. That is too blunt for per-row
  central-band probability corrections.
- The frozen sidecar encoders were built around generic identity/reconstruction
  objectives, not cross-fit prediction of semantic residual direction. They may
  preserve broad identity context while losing the row-level signal needed for
  boundary flips.
- The current loss can reduce audit gaps or move thresholds without learning a
  stable held-out probability correction, which is why NLL remains nearly flat.
- Direct matchup/relationship priors and completed-game build-profile signals
  are intentionally excluded from production unless a split-safe, pregame-safe
  source exists.

What has already been tried:

- Production semantic MoE promotion from smaller expert grids to 128x32
  `convex_encoder_mix`.
- Focus-slot semantic audits instead of repeated match-level probabilities.
- Group EB audit to reduce noisy champion-bin target variance.
- Calibration objectives using champion raw targets, context EB, group EB, and
  group+context EB surfaces.
- Absolute and residual-style semantic calibration losses, including
  uncertainty-aware variants, support-family bin weighting, train-core group
  surfaces, and group-spec holdout diagnostics.
- Gradient diagnostics for semantic MoE and group-relationship parameters.
- Gate/confidence/isolation-style experiments around the semantic relationship
  path.
- Sidecar/encoder-input checks, including current compact encoder sidecar and
  context/semantic-target variants.
- Fixed-feature/replay probes and relationship-surface ceiling probes.
- Harm penalties, amplitude/cap/scale sweeps, direct-head smokes, and
  NLL-focused utility probes.

Do not repeat those as hyperparameter sweeps unless you first identify a new
target/data surface whose fixed-feature ceiling clears the NLL gate.

Required evaluation gates for a promotion candidate:

- Global validation and test accuracy must move toward the hard `60% / 60%`
  goal, with global NLL improving or not regressing.
- Boundary causal lift: on `p_no_group in [0.45, 0.55]`, full model must beat
  no-group by at least `+0.50pp` validation accuracy and `+0.30pp` test accuracy.
- Boundary NLL lift: at least `+0.003` validation and `+0.002` test.
- Directional semantic use: support-weighted sign agreement between semantic
  movement and train-only residual direction at least `55%` validation and
  non-regressing on test.
- Audit sanity: high-support semantic group bins should target
  `max_abs_gap <= 3.0pp` validation and `<= 3.5pp` test, with p95 gaps reported.
- Global guardrails: validation/test NLL must not worsen by more than `0.0002`;
  accuracy/AUC must not drop by more than `0.05pp`.

Your task:

1. Explain the most likely reason semantic grouping is failing to produce strong
   boundary semantics.
2. Identify the next data/target construction that should be tested before any
   architecture changes.
3. Define a split-safe fixed-feature ceiling or replay experiment that can prove
   the target surface has enough central-band NLL signal before wiring it into
   HGNN training.
4. If the ceiling clears, propose the smallest production-aligned architecture
   change needed to integrate the target. If it does not clear, reject the branch.
5. Keep test data untouched for selection. Use train-only, leave-one-out, or
   cross-fit targets where labels are involved.

Prefer solutions that add better information or better supervision over bigger
heads. The likely breakthrough is how the data is provided, not parameter tuning.
