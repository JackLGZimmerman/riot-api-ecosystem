# HGNN Rolled Split Handoff Prompt

Use this prompt with the attachment manifest below. Keep the review focused on
testing the current next data direction: refreshed chronological splits and
split-scoped artifacts. Do not start another semantic capacity, learning-rate,
cap, calibration-weight, or loss sweep unless the refreshed data protocol clears
the existing NLL gates.

## Attachment Manifest

Attach these primary documentation files:

- `app/ml/documentation/HGNN_CURRENT.md`
- `app/ml/documentation/EXPERIMENTS.md`
- `app/ml/documentation/README.md`
- `app/ml/documentation/HGNN_GROUP_CONTEXT_AUDIT.md`
- `app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md`
- `app/ml/documentation/HGNN_CENTRAL_BAND_REVIEW.md` (historical methodology only)
- `database/clickhouse/commands.md`
- `app/ml/data/metrics_latest.json`

Attach these pipeline and model files for the rolled split rebuild:

- `database/clickhouse/schema/5900_ml_game_split_build.sql`
- `database/clickhouse/schema/6900_ml_game_player_pivot_build.sql`
- `app/ml/build_dataset.py`
- `app/ml/dataset.py`
- `app/ml/cache_layout.py`
- `app/ml/loadout_patch_features.py`
- `app/ml/build_encoder_sidecar.py`
- `app/ml/encoder_sidecar.py`
- `app/ml/train.py`
- `app/ml/config.py`
- `app/ml/hgnn_model.py`
- `app/ml/semantic_group_features.py`
- `app/ml/context_audit_specs.py`
- `app/ml/context_examples_audit.py`
- `app/ml/group_context_audit.py`

Attach these classification rebuild files when sidecar or semantic-source
artifacts are regenerated:

- `app/classification/documentation/README.md`
- `app/classification/documentation/AUTOENCODER_README.md`
- `app/classification/documentation/ENCODER_METRICS.md`
- `app/classification/embeddings/build_tables.py`
- `app/classification/embeddings/config.py`
- `app/classification/embeddings/context_features.py`
- `app/classification/embeddings/load.py`
- `app/classification/embeddings/matrices.py`
- `app/classification/embeddings/registry.py`

Attach these tests if proposing code changes:

- `tests/ml/test_dataset.py`
- `tests/ml/test_encoder_sidecar.py`
- `tests/ml/test_train_sidecar_gather.py`
- `tests/ml/test_train_defaults.py`
- `tests/ml/test_train_calibration.py`
- `tests/ml/test_semantic_group_features.py`

## Prompt

You are reviewing the HGNN win-rate model in `app/ml`. The goal remains at least
`60%` raw validation accuracy and `60%` raw test accuracy, with NLL moving in the
same direction. Accuracy-only gains are not enough; if NLL is static, reject the
branch.

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
  maintained v30 path does not consume them.

The 2026-06-10 ceiling work settled the previous semantic-target question:
identity-derived surfaces and classification sidecar information did not become
stable enough row-level boundary direction under the frozen Apr-22 train
boundary. The remaining headroom is data freshness. Same-patch history carried
gate-level signal, while one-patch-stale cohorts stayed near the historical
`~0.0015` central NLL plateau.

The production-true rolled split protocol has now been run and rejected for
promotion under the pre-registered validation gates. Keep the plan below as the
historical execution recipe for future data-refresh work:

1. Refresh ClickHouse filtered tables from the latest ingested season 16 data.
2. Roll the chronological `ml_game_split` boundary forward so train contains
   strictly earlier rows available at the model refresh point. The split may cut
   inside a patch; leakage safety comes from row chronology and train-row-scoped
   artifacts, not whole-patch ownership.
3. Assign validation and test to later chronological windows only; do not select
   checkpoint, teacher, cadence, or thresholds on test.
4. Rebuild split-scoped artifacts: ML pivot, train-scoped priors/dictionaries,
   loadout/patch features, sidecar artifacts, semantic context tables, and the
   v30 cache.
5. Retrain the production recipe on the rolled cache using at least seed `4`
   plus one additional seed before any promotion claim.
6. Evaluate unchanged gates on the rolled validation/test windows.

Promotion gates:

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

Required review output:

1. Confirm the refreshed split ranges by season, patch, timestamp, and row count.
2. Confirm every train-scoped artifact was rebuilt after the split change.
3. Compare rolled-split production runs against the no-group ablation and the
   previous frozen-boundary production checkpoint.
4. Report central-band accuracy and NLL lift for `[0.45, 0.55]` and diagnostic
   `[0.475, 0.525]`.
5. Re-run the group EB audit and the high-support context examples as guardrails.
6. Decide whether the refreshed protocol clears promotion gates. If not, reject
   the branch before proposing architecture changes.
