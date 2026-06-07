# HGNN Group Context Audit

Updated: 2026-06-07.

Companion to `HGNN_CONTEXT_EXAMPLES_AUDIT.md`. That audit measures the model gap
against each split's **raw, champion-specific** empirical win rate, whose per-bin
sampling variance `p(1-p)/n` is irreducible (median bin n~500 -> floor ~10.5 pp^2).
No architecture can drive that metric to 0; most of the apparent champion-bin gap is
sampling noise in the targets, not model error.

This audit fixes the **measurement**, not just the model:

1. **Group pooling** - focus is a deterministic build/role group (e.g. all frontline
   tanks, all AP casters, all marksmen), not a single champion. Median bin n jumps
   from ~500 to ~47,000. Groups use the same build vocabulary the relationship head
   consumes; no hand-authored champion archetypes.
2. **Empirical-Bayes target** - each bin's win rate is shrunk toward the n-weighted
   row mean (Gaussian EB, tau^2 by method of moments). The target is the best
   estimate of the *true* WR, not the noisy sample mean.
3. **Debiased Gap MSE** - subtract the EB target's residual variance from each
   squared gap. The result, `systematic_gap_mse`, estimates genuine model error and
   *can* approach 0.

References: multicalibration (Hebert-Johnson et al. 2018), debiased calibration
error (Kumar et al. 2019; Roelofs et al. 2022), James-Stein / empirical Bayes
(Efron & Morris 1975).

## Production Model

The promoted production checkpoint is `app/ml/data/hgnn_production_model.pt`.

This is an architecture change, not just an inclusion test for extra encoder inputs.
The model uses:

- `semantic_moe_architecture = convex_encoder_mix`
- `semantic_moe_num_experts = 128`
- `semantic_moe_top_k = 32`
- `use_learned_semantic_moe = true`
- `use_semantic_group_features = true`
- compact encoder sidecar: `semantic_identity_sidecar_compact.npz`
- sidecar dimensions: static 16, full-game 64, temporal 64
- node-init sidecar flags disabled: `use_identity_static_sidecar = false`,
  `use_identity_full_game_sidecar = false`, `use_identity_temporal_sidecar = false`

So the conclusion is not "remove static encoding." Static information such as
melee/range tendency and natural tankiness is still relevant; production keeps it,
but routes it through the learned semantic encoder mixer instead of injecting it
directly into node initialization.

## Production Result (16 groups, 67 populated bins)

Prediction cache:

```text
app/ml/data/audit_focus_side_probability.npy
```

JSON output:

```text
app/ml/data/group_context_audit_production.json
```

`systematic_gap_mse` is in pp^2 and lower is better. On held-out splits, the raw
target floor is ~0.18 pp^2 and the EB target floor is ~0.11 pp^2. Values below
are from `app/ml/data/metrics_latest.json` for the promoted 128x32 checkpoint.

| Split | bins | median n | min n | raw MSE | raw floor | EB MSE | EB floor | systematic | clipped | mean abs EB gap | max abs EB gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 67 | 383,583 | 9,383 | 0.02 | 0.02 | 0.02 | 0.02 | 0.00 | 0.01 | 0.12 | 0.39 |
| Validation | 67 | 47,403 | 1,154 | 0.29 | 0.18 | 0.18 | 0.11 | 0.07 | 0.12 | 0.30 | 1.67 |
| Test | 67 | 45,828 | 1,193 | 0.33 | 0.18 | 0.40 | 0.11 | 0.29 | 0.33 | 0.41 | 3.20 |

Held-out production metrics from `app/ml/data/metrics_latest.json`:

| Split | accuracy | AUC | NLL | ECE |
|---|---:|---:|---:|---:|
| Validation | 0.5789 | 0.6091 | 0.6732 | 0.0320 |
| Test | 0.5730 | 0.6029 | 0.6762 | 0.0341 |

Train->test EB target movement: MSE 0.29 pp^2, mean abs 0.37 pp, max abs 2.14 pp
over the same 67 bins.

## Key Findings

- The promoted `convex_encoder_mix` checkpoint clears the old calibration-weight
  concern. Validation systematic gap is **0.07 pp^2**, below the EB target floor
  of **0.11 pp^2** and far below the retired calibration-weight comparison.
- Test systematic gap is higher at **0.23 pp^2**, but still in the range implied by
  train->test target movement (**0.29 pp^2**). The remaining issue looks more like
  temporal/group drift and tail behavior than a broad validation calibration miss.
- The largest validation residuals are concentrated in a few interpretable static
  contexts: armor/frontline tanks into high physical damage, marksmen into low enemy
  range count, AP casters into enemy magic profile, and MR tanks into magic bins.
- The raw 60% accuracy target is still not met (`57.89%` val, `57.30%` test), but
  group-level semantic calibration is no longer the obvious bottleneck.
- We did not prove that separate, per-encoder MoEs are better. The production win is
  specifically the learned convex mixer across static/full-game/temporal encoder
  views plus semantic group features.

## Largest Validation Residuals

Rows are sorted by debiased `systematic_gap_mse`; `gap` is HGNN minus EB target.

| systematic | group | bin | n | empirical | EB target | HGNN | gap |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1.49 | Armor tanks vs enemy physical | >= 0.557 | 28,805 | 52.0 | 51.9 | 53.1 | +1.3 |
| 1.33 | Marksmen BOTTOM vs enemy range count | <= 1 | 18,595 | 50.5 | 50.1 | 51.3 | +1.2 |
| 1.07 | Frontline tanks vs enemy physical | >= 0.557 | 62,543 | 50.7 | 50.7 | 51.7 | +1.1 |
| 0.79 | AP casters vs enemy magic | <= 0.373 | 79,638 | 51.3 | 51.3 | 52.2 | +0.9 |
| 0.45 | Armor tanks vs enemy physical | <= 0.387 | 13,302 | 48.8 | 48.9 | 48.1 | -0.8 |
| 0.37 | Frontline tanks vs enemy physical | <= 0.387 | 54,951 | 49.4 | 49.4 | 48.7 | -0.6 |
| 0.25 | AP casters vs enemy magic | 0.373-0.423 | 76,463 | 49.9 | 49.9 | 50.5 | +0.5 |
| 0.21 | MR tanks vs enemy magic | <= 0.373 | 2,376 | 47.3 | 48.1 | 49.1 | +0.9 |
| 0.20 | AP casters vs enemy high-HP count | >= 3 | 35,567 | 51.6 | 51.4 | 52.0 | +0.5 |
| 0.19 | Enchanters UTILITY with skirmish allies | 1 | 26,062 | 52.1 | 51.8 | 51.3 | -0.5 |

## Implications

- The old calibration-weight revert recommendation is obsolete. Production is
  now the promoted `convex_encoder_mix` checkpoint.
- Static context should stay in the system, but as a learned semantic view rather
  than a direct node-init shortcut.
- The group EB audit should remain a promotion guard alongside accuracy, NLL, AUC,
  and ECE. It caught the noisy calibration-weight failure mode and now gives a clean
  read on whether semantic architecture changes generalize.
- Next semantic work should focus on test-split residual tails and raw ranking /
  accuracy lift. A separate MoE per encoder remains an untested architecture
  hypothesis; this audit only validates the current convex encoder mixer.

## Reproduction

Regenerate the prediction cache from the promoted production checkpoint if needed:

```bash
uv run python -m app.ml.context_examples_audit \
  --context-cache-dir app/ml/data/cache \
  --model-cache-dir app/ml/data/cache \
  --model-path app/ml/data/hgnn_production_model.pt \
  --encoder-sidecar-path app/ml/data/experiments/semantic_identity_sidecar_compact.npz \
  --prediction-cache app/ml/data/audit_focus_side_probability.npy \
  --audit-split val \
  --output app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md \
  --refresh-predictions
```

Run the group EB audit:

```bash
uv run python -m app.ml.group_context_audit \
  --context-cache-dir app/ml/data/cache \
  --prediction-cache app/ml/data/audit_focus_side_probability.npy \
  --per-row \
  --json-output app/ml/data/group_context_audit_production.json
```

Module: `app/ml/group_context_audit.py`.
