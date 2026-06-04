# HGNN Group Context Audit

Updated: 2026-06-04.

Companion to `HGNN_CONTEXT_EXAMPLES_AUDIT.md`. That audit measures the model gap
against each split's **raw, champion-specific** empirical win rate, whose per-bin
sampling variance `p(1-p)/n` is irreducible (median bin n~500 -> floor ~10.5 pp^2).
No architecture can drive that metric to 0; ~89% of its 13.3 pp^2 val Gap MSE is
sampling noise in the targets, not model error.

This audit fixes the **measurement**, not just the model:

1. **Group pooling** — focus is a deterministic build/role group (e.g. all frontline
   tanks, all AP casters, all marksmen), not a single champion. Median bin n jumps
   from ~500 to ~47,000. Groups use the same build vocabulary the relationship head
   consumes; no hand-authored champion archetypes.
2. **Empirical-Bayes target** — each bin's win rate is shrunk toward the n-weighted
   row mean (Gaussian EB, tau^2 by method of moments). The target is the best
   estimate of the *true* WR, not the noisy sample mean.
3. **Debiased Gap MSE** — subtract the EB target's residual variance from each
   squared gap. The result, `systematic_gap_mse`, estimates genuine model error and
   *can* approach 0.

References: multicalibration (Hebert-Johnson et al. 2018), debiased calibration
error (Kumar et al. 2019; Roelofs et al. 2022), James-Stein / empirical Bayes
(Efron & Morris 1975).

## Result (16 groups, 67 populated bins)

`systematic_gap_mse` (pp^2), lower is better; raw floor ~0.18, EB floor ~0.11.

| Model | calib weight | train | val | test | val AUC | test AUC |
|---|---:|---:|---:|---:|---:|---:|
| `semantic_focus_reference_w100` | 100 | 0.80 | **0.57** | **0.41** | 0.5975 | 0.5920 |
| `semantic_focus_reference_w300_cont6` | 300 | 0.65 | 0.70 | 0.77 | 0.5903 | 0.5859 |
| `semantic_focus_reference_w3000_cont6` (checked-in) | 3000 | 0.73 | 0.78 | 0.86 | 0.5827 | 0.5790 |
| `semantic_focus_calibrated_w100` | 100 | 1.56 | 1.14 | 0.75 | 0.6009 | 0.5950 |

Key findings:

- **The floor collapses as predicted.** Group raw Gap MSE on val is **0.88 pp^2**
  (vs 13.3 champion-specific) against a **0.18 pp^2** floor. The metric is now
  dominated by model error, not noise, so it is meaningful to drive toward 0.
- **Generalization gap is tiny at the group level.** For `w100`, train/val/test
  systematic are 0.80 / 0.57 / 0.41 — the model relates identities to group
  compositions consistently across unseen games. The "huge val gap" was a
  small-n artifact of champion-specific bins.
- **train->test drift is negligible** (0.29 pp^2 EB-target movement). Distribution
  shift across the time-ordered splits is not the bottleneck.
- **Calibration weight overfits, confirmed on a clean metric.** w100 -> w300 ->
  w3000 monotonically worsens val/test systematic AND AUC. The checked-in w3000 is
  strictly dominated by w100. The calibration loss uses champion-specific raw train
  targets (noisy); cranking it fits that noise.
- **The genuine residual is effect-shrinkage.** The model compresses contextual
  effects toward 50%: it under-predicts extreme/tail bins and over-predicts near
  baseline. At the group level the targets are not noisy, so this shrinkage leaves
  ~0.4-0.5 pp^2 of real, recoverable signal (w100 val 0.57 vs floor 0.11).

## Implications

- Revert the production checkpoint from `w3000` to `w100` (better on every
  generalizing metric).
- The recoverable target is the ~0.4-0.5 pp^2 systematic shrinkage, measured by
  group `systematic_gap_mse` (floor ~0.11). Two levers, both now measurable on a
  clean metric:
  - Objective: calibrate against **group EB targets** instead of champion raw
    targets (removes the overfitting source).
  - Architecture: a low-rank bilinear identity x group-composition head
    (`Δlogit = z_i^T W g(context)`) to express larger per-identity contextual
    corrections with a generalizing inductive bias.

## Reproduction

```bash
uv run python -m app.ml.group_context_audit --per-row \
  --prediction-cache app/ml/data/experiments/<run>/audit_focus_side_probability.npy
```

Predictions are reused from the existing per-slot audit cache; no model re-run is
needed for analysis. Module: `app/ml/group_context_audit.py`.
