# HGNN Context Win-Rate Validation

Generated: 2026-06-01. Updated: 2026-06-02 for the identity-conditioned head.

Reproduce:

```bash
python -m app.ml.context_wr_validation --model-path app/ml/data/experiments/identity_conditioned/cond_raw.pt
```

This validation asks whether the context residual predicts the win-rate that
draft-time context can explain after the base model has already used champion,
role, build, `1vX`, `1v1`, and `2vX` priors.

## Method

`base = final_logit - context_logit` is exact because both context heads are
additive residuals. The validation compares:

| Measurement | Meaning |
| --- | --- |
| `base` | Model with the context residual removed. |
| `head` | Full model with the actual context residual. |
| `ceiling linear` | Independent linear extractor over the frozen base offset. |
| `ceiling MLP` | Independent 64-wide MLP extractor over the frozen base offset. |

The ceiling extractors use the 24-dim interpretable descriptor, explicit
products, per-axis max/variance, and the low-rank tail. This is a fair ceiling
for the shared head. The identity-conditioned head uses the wider 62-dim raw
atlas, so it can legitimately exceed that descriptor-only ceiling.

## Ceiling Result

| Split | Base AUC | Head AUC | Ceiling linear AUC | Ceiling MLP AUC | Base NLL | Head NLL | Ceiling MLP NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.5921 | 0.6054 | 0.5978 | 0.5971 | 0.6806 | 0.6741 | 0.6774 |
| val | 0.5905 | 0.6027 | 0.5964 | 0.5962 | 0.6804 | 0.6746 | 0.6761 |
| test | 0.5855 | 0.5979 | 0.5920 | 0.5911 | 0.6814 | 0.6762 | 0.6776 |

The identity-conditioned head adds test AUC `+0.0124` over `base` and beats the
24-dim descriptor ceiling (`0.5979` vs `0.5920` linear / `0.5911` MLP). That is
the key change from the shared head: the shared head matched the 24-dim
descriptor ceiling, while the raw conditioned head extracts additional
draft-safe signal.

`base` AUC is lower than the shared model's context-free residual because the
conditioned backbone is trained jointly with a stronger context term and leans
on it more. It is not a separately optimized no-context model.

## Context-Score Calibration

Games are binned by a standalone context score. Swing is decile 10 minus decile
1. Errors are `Emp WR - actual model WR`.

| Split | Emp swing | Actual model swing | Emp - actual swing | Mean abs error | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | +22.55 | +21.67 | +0.88 | 0.52 | 1.01 |
| val | +22.24 | +22.29 | -0.04 | 2.78 | 3.69 |
| test | +20.82 | +21.95 | -1.14 | 2.43 | 3.00 |

The model slightly over-swings the test context axis (`+21.95` vs empirical
`+20.82`) and is high by about `2.43` points on average across deciles. The
remaining global issue is level calibration, not missing context amplitude.

Test decile sample:

| Decile | Emp WR | Actual model WR | Emp - actual | Base WR | Emp - base | Actual - base |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | 36.22 | 37.83 | -1.61 | 44.55 | -8.33 | -6.72 |
| D5 | 46.32 | 48.23 | -1.91 | 48.79 | -2.47 | -0.55 |
| D10 | 57.04 | 59.78 | -2.74 | 53.34 | +3.69 | +6.44 |

Relative to `base`, the head realizes `109.5%` of the missable test swing. The
tail movement is strong enough; the residual error is centering/calibration.

## Extraction Experiments

| Extractor over 24-dim descriptor | Test AUC | Relation to conditioned head |
| --- | ---: | --- |
| linear: axes + products + tail | 0.5920 | below `0.5979` |
| nonlinear MLP, same features | 0.5911 | below `0.5979` |

The value of the conditioned head is the 62-dim raw atlas plus identity
conditioning. Adding more interpretable axes to the 24-dim descriptor is not the
main remaining lever.

## Verdict

The identity-conditioned context head exceeds the descriptor-only extraction
ceiling and matches or slightly exceeds the empirical context-score swing. The
remaining global payoff is small and likely comes from calibration shrinkage or
centering, not from adding more shared descriptor axes.
