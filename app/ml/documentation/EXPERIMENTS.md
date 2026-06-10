# HGNN Experiment Guidance

Last updated: 2026-06-10.

This note records the current semantic-boundary experiment findings and the rules
for future HGNN experiments. The goal is to keep useful conclusions while keeping
temporary ablation runners, probes, and tests out of the maintained code surface.

## Current Finding

The semantic group examples are still critical for evaluation. They expose
specific champion/build/context failures that aggregate accuracy, AUC, and even
group-level EB calibration can hide. The problem was not that the examples were
unhelpful; the problem was that the attempted training targets converted them
into weak row-level decision signal.

The repeated failure mode was:

- Central-band accuracy moved, often by roughly `+0.5pp` to `+0.8pp`.
- Central-band NLL stayed near `+0.001` to `+0.0015`, below the promotion gates
  of `+0.003` validation and `+0.002` test.
- Stronger heads, confidence gates, calibration weights, harm penalties, and
  conditional residual tables did not materially change that NLL ceiling.

The best fixed-feature relationship ceiling result on the manifest-backed
relationship surface reached:

| Scope | Split | Accuracy gain | New total | NLL lift |
| --- | --- | ---: | ---: | ---: |
| central `p_no_group in [0.45, 0.55]` | validation | `+0.779pp` | `53.29%` | `0.001409` |
| central `p_no_group in [0.45, 0.55]` | test | `+0.767pp` | `52.75%` | `0.001380` |
| global | validation | `+0.293pp` | `58.00%` | `0.001281` |
| global | test | `+0.286pp` | `57.38%` | `0.001496` |

This is a useful accuracy record, but it is not promotion evidence. The NLL
ceiling says the current feature/target surface does not contain a sufficiently
stable probability correction for boundary decisions.

The decisive contrast is the oracle check: a label-aligned boundary oracle at
the same rough magnitude can produce about `+0.0099` central NLL lift, while the
current EB/group residual targets remain near zero held-out NLL lift. That means
the gate is attainable in principle, but the current semantic target direction is
not precise enough.

## In-Band Ceiling Decomposition (2026-06-10)

A split-safe fixed-feature ceiling separated two competing root causes for the
static-NLL plateau: "identity-derived surfaces have no further row-level
signal" versus "the signal exists but does not transfer across the
chronological patch boundary." The v29 cache makes the decomposition
observable without new data: `patch_features[:, 1]` is `1.0` only for games
whose patch has train coverage. Validation splits into `18,561` covered-patch
and `35,420` uncovered-patch central-band games; test is 100% uncovered.

Setup: production checkpoint frozen; no-group base = forward pass with
`semantic_group_features` zeroed; feature surface = union of compact group
features (25), raw identity-context axes (62), and frozen sidecar latents
(144) composed per game as blue-minus-red team difference, team sum, and
per-role differences plus the no-group logit (900 dims, train-stat
standardized); learner = capped (`0.5` logit, tanh) residual MLP `256/64` fit
with BCE on top of the frozen no-group logit, inner 10% holdout early
stopping, seed 4. Harness checks reproduced the recorded oracle
(`+0.009919` central val NLL at `+5.30pp`) and the production group path
(`+0.56pp` / `0.001175` central val).

| fit | fit rows | scope (val) | acc lift | NLL lift |
| --- | ---: | --- | ---: | ---: |
| train_full | 1,145,051 | central | +0.55pp | `0.001354` |
| train_full | 1,145,051 | central, covered patches | -0.04pp | `0.000795` |
| train_full | 1,145,051 | central, uncovered patches | +0.86pp | `0.001647` |
| train_match | 114,505 | central | +0.30pp | `0.000925` |
| val_crossfit (5-fold OOF) | ~114,505 | central | +1.32pp | `0.002414` |
| val_crossfit (5-fold OOF) | ~114,505 | central, covered patches | -0.50pp | `-0.001013` |
| val_crossfit (5-fold OOF) | ~114,505 | central, uncovered patches | +2.27pp | `0.004210` |

`train_full` on test central (100% uncovered) reached `+1.25pp` / `0.002050`.

Findings:

- The train-fit union surface lands exactly on the historical `~0.0014`
  plateau, confirming the plateau is not a feature-representation artifact:
  richer identity-derived features do not move a train-fit teacher.
- Same-era fitting more than doubles the recoverable signal at 10x less
  fitting data: the size-matched train fit reaches `0.000925` on val central
  while the val-crossfit out-of-fold fit reaches `0.002414`.
- The decomposition is sharp: on new-patch central games the in-era ceiling is
  `+0.004210` NLL with `+2.27pp` accuracy, clearing the `+0.003` gate level,
  while the same learner *hurts* the covered-patch cohort (`-0.001013`).
  Residual direction is patch-era-conditional; mixing eras in one teacher
  actively cancels signal.
- Verdict under the pre-registered rule (overall val central, train-only and
  crossfit both below `+0.003`): reject train-fit identity-derived boundary
  surfaces. The actionable root cause is temporal/patch drift of the residual
  field, not missing identity information from `app/classification`.

A post-run audit verified test isolation, out-of-fold assembly, and
inner-holdout early stopping, and bounded the one mechanism that could have
inflated the transfer gap: train-degenerate feature columns amplified by the
`1e-6` standardization floor. Only 2 of 900 columns (context axes 54/57
JUNGLE-role diffs, train-constant at that role) are degenerate, touching ~16
of 143,131 validation rows — at most `1e-4` central NLL, two orders below the
decision margins. Future reruns should still floor or drop train-degenerate
columns.

The val-crossfit numbers are diagnostic ceilings (out-of-fold, no test
contact), never promotion evidence: promotion still requires a train-side
construction evaluated on untouched splits. Artifacts:
`app/ml/data/experiments/semantic_boundary_inband_ceiling/inband_ceiling.{json,md}`
plus cached frozen-base predictions. The one-off runner and its tests were
removed after this conclusion was recorded, per the experiment rules; the
construction above is sufficient to rebuild it.

## Where The Issue Lies

The issue lies in extracting decision signal from semantic groups, not in the
existence of semantic group information.

Semantic groups are good at saying:

- this context exists,
- this context is interpretable,
- this context has a measurable empirical effect,
- this model under- or over-expresses that effect in aggregate.

The failed experiments asked the same grouped context to also say:

- this exact near-boundary game should move blue or red,
- this move should improve held-out NLL,
- the direction should remain stable across folds and splits,
- the signal should survive outside the audit bin where it was estimated.

That conversion is where the signal broke. Mean-bin calibration targets,
champion-specific raw examples, and coarse EB residuals are too blunt for
per-game boundary decisions. They can improve threshold accuracy by moving some
scores across `0.5`, but the direction/magnitude is not reliable enough to
improve probability quality at the required NLL level.

Future work should therefore change the data/target construction before changing
model capacity. The next useful target surface should be split-safe, train-only,
cross-fit where possible, and explicitly optimized for central-band NLL
direction rather than audit-bin mean alignment.

## Documentation Review

These documents are all useful, but not all in the same way.

| Document | Keep? | Current role |
| --- | --- | --- |
| `HGNN_CURRENT.md` | Yes, critical. | Production architecture and default behavior source of truth. Keep this current whenever model inputs, cache format, serving behavior, or promoted checkpoints change. |
| `HGNN_CONTEXT_EXAMPLES_AUDIT.md` | Yes, critical as an evaluation fixture. | The specific instances of group context are valuable. They prevent aggregate metrics from hiding semantic failures and should remain available for qualitative evaluation. Do not use raw example gaps alone as a promotion metric because many bins are noisy. |
| `HGNN_GROUP_CONTEXT_AUDIT.md` | Yes, critical as the aggregate guardrail. | This is the lower-noise EB/group companion to the examples audit. It should remain the semantic calibration guardrail alongside accuracy, AUC, NLL, and ECE. |
| `HGNN_CENTRAL_BAND_REVIEW.md` | Keep, but treat as historical. | Useful for the original missing-signal taxonomy and central-band intuition. It is not the current production source of truth and contains historical ablation narrative. Prefer referencing it for forensic background, not current promotion decisions. |

The specific context instances in `HGNN_CONTEXT_EXAMPLES_AUDIT.md` are worth
keeping even if they are verbose. They are the semantic equivalent of regression
fixtures: small enough to inspect, rich enough to show why a model is wrong, and
stable enough to keep future architectures honest. Their main limitation is that
they are evaluation examples, not a direct supervised target.

## Experiment Rules

Use NLL as the first decision gate for this semantic-boundary work.

- Record accuracy gains as both gain and new total, for example `+0.779pp` and
  `53.29%`, but reject a branch when central-band NLL is static.
- The main central band is `p_no_group in [0.45, 0.55]`. Also report
  `[0.475, 0.525]` as a sharper diagnostic, but do not promote on it alone.
- Do not select checkpoints, thresholds, temperatures, or target parameters on
  test data.
- Every target surface must be train-only and split-safe. Use leave-one-out or
  cross-fit estimates when the target is derived from labels.
- If a fixed-feature ceiling learner cannot clear the NLL gate, change the data
  representation before changing HGNN architecture.
- If direct target replay cannot move held-out central NLL, change the target
  construction before adding loss weights.
- Treat accuracy-only gains as useful records, not promotion evidence.
- Run small smoke diagnostics first, then a full-data run only when the smoke
  validates the artifact path and metric writer.
- Require at least seed `4` plus one additional seed before promotion.
- Keep temporary runners, ablation scripts, and test-only probes out of the
  maintained tree after the conclusion is documented.

## Promotion Gates

For semantic-boundary promotion, require all of the following:

| Gate | Requirement |
| --- | --- |
| Boundary accuracy lift | Full model beats no-group ablation by at least `+0.50pp` validation and `+0.30pp` test on central-band games. |
| Boundary NLL lift | At least `+0.003` validation and `+0.002` test central-band NLL lift. |
| Directional semantic use | Support-weighted sign agreement between semantic movement and train-only residual direction at least `55%` on validation and non-regressing on test. |
| Audit sanity | High-support semantic bins target `max_abs_gap <= 3.0pp` validation and `<= 3.5pp` test; report p95 gap. |
| Global guardrails | Global validation/test NLL must not worsen by more than `0.0002`; accuracy/AUC must not drop by more than `0.05pp`. |

Any model that improves audit MSE but fails boundary causal lift is rejected.
Any model that improves central accuracy but leaves central NLL near the current
`+0.001` band is also rejected.

## Next Data Direction

The in-band ceiling decomposition (2026-06-10) replaces the earlier "richer
identity surface" direction: richer identity-derived features do not move a
train-fit teacher off the `~0.0014` plateau, but the same features carry
`+0.0042` central NLL on new-patch validation games when the teacher is fit
in-era. The next experiment should therefore make the target construction
time-local rather than the feature surface richer:

- rolling residual teacher fit only on games before each candidate's
  timestamp (or trailing patch window), replacing the static all-train fit,
- patch-era interaction features or era-bucketed teacher heads, so old-era
  rows stop cancelling new-era direction,
- time-decayed sample weights as the cheapest variant of the same idea,
- rolling 1vX/loadout priors as the production analogue once a teacher
  variant clears the ceiling.

Each variant should be evaluated with the same fixed-feature ceiling harness
(frozen no-group base, capped residual learner, covered/uncovered cohort
report) before any HGNN wiring. The promotion gates are unchanged; in-era
crossfit numbers remain diagnostics only.
