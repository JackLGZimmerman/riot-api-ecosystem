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

The next experiment should not be another cap/LR/confidence sweep. It should
build a richer split-safe target surface that preserves decision context:

- no-group probability band,
- role and side,
- champion/build identity group,
- patch or temporal cohort where support allows,
- relationship prior direction and support,
- semantic group balance,
- loadout context when it is known pregame,
- cross-fit label-derived residual direction.

The target should be evaluated first with replay and fixed-feature ceiling
learners. Only after that surface clears central-band NLL should it be wired into
HGNN training.
