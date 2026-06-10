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

## Time-Local Teacher Ceiling (2026-06-10)

This experiment ran the "time-local target construction" follow-up to the
in-band decomposition and settled which time-local variant carries signal.
Same documented harness (frozen production no-group base, 900-dim identity
surface, capped `0.5`-logit residual MLP `256/64` fit with AdamW lr `1e-3`,
batch `16384`, at most 12 epochs with patience 2 on an inner 10% holdout;
features train-stat standardized, degenerate columns masked, and held as
float16; 5 prequential buckets; seeds 4 and 5, plus post-decision robustness
seeds on the test refresh teachers). Validity anchors reproduced the recorded
oracle (`0.009919`) and production group path (`0.001175` / `+0.559pp`
central val) exactly.

Per-row time metadata: the cache stream is deterministic (keyset-paginated
`ORDER BY matchid` per split), so `(season, patch, gamestarttimestamp)` was
re-derived from ClickHouse (`ml_game_player_pivot` joined to `game_data.info`)
in cache row order and verified by exact `blue_win` equality on all 1,431,313
rows. Artifact:
`app/ml/data/experiments/semantic_boundary_timelocal_ceiling/row_time_meta.npz`.

The era layout this exposed: the v29 splits are purely chronological and cut
*inside* patches. Train spans 1601-1608 (ends 2026-04-22), validation is the
1608 tail (48,908 rows, train-covered) plus the first 8.6 days of 1609
(94,223 rows, uncovered); within the central band, test splits into 18,558
patch-1609 rows and 35,496 patch-1610 rows. Every protocol-clean teacher is
therefore >= 1 patch stale for most held-out rows.

Teachers (H\* = 28d half-life, selected on validation from {7, 14, 28}):

| teacher | fit data | eval | central NLL lift | era cohorts (NLL lift) |
| --- | --- | --- | ---: | --- |
| train_static (s4/s5) | train, uniform | val | `0.001550` / `0.001489` | 1608 `~0.0010`; 1609 `~0.0018` |
| train_recency_h28 (s4/s5) | train, decayed | val | `0.001399` / `0.001041` | 1608 `~0.0004`; 1609 `~0.0017` |
| prequential_h28 (s4/s5) | train + strictly-past val | val | `0.001684` / `0.001903` | 1608 `~0.0008`; 1609 `0.002138` / `0.002533` |
| prequential_uniform (s4) | train + strictly-past val | val | `0.001722` | 1609 `0.002202` |
| trainval_static (s4/s5/s6) | train + val, uniform | test | `0.002519` / `0.001957` / `0.003386` | 1609 mean `0.0034`; 1610 mean `0.0022` |
| trainval_recency_h28 (s4/s5/s6) | train + val, decayed | test | `0.002276` / `0.002078` / `0.002249` | 1609 mean `0.0030`; 1610 mean `0.0018` |

Pre-registered decision: `timelocal_mixed`. The prequential validation gate
failed (`0.001794` mean vs `0.003`); the pre-registered test candidate
(trainval_recency_h28, seeds 4/5) cleared the test gate (`0.002177` mean vs
`0.002`, global guardrails positive). Post-decision robustness seeds
(trainval_static s5/s6, trainval_recency s6) were appended without
re-evaluating the rule: the candidate clears the test gate on every seed
(three-seed mean `0.002201`, range `2.0e-4`), while the uniform refresh has a
higher but seed-noisy mean (`0.002621`, range `1.4e-3`, one seed below the
gate). Under the experiment rules a failed validation ceiling means no HGNN
architecture change and no promotion this round.

Findings:

- Era *inclusion* is the lever; era *weighting* is second-order. On
  train-only fits recency weighting was neutral-to-negative at every tested
  half-life (sign-consistent across 7 of 8 comparisons, each delta within
  seed spread). On the refresh teacher it traded a little mean lift
  (`0.00220` vs `0.00262` uniform) for roughly sevenfold lower seed
  variance. Time-decayed loss weights are not a production lever for closing
  the gate; do not pursue them as one.
- In-era history is the real lever, and its value grows with volume
  (directionally — two confounded measurement points, not a dose-response
  curve): val-1609 prequential rows (mean ~3 days of own-patch history)
  reached `~0.0023`; test-1609 rows scored with the full 8.6-day, 94k-row
  val-1609 history reached a three-seed candidate mean of `0.00304`
  (`0.00300-0.00308`) with `+1.3-1.8pp` accuracy — at the validation-gate
  level on that cohort. One-patch-stale cohorts (val-1609 from train,
  test-1610 from the refresh candidate) sit at `~0.0018`.
- The binding constraint is now the frozen chronological split boundary, not
  the feature surface, the targets, or the model: no construction that
  respects the Apr-22 train cutoff can deliver same-patch history to most
  held-out rows, and same-patch history is where the gate-clearing signal
  lives. Train-boundary-respecting teachers stayed at or below the historical
  plateau (train_static `0.001550`/`0.001489` here vs `0.001354` in the f32
  in-band run — the same plateau at gate scale under the f16/masked-column
  harness). All priors (1vX, loadout, patch) are also hard-scoped to
  `split = 'train'`, so served features carry the same staleness.

Artifacts:
`app/ml/data/experiments/semantic_boundary_timelocal_ceiling/timelocal_ceiling.{json,md}`
plus `row_time_meta.npz`. The one-off runner was removed after this conclusion
was recorded; the construction above plus the in-band section is sufficient to
rebuild it.

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

Status 2026-06-10: the two ceiling experiments below resolved this section's
question — the missing precision is era freshness, not target construction.
See "Next Data Direction".

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

The time-local teacher ceiling (2026-06-10) resolved the previous direction
list: rolling/refresh teachers carry real signal, recency loss-weighting does
not, and no train-boundary-respecting construction can clear the validation
gate because the gate-clearing signal lives in same-patch history that the
frozen Apr-22 boundary withholds. The next change is therefore to the split
protocol itself, not to targets or architecture:

- Roll the chronological windows forward: extend train through the freshest
  complete patch, assign new validation/test windows after the new boundary
  (`ml_game_split` reassignment), and rebuild the filtered tables, priors,
  sidecar artifacts, and v29 cache on the rolled boundary.
- Retrain and evaluate the standard gates on the rolled held-out windows.
  This is the production-true protocol: deployment always has data up to the
  refresh point. Measured dividend at the teacher level, by cohort: the
  half-patch-stale 1609 cohort reached a `0.00304` three-seed candidate mean
  (`+1.3-1.8pp` accuracy; uniform refresh higher-mean but seed-noisy), while
  one-patch-stale cohorts sat at `~0.0018`.
- Keep refresh cadence well inside a patch if the in-era dividend is the
  goal. The two available measurement points (~3 days own-patch history
  `~0.0023`; 8.6 days `~0.0030`) are directional cadence evidence, not a
  fitted curve — they differ in cohort and construction.
- The fixed-feature ceiling harness (in-band + time-local sections above)
  remains the gate for any cheaper variant before HGNN wiring changes.

Rolling the boundary regenerates split-scoped artifacts (filter tables,
priors, encoder sidecars, semantic context tables, cache), so it is a
deliberate pipeline operation, not a casual rerun. Promotion gates are
unchanged and apply on the rolled windows.

### Refreshed Data State (2026-06-10)

The model-development data refresh completed through the documented ClickHouse
path in `database/clickhouse/commands.md`: corrected participant rows, filter
stages, `valid_game_ids`, filtered participant rows, item-value totals,
`ml_game_split`, `ml_game_player_pivot`, active 6000/6020 priors, 7000
dictionaries, compact encoder sidecar, and the v29 Python cache. The split and
cache counts below are the verification anchor for this refreshed state.

Raw `game_data.info` currently contains season 16 through patch 24, but the
current ML-valid pool after filtering reaches patch 11. The refreshed v29 cache
contains `1,647,915` games:

| Split | Games | Season | Patch range | Timestamp range |
|---|---:|---|---|---|
| train | 1,318,331 | 16 | 1-9 | `1767834983987` - `1777993488223` |
| validation | 164,792 | 16 | 9-10 | `1777993501649` - `1779515542851` |
| test | 164,792 | 16 | 10-11 | `1779515564023` - `1780922846108` |

Patch distribution:

| Split | Patch rows |
|---|---|
| train | S16.1 `196,280`; S16.2 `166,734`; S16.3 `167,810`; S16.4 `136,967`; S16.5 `138,764`; S16.6 `159,109`; S16.7 `151,105`; S16.8 `122,466`; S16.9 `79,096` |
| validation | S16.9 `76,521`; S16.10 `88,271` |
| test | S16.10 `40,357`; S16.11 `124,435` |

The compact sidecar was regenerated from train-only classification matrices and
rewritten to `app/ml/data/semantic_identity_sidecar_compact.npz` (`5,518`
identity rows; static/full-game/temporal dims `16/64/64`). `build_dataset` was
then rerun so `app/ml/data/cache/cache_meta.json` records the refreshed sidecar
metadata and split sizes.

### Rolled Split Test Plan

1. Freeze this refreshed data state as the candidate rolled-boundary protocol.
   Record the split ranges above with every run so results are not compared
   against the older Apr-22/S16.8 boundary by accident.
2. Run a smoke train on seed `4` with the existing production recipe:
   1vX prior, champion/build identity embeddings, Loadout, patch-only Temporal,
   compact sidecar, semantic group features, and `convex_encoder_mix` 128x32.
   Verify cache loading, sidecar gather, metric writing, and no-group ablation
   evaluation before spending full training time.
3. If the smoke is clean, run the full production recipe on seed `4` plus at
   least one additional seed. Select only on validation; keep test untouched
   until the predeclared candidate is fixed.
4. For each seed, evaluate the full model and the no-group ablation on:
   global validation/test, central `p_no_group in [0.45, 0.55]`, and diagnostic
   `p_no_group in [0.475, 0.525]`.
5. Apply validation selection gates first: central NLL lift at least `+0.003`,
   central accuracy lift at least `+0.50pp`, positive global guardrails, and
   non-regressing semantic direction/audit metrics. Do not inspect test while
   choosing seeds, cadence, thresholds, teachers, or checkpoints.
6. After the validation-selected candidate is fixed, run one final test
   confirmation: central NLL lift at least `+0.002`, central accuracy lift at
   least `+0.30pp`, and the same global/audit guardrails. If test fails, reject
   the candidate; do not tune and retest on the same test window.
7. Re-run `HGNN_GROUP_CONTEXT_AUDIT.md` and the high-support examples in
   `HGNN_CONTEXT_EXAMPLES_AUDIT.md` as guardrails. Treat low-support context
   examples as qualitative only, not max-gap evidence.
8. Compare the rolled-boundary results to the old frozen-boundary production
   checkpoint and the time-local teacher ceiling. The key question is whether
   adding same-patch train history in a production-true split closes the
   historical `~0.0015` central NLL plateau.
9. If the refreshed protocol fails the NLL gates, reject another semantic
   architecture sweep until a cheaper fixed-feature ceiling on the refreshed
   split shows gate-level central NLL signal.
