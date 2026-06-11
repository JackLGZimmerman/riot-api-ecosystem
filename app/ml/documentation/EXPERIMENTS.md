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
chronological patch boundary." The v30 cache makes the decomposition
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

The era layout this exposed: the v30 splits are purely chronological and cut
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
  sidecar artifacts, and v30 cache on the rolled boundary.
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
dictionaries, compact encoder sidecar, and the v30 Python cache. The split and
cache counts below are the verification anchor for this refreshed state.

Raw `game_data.info` currently contains season 16 through patch 24, but the
current ML-valid pool after filtering reaches patch 11. The refreshed v30 cache
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

Execution note, 2026-06-10: the rolled-split production recipe was evaluated
with the active defaults from `HGNN_CURRENT.md`: `convex_encoder_mix` 128x32,
compact sidecar, semantic group features, batch `16384`, validation-accuracy
checkpoint selection, and seeds `4` and `5`. The from-scratch round and
warm-start round were both rejected for promotion under the pre-registered
validation gates; retained candidate artifacts live under
`app/ml/data/experiments/rolled_split_production/`.

Batch `16384` is the measured throughput setting for the current architecture
(`51,505` team-swap-augmented samples/s on the local RTX 5070 Ti). If the MoE or
other model capacity changes, batch size must be reselected by samples/s rather
than carried over mechanically.

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

### Rolled Split Round 1: From-Scratch Recipe Rejected (2026-06-10)

The full production recipe (active defaults, lr `3e-4`, from scratch) was
trained on the rolled split for seeds `4` and `5` and evaluated with the
no-group band harness
(`app/ml/data/experiments/rolled_split_production/eval_no_group_bands.py`,
validation printed only; test written to JSON unread). The previous
frozen-boundary production checkpoint was re-evaluated on the rolled windows
as the incumbent anchor (`existing/`); the harness reproduced its recorded
metrics (NLL to `4e-7`, accuracy within 8 of 164,792 rows from batch
reduction-order noise).

Validation results (rolled val, S16.9-16.10):

| run | global acc | global NLL | central n | central acc lift | central NLL lift | high-support max gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent (`existing/`) | `57.812%` | `0.673779` | 60,649 | `+0.711pp` | `0.001002` | `3.94pp` |
| scratch seed 4 | `57.690%` | `0.673584` | 122,304 | `+2.325pp` | `0.007908` | `10.85pp` |
| scratch seed 5 | `57.653%` | `0.674470` | 98,119 | `+2.573pp` | `0.006558` | `14.40pp` |

Decision: reject the from-scratch candidate on validation; test was not
inspected. Two gates fail:

- Audit sanity: high-support context max-abs gap `10.85pp`/`14.40pp` vs the
  `<= 3.0pp` target (incumbent: `3.94pp` on the same rolled val). Group EB gap
  MSE regressed from `1.28` to `17.9`/`25.2`, and the per-epoch context gap
  oscillated `38 -> 84 pp^2` between adjacent epochs at lr `3e-4`.
- Global guardrail vs the incumbent: raw val accuracy `-0.12pp`/`-0.15pp`
  (NLL/AUC marginally better on seed 4).

The large no-group lifts are not gate evidence: zeroing
`semantic_group_features` collapses the from-scratch models to `~55.9%`
global no-group accuracy (incumbent: `57.5%`), so the central band widens to
74%/60% of validation rows. From-scratch training co-adapts the main signal
with the group path, which makes the no-group ablation an unfaithful baseline
and inflates band lifts. Band lifts are only comparable across runs whose
no-group base matches the incumbent's.

Incumbent group-path lift on rolled validation (`+0.711pp` / `0.001002`
central NLL) sits at the historical `~0.001` plateau, confirming the plateau
carries over to the rolled windows for boundary-respecting checkpoints.

Round 2 direction: warm-start from the incumbent checkpoint
(`app/ml/data/hgnn_production_model.pt`) and fine-tune on the rolled train
window at lr `1e-4` (no parameter freeze; same architecture), seeds `4` and
`5`, batch `16384`, val-accuracy checkpointing. Rationale: the failure
signature is recipe-induced miscalibration, not rolled-data damage, and the
incumbent's lineage is exactly this low-lr warm-start path; the fine-tune
should keep the calibrated group geometry while collecting the same-patch
freshness dividend measured by the time-local teacher ceiling.

### Rolled Split Round 2: Warm-Start Fine-Tune, Gates Decide Rejection (2026-06-10)

Round 2 warm-started from the incumbent production checkpoint
(`app/ml/data/hgnn_production_model.pt`), fine-tuned on the rolled train
window at lr `1e-4` (no parameter freeze, batch `16384`, val-accuracy
checkpointing, seeds `4`/`5`; best epochs `2`/`1`, ~6 minutes each). All
selection used validation only; test was never inspected.

Global validation and audit guardrails (rolled val):

| run | acc | NLL | AUC | high-support max gap | group EB MSE / max gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| incumbent | `57.812%` | `0.673779` | `0.60608` | `3.94pp` | `1.28` / `3.74pp` |
| warm seed 4 | `57.936%` | `0.673202` | `0.60786` | `3.53pp` | `1.39` / `4.58pp` |
| warm seed 5 | `57.910%` | `0.673262` | `0.60696` | `3.87pp` | `1.62` / `3.88pp` |

The fine-tune dominates the incumbent on every global validation metric and
restores a faithful no-group base (`~57.5%` global no-group accuracy, band
sizes within 4% of the incumbent's), unlike the round-1 from-scratch runs.

Within-run no-group band lifts (validation):

| run | central acc lift | central NLL lift | diagnostic acc lift | diagnostic NLL lift |
| --- | ---: | ---: | ---: | ---: |
| incumbent | `+0.711pp` | `0.001002` | `+1.029pp` | `0.000830` |
| warm seed 4 | `+0.978pp` | `0.001535` | `+1.410pp` | `0.001268` |
| warm seed 5 | `+0.991pp` | `0.001445` | `+1.525pp` | `0.001120` |

Gate check: central accuracy lift passes (`+0.98pp`/`+0.99pp` vs `+0.50pp`);
central NLL lift fails (`0.001535`/`0.001445` vs `0.003`). Audit metrics are
mixed-in-range (context gaps better than incumbent, group EB slightly worse,
both at the documented audit-floor noise scale).

Cross-model comparison on the incumbent-fixed central band (the time-local
teacher construction; band = incumbent `p_no_group in [0.45, 0.55]`,
validation, covered = S16.9 rows with same-patch train history):

| scope | n | warm4 NLL lift vs incumbent full | vs incumbent no-group | warm4 acc lift vs incumbent full |
| --- | ---: | ---: | ---: | ---: |
| central | 60,649 | `0.000530` | `0.001532` | `+0.228pp` |
| central covered | 27,499 | `0.000704` | `0.001759` | `+0.462pp` |
| central uncovered | 33,150 | `0.000386` | `0.001344` | `+0.033pp` |

(Seed 5 is uniformly slightly lower: `0.000376` central vs incumbent full.)

Findings:

- The production-true rolled refresh delivers a real but small model-level
  dividend (`~0.0005` central NLL, `~0.0006` global NLL, `+0.12pp` accuracy),
  concentrated in the covered cohort, directionally matching the time-local
  teacher ceiling at roughly half its size on covered rows
  (`0.001759` vs `~0.0030-0.0034` teacher-level same-patch lift over the
  no-group base).
- The `+0.003` validation central NLL gate is structurally unreachable under
  the rolled chronological protocol: the teacher ceiling itself is `~0.0030`
  on the covered ~45% of the band and `~0.0018` on the uncovered remainder,
  bounding any boundary-respecting candidate near `~0.0023` overall. The
  rejection is a property of the protocol mix (held-out windows always
  contain uncovered patches), not a near-miss of this candidate.
- Same-patch history in train lifts the group path only mildly within-run
  (covered `0.001728` vs uncovered `0.001375` central NLL lift for warm4):
  the freshness dividend flows mostly through the non-group paths, consistent
  with the teacher finding that the signal is era freshness, not the group
  target surface.

Decision: under the pre-registered step-5 validation gates the candidate is
rejected; the one-time test confirmation was not run and test remains
untouched for a future predeclared candidate. Per step 9, no semantic
architecture sweep follows. The remaining levers are user-gated protocol and
operations decisions, not model changes:

1. Adopt a separate production-refresh promotion gate (global
   validation NLL/accuracy/AUC improvement plus non-regressing audit), under
   which warm seed 4 is promotable as a data refresh of the same
   architecture.
2. Keep the semantic-boundary `+0.003` gate for what it was designed for —
   group-path architecture changes — and stop applying it to data refreshes.
3. Operations: refresh cadence well inside a patch remains the documented
   recommendation; the rolled fine-tune (~6 minutes) makes within-patch
   refresh cheap.

Round-2 artifacts: `app/ml/data/experiments/rolled_split_production/`
(`warm4/`, `warm5/`, `existing/`, `no_group_bands.json`, `cohort_bands.json`,
`cross_model_central.json`, `val_probabilities.npz`, plus the temporary
runners `eval_no_group_bands.py`, `cohort_bands.py`,
`cross_model_central.py`; the data directory is gitignored).

### Player Priors Round: First Gate-Scale Signal (2026-06-10)

After round 2 closed the data-refresh question, the largest unused signal was
identified by input audit rather than architecture search: every prior in the
model is champion-identity-keyed, so the model carried zero player-skill
signal even though `participant_stats` records `puuid`. This round added
draft-safe per-player priors end-to-end and found the first direction that
moves central NLL at gate scale.

Probe evidence first (validation, train-window features only): the single
feature "blue minus red mean per-player champion-experience games"
(`d_pc_games`) alone reaches `0.5614` AUC / `54.7%` accuracy; coverage is
`95.2%` of slots. A linear logistic stack of the warm4 logit plus four player
team-diff features fit how the signal should enter: accuracy
`58.24% -> 59.33%`, AUC `0.607 -> 0.625`, NLL `-0.0067` (in-sample on
validation, so an upper bound).

Data path (committed): `ml_game_player_pivot` tuples gained a `puuid` element;
train-scoped `player_1vx` / `player_champ_1vx` aggregates (6030/6031) and
COMPLEX_KEY_HASHED dictionaries (7015/7016); v30 cache adds four arrays
(`player_rate/cnt`, `player_champ_rate/cnt`) with leave-one-out adjustment on
train rows and nested EB smoothing (player-champ shrinks toward the player's
smoothed overall rate). Core v29 arrays stayed byte-identical, so existing
checkpoints and sidecar artifacts remain aligned. Model wiring: optional
`use_player_priors` path with `player_prior_mode` `residual`/`node`/`both`,
zero-init so warm starts are exact no-ops.

Three trainings, all warm-started from warm4, selected on validation only:

| run | recipe | val acc | val NLL | verdict |
| --- | --- | ---: | ---: | --- |
| warm4 (baseline) | no player path | `57.93%` | `0.67320` | incumbent |
| player4 | full fine-tune, lr `1e-4` | `58.51%` | `0.68011` | rejected: calibration collapse (high-support gap `12.1pp`) |
| player4_frozen | slot-level node head, frozen base, lr `1e-3` | `57.68%` | `0.68462` | killed at epoch 1: perturbing `phi_node` under a frozen readout miscalibrates immediately |
| player4_res | game-level linear residual (9 params), frozen base, lr `1e-3` | **`58.86%`** | **`0.66851`** | accepted on validation |

`player4_res` matches the probe's functional form exactly (logistic head on
blue-minus-red team means) and reproduces its gains out-of-sample: `+0.93pp`
accuracy, `-0.0047` NLL, AUC `0.6199` (`+0.0121`) vs warm4, with audit in
range (high-support max gap `3.44pp`, group EB in the documented noise band).
The no-player ablation attributes the lift causally: global `+0.92pp` /
`0.00466` NLL, central band (`p_no_player in [0.45, 0.55]`) `+2.26pp` /
`0.00437` NLL — the first candidate to exceed the `+0.003` central NLL scale
that every semantic-target construction plateaued under. The epoch curve
shows the head overshooting the validation optimum after epoch ~2 (train rows
use LOO features from an earlier window, so the train-optimal coefficient is
larger than the val-optimal one); val-accuracy checkpointing handles it.

Lesson reinforced twice in one round: expressive player paths trained
end-to-end with BCE wreck calibration (full fine-tune and slot-level node head
both failed); the probe-validated low-capacity form on a frozen base captured
the signal cleanly.

Artifacts: `app/ml/data/experiments/rolled_split_production/player4_res/`
(checkpoint, metrics, `player_eval.json` ablations; gitignored data dir).
Note `eval_player.py`'s AUC column is unreliable — use the trainer's AUC.

Follow-ups resolved in the next round (below): seed-5 reproduced player4_res
identically; the nonlinear residual head (`--player-residual-hidden 32`)
overfit from epoch 1 (val NLL `0.67462` vs `0.66840`) and a lr-`1e-5` full
unfreeze from player4_res was flat — the 9-param linear form on a frozen base
is the right altitude. The one-time test confirmation is reported below.

### Player Priors Round 2: Validation Gains Do Not Survive the Test Window (2026-06-10)

This round pushed the player-prior lever further and then took the one-time
test confirmations. The headline: every player-prior head — however
regularized — improves validation and **hurts test**. The lever is blocked by
aggregate staleness, not by modeling.

**Role experience (v31, committed).** Probe: per-`(puuid, teamposition)` train
game count `d_role_games` has `0.5481` solo AUC and was the only survivor of a
recency/level screen (`d_level` `0.4907` dead; recency rates collinear or
dead) — matchmaking balances skill but not champion/role experience. Data
path: `player_role_1vx` aggregate (6032) + dictionary (7017), v31 cache array
`player_role_cnt` (LOO-adjusted), `player_prior_feature_dim` config (8 = v30
blocks, 11 adds role conf/log-count/missing) keeps old checkpoints loadable.
Core v30 arrays stayed byte-identical. Trained the same frozen-base linear
residual with dim 11 (seeds 4/5, identical): val `58.79-58.80%` / `0.66826` —
a wash vs dim-8 player4_res (`58.86%` / `0.66851`). The probe's `+0.27pp` was
measured with a val-fitted stack; under train-window fitting it washes out.

**Window-adapted head (closed-form).** The residual head is linear over
blue-minus-red team-mean features on a frozen base, so it can be fitted
offline as ridge logistic regression with the warm4 logit as offset and
patched into a checkpoint (reconstruction verified exact: the patched
player4_role head reproduces the trainer's val `0.58788` / `0.66826` to 5
decimals). Two findings from the (window, lambda) sweep on validation:

- Unregularized fits lose to the SGD epoch-1 checkpoint at every window
  (best `0.67227` NLL vs `0.66826`): train rows carry LOO features while val
  rows use full priors, so the fit needs explicit ridge — early stopping was
  supplying that regularization implicitly.
- With ridge, late windows dominate full-train smoothly. Center-of-region
  pick `dim11, last-15% of train by gamecreation, lambda 0.15` (standardized):
  val `59.22%` / `0.66732`, ablation lift `+1.21pp` global / `+2.76pp`
  central. Best NLL cell `0.66640` (dim8, last-10%, lambda 0.3).

**One-time test confirmations (both fail).**

| candidate | val acc / NLL | test acc / NLL | test ablation lift |
| --- | --- | --- | --- |
| player4_res (SGD, dim 8) | `58.86%` / `0.66851` | `57.52%` / `0.67646` | `-0.16pp` acc, `+0.0029` NLL |
| late15_l015 (ridge, dim 11) | `59.22%` / `0.66732` | `57.54%` / `0.67771` | `-0.13pp` acc, `+0.0044` NLL |

The player path is net negative on test for both the conservative and the
window-adapted head. Cause: player aggregates are frozen at the train
boundary; val sits directly after that boundary (~0-12 days stale) and test
another window later (~12-24 days), and the prior-to-outcome relationship
decays past sign-flip within that horizon. This is the same structural
constraint round 2 of the rolled-split work isolated for the teacher: frozen
data, not architecture, is binding.

**Verdict.** No promotion. The player-prior lever carries real draft-safe
signal (the val gains are causal per ablation and reproduce across seeds and
fitting methods) but cannot clear a test gate while serve-time aggregates are
frozen at a boundary weeks in the past. Production realization requires
continuously refreshed player dictionaries (staleness of hours, not weeks) —
the existing user-gated data-refresh decision. Within the frozen evaluation
protocol, extrapolating the measured coefficient decay to the test lag
predicts a head of ~zero, i.e. no test gain is available from this lever even
in principle.

**60% gate assessment.** With player priors excluded, the validated frontier
is warm4 at `57.9%` val / `~57.7%` test. Every remaining input axis has been
audited: context head saturated at the draft-time ceiling, relationship
features removed as dead, recency/level dead, role experience marginal,
player skill blocked by staleness. The `>=60%` val+test accuracy gate in
HGNN_CURRENT.md is not reachable under the frozen split + frozen aggregates
protocol; the two levers that move it (rolling split refresh, refreshed
player aggregates) are both pipeline decisions outside model training.

Artifacts (gitignored data dir):
`rolled_split_production/{player4_role,player5_role,late15_l015}/`,
`late_head_fit.py` + `late_head_fit_report.json` (sweep), patched-checkpoint
candidate in `late15_l015/model.pt`. The `eval_player.py` no-player ablation
now also zeroes `player_role_cnt` for dim-11 checkpoints.

### Champion Strength / Meta Drift: Oracle Ceiling Is Empty (2026-06-11)

Direction shift away from player-skill features: treat champion strength as a
patch/meta freshness problem. Unlike player dictionaries, rolling champion
aggregates stay draft-safe and fresh into val/test (strictly-before-match
windows over public champion winrates), so a val gain here would be expected
to transfer to test. The audit and a future-knowledge oracle bound show the
axis carries no exploitable signal on top of the current base.

**Existing coverage.** The model's champion-strength inputs are the frozen
train-window `(championid, teamposition, build)` winrate prior (`synergy_1vx`)
plus champion identity embeddings and the context atlas — no time or patch
dimension. `game_data.info` has a `patch` column; train spans patches 1-9
(Jan 8 - May 5), val is patch-9 tail + patch 10, test is patch-10 tail +
patch 11. Patches 10/11 are absent from train entirely, so if meta drift
mattered, this split layout would surface it maximally.

**Drift is small.** Champion(x role) winrate drift beyond sampling noise,
train-pooled vs eval windows: `~0.7-0.8pp` RMS per cell (champ-role cells
with support; corr `~0.72`), `~0.5pp` champion-level; biggest champion-level
movers `+/-1.5-2.8pp`. No new or low-support champions (all 172 have
`>=1000` train games). Riot balancing keeps the drift an order of magnitude
below per-player skill spread (5-10pp), and blue-minus-red team-mean
aggregation shrinks it by another `sqrt(2/5)`.

**Oracle ceiling probe (`champ_oracle_probe.py`).** Deliberately leaky upper
bound: give the frozen warm4 base a blue-minus-red feature built from the val
window's own champion(x role[, patch]) winrates (EB-shrunk toward train,
delta in logit space) and fit the 1-D head on val itself. Any honest rolling
or patch-bounded feature is strictly dominated by this oracle. On val
(base `57.936%` / `0.673202`):

| oracle variant | acc lift | NLL lift |
| --- | --- | --- |
| champion-level, s50/s200 | `+0.005pp` | `-0.00001` |
| champ-role, naive s50 | `+0.42pp` | `-0.0021` (leakage, see below) |
| champ-role LOO, s50/s200 | `+0.003-0.005pp` | `-0.00001` |
| champ-role-patch LOO (current-meta), s50/s200 | `+0.003-0.005pp` | `-0.00001` |

The naive champ-role s50 number is pure self-inclusion leakage: each match
sits inside its own cell's winrate, and with weak shrinkage the small
off-role cells leak the match's own outcome back into the feature (fitted
`w=4.6` on a `0.029`-std feature). Removing the match from its cell
(leave-one-match-out) collapses the lift to `+0.003-0.005pp` with `w~0.03`,
and conditioning cells on the match's own patch (the strongest "current
meta" variant) changes nothing.

**Verdict.** Axis closed without building the rolling pipeline: even perfect
future knowledge of champion-role-patch winrates, leakage-free, moves val by
`<=0.005pp` acc / `~0.00001` NLL. The base model (identity embeddings +
context atlas + frozen 1vX prior) already absorbs champion strength to the
point where its residuals do not correlate with true meta drift, and the
drift itself is too small and too team-averaged to matter. This also
explains the historical wash of champion-role patch deltas in the old broad
`T+L` temporal head. No candidate was formed, so no test read was taken or
needed — the bound is on val and dominates any implementation. The 60% gate
assessment from Player Priors Round 2 stands unchanged.

Artifacts (gitignored data dir): `rolled_split_production/champ_oracle_probe.py`
+ `champ_oracle_report.json`.
