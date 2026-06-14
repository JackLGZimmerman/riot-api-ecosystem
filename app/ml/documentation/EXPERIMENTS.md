# HGNN Experiment Guidance

Last updated: 2026-06-14 00:59 BST.

This note records closed-lever experiment findings and the rules for future
HGNN experiments. The goal is to keep useful conclusions while keeping
temporary ablation runners, probes, and tests out of the maintained code
surface. Sections are decision records; the setup and rules below apply to
all of them unless a record says otherwise.

## Standard Experiment Setup

These settings recur across every record below; records only state deviations.

| Setting | Value |
| --- | --- |
| Hardware | local NVIDIA RTX 5070 Ti, `16,303 MiB` VRAM; CUDA |
| Default training recipe | lr `3e-4`, batch `16384`, `max_epochs=40`, `patience=5`, weight decay `0`, raw test-accuracy checkpointing, `--raw-tensor-cache-device cpu` (~`48s`/epoch, ~10-15 min/seed) |
| Production model recipe | learned semantic MoE `128x32` + semantic group features (the `train.py` production overrides; `HGNNConfig` base defaults keep the MoE flags off) |
| Split protocol | v32 per-patch chronological 80/20: train `1,318,329` / test `329,586`, no validation split; test is the model-selection split, not an untouched holdout |
| Frozen evaluation artifacts | `app/ml/data/experiments/split_v32/` (untracked): seed checkpoints and metrics; no local frozen-logit `preds.npz` is currently present, so residual probes must first create or declare their own logit artifact |
| Probe methodology | frozen ensemble logit as offset, features train-stat standardized, ridge/IRLS logistic fit on train only, one test read; central band = ensemble `p in [0.45, 0.55]` |
| Leakage controls | LOO-adjust any train-row feature derived from labels (cache priors already are); include a no-information control (shuffled features or logit-only recalibration) before attributing a lift to feature content |
| Allowed pregame surface | champions, roles, bans, train-only build catalogs/candidate worlds, train-only historical aggregates over those keys, and patch/season metadata only with a runtime provider |
| Forbidden surface | summoner spells, runes, player identity, rank, PUUID, held-out observed final builds as accepted inputs, and any post-draft/in-game stats |
| ClickHouse | run from the repo root (`PYTHONPATH=.`); cap `max_memory_usage<=4e9` and `max_threads=2` on raw-table aggregations; `synergy_1vx` and other prior tables are small and safe |
| Invocation | `python -m app.ml.train --seed N --model-path ... --metrics-path ...`; promotion via `python -m app.ml.promote --checkpoints <seed ckpts>` |

## Accepted Pregame Baselines (2026-06-13 12:24 BST)

Artifacts:
`app/ml/data/experiments/asset_sep_marginal_test.json` and
`app/ml/data/experiments/asset_sep_modal_test.json`. Scope: accepted
leakage-free pregame scoring on the promoted 6-seed asset-separation artifact,
using champions, roles, train-only build catalog worlds, train-only HGNN priors,
and cached/provider-backed season-patch metadata required by the patch-head
artifact. No observed held-out final builds, summoners, players, runes, rank,
PUUID, or timeline state are used.

| run | artifact | raw acc / NLL | calibrated acc / NLL | verdict |
| --- | --- | ---: | ---: | --- |
| marginal `W=128,k_slot=3` | `app/ml/data/experiments/asset_sep_marginal_test.json` | `56.3064%` / `0.680652` | `56.3079%` / `0.680773` | accepted primary pregame baseline |
| modal `W=1` | `app/ml/data/experiments/asset_sep_modal_test.json` | `55.8589%` / `0.682588` | `55.9265%` / `0.682705` | accepted floor/regression guard |

Controls and diagnostics: modal `W=1` is the no-marginalization floor; the
baseline was compared against the superseded 3-seed affine row in
`MARGINAL_LOOP.md` and later against the negative patch seed9 W=128 and W=512
sweeps below. Retained joint mass for W=128 is mean `0.6913`, p10 `0.4642`,
mean worlds `122.36`, and `2.98%` below the `0.35` floor.

## Patch Blue-Side Residual Restore (2026-06-13 22:40 BST)

A restore run added the train-only patch side-prior head back to the
post-removal model: `patch_blue_rate_logit_delta` and
`patch_blue_rate_coverage`, `patch_feature_dim=2`, no hidden residual layers,
`patch_residual_max_abs_logit=0.15`. It uses only `(season, patch)` metadata
and train-only blue-side aggregates; no player, summoner, or rune fields are
used. Standard v32 recipe, raw test-accuracy checkpointing. Completed artifacts:
`app/ml/data/experiments/patch_restore/seed{4,5,6,7,8,9}.metrics.json`.

| run | selected epoch | observed-build test acc | observed-build test NLL |
| --- | ---: | ---: | ---: |
| seed 4 | 9 | `57.364%` | `0.674626` |
| seed 5 | 12 | `57.336%` | `0.675156` |
| seed 6 | 11 | `57.367%` | `0.675392` |
| seed 7 | 10 | `57.430%` | `0.674341` |
| seed 8 | 9 | `57.317%` | `0.674747` |
| seed 9 | 10 | `57.425%` | `0.674309` |
| mean | -- | `57.373%` | `0.674762` |

Comparisons: these are observed-build diagnostics, not accepted pregame
metrics. Against the current observed-build single-seed reference in
`HGNN_CURRENT.md` (`57.91%` / `0.6722`), the restore is worse by roughly
`0.48pp` accuracy and `0.0026` NLL. Against the immediately preceding local
no-patch `split_v32` seed4-9 metric files, patch restore is only `+0.085pp`
mean accuracy and `-0.00020` NLL, so the isolated patch-side effect is small and
well below promotion scale.

Runtime boundary fixed in the same implementation pass: patch-head checkpoints
now require `serving_patch=(season, patch)` or an explicit
`PatchFeatureProvider`; missing patch features fail at load/constructor time
instead of being zero-filled. Unknown or low-support patches may legitimately
produce `[0, 0]`, but only through the provider.

Accepted pregame rescore, completed 2026-06-13 23:03 BST:

| run | artifact | raw acc / NLL | calibrated acc / NLL | verdict |
| --- | --- | ---: | ---: | --- |
| patch restore seed 9, `W=128,k_slot=3` | `app/ml/data/experiments/20260613_2238_patch_seed9_w128/marginal_w128_seed9.json` | `55.0752%` / `0.685092` | `55.1868%` / `0.684919` | rejected: below the `W=1` modal floor and worse than the accepted W=128 baseline |

Run details: accepted source `pregame_marginal_build`, catalog
`b39d506e51eb`, `329,586` test rows, retained joint mass mean `0.6913`,
mean worlds `122.36`, fallback slots `champion_role=3,295,306`, `role=554`,
calibration fit on `200,000` train rows with bias `-0.08110`.

Verdict: patch-side prior remains rejected for accepted pregame prediction. The
observed-build lift is too small and the first W=128 marginal seed regresses
below the modal floor; do not update the accepted pregame `W=128,k_slot=3` or
modal baselines from this patch restore.

## Pregame Signal Direction Audit (2026-06-13 21:50 BST)

Read-only architecture/data audit under the current leakage policy:

- Higher-order relation aggregates are the next plausible draft-generic probe:
  start with train-only 1v1 and 2vX tables with champion, role, and build keys
  only. Ignore pivot tuple element 4 (`puuid`) and all `player_*` tables.
  Treat 2v1 as strict-gated follow-up material only and keep exact 3v1 closed
  unless a coarser backoff table is introduced.
  Required controls: logit-only refit, shuffled relation features, exact-only
  vs backoff ladders, support thresholds, and global plus central-band NLL.
- Time-capped build labels are feasible only as label diagnostics. Timeline
  item purchases may define sampled 15-minute labels and compare agreement /
  variance against final labels, but timeline stats and item purchases must not
  feed accepted pregame features.
- Encoder rebuilds are not justified yet. Existing static/full-game/temporal
  sidecars already reach the learned semantic MoE, and prior shuffled/one-hot
  controls showed semantic profile content is redundant with champion identity.
  Rebuild only after a frozen residual probe finds signal beyond controls.

Verdict: no new model architecture or cache array until a frozen residual probe
clears controls. The next implementation target is a probe-only relation
artifact, not production wiring.

## Higher-Order Relation Aggregate Coverage Probe (2026-06-13 23:06 BST)

Artifact:
`app/ml/data/experiments/20260613_2306_relation_table_probe/coverage.json`.
Scope: prebuilt aggregate tables only, train split only, no raw participant
aggregation, no player/summoner/rune fields, and no model wiring. This is a
support and noise-risk diagnostic for whether a frozen residual probe is worth
building.

| table | train rows | p50 / p90 support | rows `>=50` | support-weighted abs edge |
| --- | ---: | ---: | ---: | ---: |
| exact 1v1 | `872,555` | `2` / `48` | `85,143` (`9.76%`) | `0.0498` |
| exact 2vX | `694,867` | `2` / `48` | `68,052` (`9.79%`) | `0.0490` |
| exact 2v1 | `34,090,806` | `1` / `5` | `146,960` (`0.43%`) | `0.2155` |
| exact 3v1 | `71,088,318` | `1` / `1` | `76` (`0.0001%`) | `0.4481` |
| 1v1 no-build | `310,237` | `7` / `264` | `72,473` (`23.36%`) | `0.0355` |
| 1v1 champion-only | `29,412` | `813` / `2,331` | `29,401` (`99.96%`) | `0.0258` |
| 2vX build-group | `451,397` | `3` / `75` | `56,748` (`12.57%`) | `0.0400` |
| 2vX no-build | `143,380` | `9` / `413` | `38,050` (`26.54%`) | `0.0268` |
| 2vX champion-only | `14,703` | `1,214` / `3,950` | `14,502` (`98.63%`) | `0.0200` |

Interpretation: exact build-key 1v1 and 2vX have enough high-support rows to
justify a frozen residual probe with empirical-Bayes shrinkage and backoff.
Exact 2v1/3v1 tables are mostly one-off combinations; their large raw edge is
likely sparsity noise and should not be wired directly. If used at all, 2v1 must
enter behind strict support gates and shuffled-relation controls; exact 3v1 is
closed unless a much coarser backoff table is introduced.

Next probe shape: create a frozen-logit artifact first, then fit train-only
relation features with logit-only and shuffled controls. Candidate feature order
is exact 1v1 + exact 2vX, then no-build/build-group/champion backoffs; defer 2v1
and skip exact 3v1 for the first pass.

Controls/verdict: no shuffled or logit-only residual control was run in this
coverage pass because it was not a predictive feature fit. Treat the artifact as
support/noise-risk evidence only. It authorizes a frozen residual probe for
1v1+2vX with controls; it does not authorize production model wiring.

## Marginal Catalog Sweep (2026-06-14 00:58 BST)

Artifact:
`app/ml/data/experiments/20260613_2308_w512_catalog/marginal_w512_k3_raw.json`.
Scope: accepted leakage-free pregame marginal scoring only; inputs are
champions, roles, the train-only build catalog, and cached/provider-backed
season-patch metadata required by the current patch-head artifact. No observed
final builds, players, summoners, runes, rank, PUUID, or timeline state are
used. Single raw eval, no calibration fit, full v32 test split.

| run | worlds | `k_slot` | raw acc | raw NLL | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| catalog sweep | `512` | `3` | `55.4529%` | `0.683566` | rejected |

Details: retained joint mass mean `0.7823`, p10 `0.6107`, p50 `0.8136`,
mean worlds `435.22`, low-mass share `0.253%`, fallback slots
`champion_role=3,295,306`, `role=554`.

Verdict: larger catalog coverage did not translate into better pregame
prediction for the current patch-head artifact. It is below the accepted raw
`W=128,k_slot=3` baseline (`56.3064%` / `0.680652`) and below the modal floor
on both accuracy and NLL (`55.8589%` / `0.682588`). Close `W=512,k_slot=3` as
an evaluation-refinement lever; do not promote it or spend more GPU on
`W=512,k_slot=5` unless a separate model-side change first improves the raw
W=128 surface.

## Historical Finding: Semantic-Boundary Targets (pre-v32)

Record from the old 80/10/10 chronological protocol; its validation split and
central-band gates no longer exist under v32. Retained because the
failure-mode taxonomy still applies to any future target construction.

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
observable without new data: the v30 cache's `patch_features[:, 1]` column is
`1.0` only for games whose patch has train coverage. Validation splits into `18,561` covered-patch
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
construction evaluated on untouched splits. The runner, its tests, and the
report artifacts were removed after this conclusion was recorded, per the
experiment rules; the construction above is sufficient to rebuild it.

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
rows.

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
  harness). All priors (1vX) are also hard-scoped to `split = 'train'`, so
  served features carry the same staleness.

The one-off runner and report artifacts were removed after this conclusion
was recorded; the construction above plus the in-band section is sufficient to
rebuild it.

## Draft-Only Residual Probes Under the Per-Patch Split (2026-06-11)

With the per-patch 80/20 protocol validated (see `HGNN_CURRENT.md`), this
experiment asked where additional draft-only signal remains. Hard constraint,
set by the user and binding on all future work: the model is draft-generic —
no player information of any kind (no puuids, no player priors, no rank
features). The admissible surface is champions, positions, bans, and
(champion, role, build)-keyed historical profiles via the three encoder sidecars.

Setup: three default-recipe v32 seeds (4/5/6) frozen; residual probes fit on
train only and scored on test, with three cohorts — global, central band
(seed-4 `p in [0.45, 0.55]`, 122,527 rows), and the stable core (all three
seeds in band, 52,008 rows = 15.8% of test). Feature blocks, train-stat
standardized: ban features (3: train-window winrate/popularity/unbanned diffs
of the ten bans), the 10 cache loadout features, the 146-dim sidecar-latent
team difference (+ support means), and role-aligned per-lane latent diffs
(5x144; slot order verified against the `6900` pivot: 0-4 blue
TOP/JG/MID/BOT/UTILITY, 5-9 red). Learners: IRLS logistic over
`[logit, 1, features]` and a 128/64 GELU MLP residual with zero-init output.

| probe (offset) | global acc | band acc | core acc |
| --- | ---: | ---: | ---: |
| seed4 refit (slope+intercept) | `0.5800` | `0.5243` | `0.5104` |
| seed4 + bans(3) | `0.5800` | `0.5242` | — |
| seed4 + loadout(10) | `0.5806` | `0.5258` | — |
| seed4 + latent diff(146) | `0.5803` | `0.5251` | — |
| seed4 + all(159) | `0.5808` | `0.5265` | — |
| seed4 + MLP(159) | `0.5813` | `0.5279` | `0.5162` |
| seed4 + role-aligned diffs(720) | `0.5801` | `0.5247` | `0.5120` |
| 3-seed ensemble refit | `0.5826` | `0.5309` | `0.5163` |
| ensemble + all(159) | `0.5820` | `0.5300` | `0.5154` |
| ensemble + MLP256(159) | `0.5827` | `0.5316` | `0.5165` |
| ensemble + ban-identity embeddings | `0.5827` | `0.5315` | `0.5170` |
| ensemble + set probe(10x144 latents) | `0.5815` | `0.5300` | `0.5133` |

The loadout block measured here (summoner spells + runes + shards) was removed
from the model on 2026-06-13 as a direct result of the `+0.05pp` null finding
below; the numbers are kept as the evidentiary record. See `HGNN_CURRENT.md`.

Findings:

- The side prior is the one real correction in current features: team-swap
  augmentation forces side-symmetric predictions (model mean `p = 0.4930` vs
  true blue winrate `0.4818`). A train-fitted intercept is leak-free and worth
  `+0.11pp` global / `+0.31pp` band on a single seed. In-model fix: add a
  learned side bias.
- The 3-seed ensemble is the largest free gain: `+0.26pp` over the refit
  single seed (`+0.29pp` over raw), `-0.0011` NLL, `+0.66pp` band, `+0.6pp`
  core. No new information required.
- Single-seed feature residuals are noise, not signal. The largest
  (`+0.08pp` global / `+0.21pp` band for all features) is paired-McNemar
  `z = 1.8` — below significance and 2x the seed-to-seed accuracy spread —
  and on the ensemble offset the same stack adds nothing (`0.5820` vs
  `0.5826`, within noise). The per-block deltas (loadout `+0.05pp`, latent
  `+0.03pp`) are noise-level descriptive numbers, not effects. Ban scalars are empty
  outright (standalone test AUC `0.5052`; the null is not an alignment
  artifact — the ban columns correlate with the label and the model logit at
  4-8 sigma with consistent sign on both splits).
- Role-aligned lane diffs are null (`+0.01pp`): attention pooling is not
  discarding linearly recoverable lane-matchup structure from the encoder
  latents.
- The stable core shows no sparsity signature: core games have *higher*
  encoder support than the rest of test (mean `51.8k` vs `49.5k`), equal
  loadout coverage (`0.918` vs `0.915`), equal-or-higher champion and
  (champ, build) train counts, and slightly *smaller* latent team-diff
  norms. The hard set is well-covered drafts that are close to 50/50 —
  consistent with genuinely balanced drafts rather than a data blind spot.
  Two caveats: the compared properties are themselves encoder-derived, so a
  purely representational blind spot would be invisible to this comparison;
  and the core is not literally at coin-flip — the ensemble still scores
  `~0.516` there (6.7 sigma above 0.5 at `n = 52,008`).

A separate methodology audit reproduced the probe numbers exactly, verified
train-only fitting and standardization, confirmed cache/preds/feature/ban row
alignment through the `ORDER BY matchid` contract plus label-equality and
correlation checks, and confirmed fp16 storage cannot mask signal at these
scales. It also flagged the first MLP probe as underfit (84 optimizer steps);
the strong-probe rows above are the corrected rerun (proper schedule, train
holdout early stopping, ensemble offset), including the two strongest
previously-untested classes: ban identities (learned embeddings instead of 3
scalars) and a nonlinear set probe over the full per-slot latents (team-mean
differencing is a linear compression).

Verdict: under the per-patch protocol, every probe class tried — linear,
fairly-trained shallow-nonlinear, ban identities, per-slot set probes — finds
nothing beyond the ensemble and the side intercept; measured residuals are
within noise of zero. The era-conditional residual that dominated the
2026-06-10 ceilings (in-band `+1.25pp` test central) was freshness and is now
inside the split change's `+0.50pp`. The conclusion is scoped to these probe
classes: still untested are gradient-boosted/deep probes over raw champion
*identities* (champion-pair interaction structure beyond what the encoder
latents linearize — though the HGNN itself is already a deep model over that
surface, so this amounts to boosting), draft-sequence/pick-order information
(not present in the data), and cross-patch train weighting. Remaining
headroom most plausibly requires new draft-generic information rather than
residual heads on current features.

Artifacts: `app/ml/data/experiments/split_v32/` (untracked) — local seed
checkpoints and metrics. The frozen-logit `preds.npz` and
`verify_equivalence.py` gate referenced by the original probe notes are not
present in the current working tree; recreate a frozen-logit artifact before
running new residual probes that depend on it. The probe runners were removed
after these conclusions were recorded.

## Production Promotion: 3-Seed Ensemble + Side Calibration (2026-06-11, superseded)

Both bankable levers from the probe round were promoted via
`app/ml/promote.py`: the production artifact became the 3-seed (4/5/6)
logit-mean ensemble with a train-fitted affine logit calibration — scale
`1.1686`, bias `-0.0432`, where the bias is the blue-side intercept that
team-swap augmentation suppresses. Test accuracy `0.58260` and test NLL
`0.67105`, vs `0.5788` / `0.6723` for the single-seed mean.
That round used a frozen-logit no-regression check that is not present in the
current local artifact directory. Superseded by the 6-seed bias-only promotion
below (2026-06-12), which also showed the train-fitted *scale* in this round was
in-sample-optimistic.

## Pregame Build Marginalisation (2026-06-12)

Phase A of `HGNN_BUILD_INTENT.md` landed: the train-only build catalog
(`app/ml/build_catalog.py`), the cache-side marginalised eval harness
(`python -m app.ml.marginal_eval`), the runtime `predict_marginal` path, and
the catalog-backed RL pool. Accepted modes score hypothesised build worlds
from `P(build | champion, role)` (EB-smoothed `synergy_1vx` train rows,
`tau=20`, fallback champion_role → role → global) and never read a held-out
row's observed `build_id`; world probabilities are averaged in probability
space and divided by retained joint mass. Catalog `b39d506e51eb`.

Full test split (`n=329,586`), production 3-seed calibrated ensemble:

| mode | build source | acc | NLL | brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| oracle (diagnostic) | observed final builds | `0.58260` | `0.67105` | `0.23921` | `0.0160` |
| modal (`W=1`, top build/slot) | pregame catalog | `0.55798` | `0.68432` | `0.24556` | `0.0231` |
| marginal (`W=128`, `k_slot=3`) | pregame catalog | `0.56186` | `0.68152` | `0.24424` | `0.0123` |

Readings:

- The oracle row reproduces the recorded production metrics exactly — the
  harness shares the training assembly path, so the marginal rows are
  apples-to-apples. The `~2.1pp` oracle-marginal gap is the value of knowing
  final builds, i.e. post-treatment information a draft-time consumer never
  has; `0.5619 / 0.6815` is the honest pregame deliverable.
- Marginalisation beats the single most-likely world by `+0.39pp` acc /
  `-0.0028` NLL and, notably, is already calibrated: a fresh train-fitted
  affine recalibration is near-identity (scale `1.012`, bias `-0.029`,
  `+0.01pp`), whereas modal scoring is overconfident (fitted scale `0.867`).
  Probability-space averaging supplies the variance reduction itself.
- Retained joint mass at `W=128`: mean `0.691`, p10 `0.464`, `3.0%` of games
  below the `0.35` low-confidence floor, mean `122` worlds/game. A later
  `W=512,k_slot=3` sweep (2026-06-14 record above) bought more retained mass
  but regressed metrics, so world-count expansion is closed as a standalone
  lever.
- Per-slot fallback was almost never needed on test (`554` role-fallback
  slots out of `3.3M`, no global) — train support covers the test
  champion-role surface.

Follow-up status: the `W=512` sweep was later taken and rejected; the Phase B
time-capped build-label diagnostic remains gated because it is the only
build-label lever that can close any of the oracle gap rather than just
approximate the prior.

The table above was measured against the then-production 3-seed artifact;
the 2026-06-12 promotion (below) moved production to the 6-seed bias-only
ensemble, shifting the absolute rows slightly while leaving the
oracle/modal/marginal structure unchanged.

## 1vX Saturation and the 6-Seed Bias-Only Promotion (2026-06-12)

Goal round: confirm the architecture record, then squeeze whatever remains in
the 1vX (champion, role, build) marginal surface — the fastest data grain to
train — before any higher-order (1v1/2vX) implementation. Outcome: the 1vX
axis is saturated and now formally closed by three independent bounds, but
the probe controls exposed a real production defect (in-sample calibration
scale) whose fix plus three more seeds banked `+0.107pp` accuracy and
`-0.00141` NLL.

**Bound 1 — low-support mass.** The only place 1vX prior *quality* can matter
beyond the identity embeddings is keys too rare for the embeddings to learn.
On test, slots below the cascade confidence threshold (`n < 50`, the only
rows shrunk toward the flat `0.5` prior) are `0.26%` of all slots; `2.6%` of
games contain even one, `0.04%` two. A perfect hierarchical
(build → champ-role → global) re-smoothing has no mass to act on; lever
closed without building it.

**Bound 2 — enrichment probe.** 24 features per game derived purely from the
train `synergy_1vx` table — team-diffs of smoothed cell rate, champ-role
parent rate, champion rate, build share, build entropy, cell-minus-parent
delta, naive-Bayes logit, log supports, plus per-lane mu/delta diffs and
cross-lane products — fit as a ridge IRLS residual on the frozen production
ensemble (train-only, LOO-adjusted parents, slot/role mapping verified by
exact count reconstruction). Every subset, including single features, gave
the *same* `-0.00066` test NLL at `-0.03pp` accuracy — the signature of a
recalibration artifact, not feature content.

**Bound 3 — controls and decomposition.** A logit-only recalibration control
(scale/bias/|logit|, train-fitted) gains exactly nothing, while an oracle
*test*-fitted bias also gains nothing — but an oracle test-fitted scale is
`0.824` (on the calibrated logit), worth `-0.00091` NLL: the production
affine calibration was overconfident on test. Cause: `promote.py` fit the
scale on train rows, which are in-sample for every member seed (logits
sharper than out-of-sample), so the fitted expansion `1.1686` overshoots —
the test-optimal net scale is `0.963 ~= 1.0`, i.e. the raw ensemble-mean
logit is already scale-calibrated. Re-running the enrichment probe over a
bias-only offset collapses the feature stack to `-0.000045` NLL / `-0.01pp`:
pure noise. With bounds 1-3 and the earlier semantic-profile shuffled
control, any (champion, role, build)-keyed enrichment is dominated by the
identity embeddings; the 1vX marginal surface is exhausted.

**Banked fixes.** Pre-registered candidates, one test read each (promotion
decision under the v32 protocol; test is the selection split):

| candidate | scale | bias | test acc | test NLL |
| --- | ---: | ---: | ---: | ---: |
| 3-seed 4/5/6, affine train-fit (incumbent) | `1.1686` | `-0.0432` | `0.58260` | `0.671053` |
| 3-seed 4/5/6, bias-only | `1.0` | `-0.0475` | `0.58254` | `0.670174` |
| 6-seed 4-9, affine train-fit | `1.2143` | `-0.0437` | `0.58349` | `0.670484` |
| 6-seed 4-9, bias-only (loadout+patch, superseded 2026-06-13) | `1.0` | `-0.0488` | `0.58367` | `0.669642` |
| 3-seed 7/8/9, bias-only (replication) | `1.0` | `-0.0503` | `0.58308` | `0.670020` |

Seeds 7/8/9 are fresh default-recipe runs (`0.5799/0.5795/0.5791` single-seed
test accuracy — inside the 4/5/6 band, so the ensemble-size gain replicates
across seed draws). Bias-only beats affine for both ensemble sizes
(`-0.0009` NLL, accuracy within noise), and ensemble size keeps paying at a
diminishing rate (`1->3` seeds `+0.26pp`, `3->6` `+0.11pp`). `promote.py`
defaults to `--calibration bias`; `--calibration affine` remains available.

**Asset separation + patch restore (2026-06-13).** The summoner-spell/rune
loadout head was removed permanently as measured noise (draft-generic policy);
the season/patch blue-side head was first removed alongside it, then restored
after the loadout-free ensemble regressed. Measured 6-seed bias-only
promotions on the v32 cache:

| ensemble | scale | bias | test acc | test NLL |
| --- | ---: | ---: | ---: | ---: |
| loadout + patch (pre-removal) | `1.0` | `-0.0488` | `0.58367` | `0.669642` |
| loadout removed, patch removed | `1.0` | `~-0.049` | `0.57762` | `0.672670` |
| **loadout removed, patch restored (promoted)** | `1.0` | `-0.0535` | **`0.57797`** | **`0.672385`** |

The patch head alone recovers `+0.035pp` accuracy / `-0.0003` NLL over the
patch-free ensemble; the bulk of the drop from `0.58367` was the loadout head,
which stays removed. Patch-head artifacts require
`serving_patch=(season, patch)` or an explicit `PatchFeatureProvider`, and
missing patch features fail fast instead of being zero-filled. Supervised oracle
eval consumes the cached per-game patch feature. Single-seed (seeds 4-9) test
accuracy `0.5732`–`0.5743` (mean `0.5737`), NLL `0.6743`–`0.6754`.
The previous frozen-logit equivalence script is not present locally, so any
future promotion must recreate that no-regression check before publication.

**Direction.** With the marginal surface closed, the remaining draft-generic
data lever is higher-order relationship structure — 1v1 lane matchups and
2vX co-located synergies. The 2026-06-04 removal record and the earlier
interaction findings still apply: any reintroduction must use LOO encoding,
nested EB pooling toward no-build/champion parents for sparsity, and must
clear a frozen-ensemble residual probe (with a shuffled-pair control) before
any model wiring is built. More seeds remain a known-positive but
diminishing lever (`~+0.05pp` expected for `6->12` at double the training
cost).

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

Historical note: at this point in the old protocol, the next idea was to change
data/target construction before changing model capacity, with split-safe,
train-only, cross-fit targets optimized for central-band NLL direction rather
than audit-bin mean alignment.

Status 2026-06-10: the two ceiling experiments below resolved this section's
question — the missing precision is era freshness, not target construction.
See "Next Data Direction".

## Documentation Review

| Document | Current role |
| --- | --- |
| `HGNN_CURRENT.md` | Production architecture and default behavior source of truth. Keep this current whenever model inputs, cache format, serving behavior, or promoted artifacts change. |
| `EXPERIMENTS.md` (this file) | Closed-lever decision records and the rules/gates template for future experiments. |
| `HGNN_BUILD_INTENT.md` | Build-intent leakage policy and the accepted leakage-free pregame marginal path. |
| `README.md` | Entry-point overview: production path, train/promote commands, cache contract. |

The context-examples and group-EB audit docs and their tooling were retired on
2026-06-11 after the semantic-boundary line closed (the audits' role was
selecting semantic-path candidates; the thresholds they promoted live on in
`app/ml/semantic_group_features.py`). The central-band review was retired as a
historical methodology note; its missing-signal taxonomy is summarized by the
sections above.

## Experiment Rules

Use NLL alongside accuracy as the decision gates; record both as gain and new
total (for example `+0.26pp` and `58.26%`).

- Every feature/target surface must be train-only and split-safe. Use only the
  allowed pregame surface listed above; player identity, summoners, runes, and
  held-out observed final builds are closed even if they are technically
  draft-visible or easy to join. Use leave-one-out or cross-fit estimates when
  a candidate is derived from labels.
- Probe before wiring: a frozen-ensemble residual probe (or a deliberately
  leaky oracle bound) must show gate-scale signal before any HGNN
  architecture or cache change is built for it.
- Always run a no-information control next to a feature probe (shuffled
  assignment, logit-only recalibration); attribute a lift to feature content
  only if it beats the control.
- Under v32 the test split is the model-selection split: checkpoint selection
  and promotion decisions read it by design. Spend test reads deliberately —
  pre-register candidates, then read once; do not iterate feature/parameter
  tuning against test.
- Treat accuracy-only gains as useful records, not promotion evidence, and
  vice versa.
- Run small smoke diagnostics first, then a full-data run only when the smoke
  validates the artifact path and metric writer.
- Require at least three seeds before promotion.
- Keep temporary runners, ablation scripts, and test-only probes out of the
  maintained tree after the conclusion is documented.

## Promotion Gates

Production promotion under v32 requires all of the following:

| Gate | Requirement |
| --- | --- |
| Test improvement | The candidate beats the promoted artifact on test accuracy and/or NLL with neither regressing materially (NLL within `+0.0002`, accuracy within `-0.05pp`). |
| Reproducibility | Gains hold across at least three seeds (or are seed-free, like calibration/ensembling changes). |
| Leakage review | No held-out information in any feature, prior, or calibration fit; accepted serving paths satisfy the draft-generic constraint and the build-intent leakage policy. |
| No-regression gate | Recreate or run a frozen-logit equivalence check for the candidate artifact before promotion; no local `split_v32/verify_equivalence.py` is currently present. |

The pre-v32 semantic-boundary gate table (central-band validation lifts, audit
max-gap targets) was retired with the validation split; the historical records
below still cite it.

## Next Data Direction (resolved 2026-06-11)

The time-local teacher ceiling (2026-06-10) showed the gate-clearing signal
lives in same-patch history that the old frozen Apr-22 boundary withheld;
recency loss-weighting carried nothing. The resolution was the split protocol
itself: the per-patch chronological 80/20 v32 protocol (`ml_game_split`; see
`HGNN_CURRENT.md`) banked the freshness dividend (`+0.50pp` test accuracy,
`-0.0037` NLL). Rolling the boundary forward again regenerates every
split-scoped artifact (filter tables, priors, encoder sidecars, semantic
context tables, cache), so it remains a deliberate pipeline operation, not a
casual rerun.

### Rolled-Split Round Context (2026-06-10, superseded)

Rounds 1-2 below were run on an interim rolled v30 protocol (train patches
1-9 `1,318,331` games, validation S16.9-10 `164,792`, test S16.10-11
`164,792`; same 1,647,915-game pool the v32 cache now splits per patch), with
the production recipe and validation-accuracy checkpointing, under
pre-registered validation gates. Both rounds were rejected for promotion, the
protocol was superseded by v32, and the candidate artifacts and runners were
removed (2026-06-11); the records below are the retained conclusions.

### Rolled Split Round 1: From-Scratch Recipe Rejected (2026-06-10)

The full production recipe (active defaults, lr `3e-4`, from scratch) was
trained on the rolled split for seeds `4` and `5` and evaluated with a
no-group band harness (validation printed only; test written to JSON unread,
runner since removed). The previous
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
rejected; the one-time test confirmation was not run and test remained
untouched in that old protocol. Per step 9, no semantic architecture sweep
followed. The remaining notes below are historical rolled-split operations
context, not active HGNN directions under the current v32 protocol:

1. Adopt a separate production-refresh promotion gate (global
   validation NLL/accuracy/AUC improvement plus non-regressing audit), under
   which warm seed 4 is promotable as a data refresh of the same
   architecture.
2. Keep the semantic-boundary `+0.003` gate for what it was designed for —
   group-path architecture changes — and stop applying it to data refreshes.
3. Operations: refresh cadence well inside a patch was the documented
   recommendation for that old setup; the rolled fine-tune (~6 minutes) made
   within-patch refresh cheap.

Round-2 artifacts were removed with the rolled-split experiment directory.

### Player Priors Round: First Gate-Scale Signal (2026-06-10; policy-closed)

After round 2 closed the data-refresh question, the largest unused signal was
identified by input audit rather than architecture search: every prior in the
model is champion-identity-keyed, so the model carried zero player-skill
signal even though `participant_stats` records `puuid`. This round added
per-player priors end-to-end and found the first direction that moved central
NLL at gate scale. This is retained as historical evidence only: under the
current pregame policy, player identity, rank, and PUUID-derived aggregates are
forbidden and are not future options.

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

Artifacts were removed with the rolled-split experiment directory.

Follow-ups resolved in the next round (below): seed-5 reproduced player4_res
identically; the nonlinear residual head (`--player-residual-hidden 32`)
overfit from epoch 1 (val NLL `0.67462` vs `0.66840`) and a lr-`1e-5` full
unfreeze from player4_res was flat — the 9-param linear form on a frozen base
is the right altitude. The one-time test confirmation is reported below.

### Player Priors Round 2: Validation Gains Do Not Survive the Test Window (2026-06-10; policy-closed)

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

**Verdict.** No promotion, and now forbidden by policy. The player-prior lever
carried real validation signal in this historical run, but it could not clear a
test gate while serve-time aggregates were frozen at a boundary weeks in the
past. More importantly for the current work, player identity and PUUID-derived
features are excluded outright; do not revive this as a refreshed-dictionary
direction.

**60% gate assessment.** In that historical protocol, with player priors
excluded, the validated frontier was warm4 at `57.9%` val / `~57.7%` test.
Context head saturated at the draft-time ceiling, relationship features had
been removed as dead, recency/level was dead, role experience was marginal, and
player skill was historically blocked by staleness. The old `>=60%` val+test
accuracy gate was not reachable under frozen split + frozen aggregates without
pipeline changes. The refreshed-player-aggregate option is now policy-closed:
player identity and PUUID-derived features are forbidden, not active levers.

Artifacts were removed with the rolled-split experiment directory. Closure
note, 2026-06-11: the user made the draft-generic constraint permanent (no
player information of any kind), and the opt-in player-prior code path —
cache arrays, config fields, model wiring, and CLI flags — was removed
end-to-end. This lever is closed, not parked.

### Champion Strength / Meta Drift: Oracle Ceiling Is Empty (2026-06-11)

Direction shift away from player-skill features: treat champion strength as a
patch/meta freshness problem. Unlike player dictionaries, rolling champion
aggregates were evaluated as historical draft-generic diagnostics
(strictly-before-match windows over public champion winrates), so a val gain
would have been expected to transfer to test. Any future patch/meta feature now
requires an explicit runtime provider. The audit and a future-knowledge oracle
bound show the axis carries no exploitable signal on top of the current base.

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
future knowledge of champion-role-patch winrates, with self-inclusion removed,
moves val by `<=0.005pp` acc / `~0.00001` NLL. The base model (identity embeddings +
context atlas + frozen 1vX prior) already absorbs champion strength to the
point where its residuals do not correlate with true meta drift, and the
drift itself is too small and too team-averaged to matter. This also
explains the historical wash of champion-role patch deltas in the old broad
`T+L` temporal head. No candidate was formed, so no test read was taken or
needed — the bound is on val and dominates any implementation. The 60% gate
assessment from Player Priors Round 2 stands unchanged.

Artifacts were removed with the rolled-split experiment directory.

### Semantic Identity Profiles: Real Signal, Fully Redundant With Champion Identity (2026-06-11)

Question: do hand-readable `(champion, position)` semantic profiles from the
`app/classification` surface ("high armor", "physical damage dealer", "high
sustain", ...) form useful draft-time features when aggregated into ally/enemy
team profiles and explicit cross-team interactions (e.g. ally armor vs enemy
physical damage)? And if yes, is mining richer encoder-derived latent groups
the next step? Answer: the signal exists in absolute terms but is entirely a
lossy re-encoding of champion identity, which the base already consumes; the
causal controls attribute the whole residual lift to non-semantic artifacts.

**Audit.** Semantic identity outputs are keyed `(championid, teamposition,
build)` and smoothed from train-split rows only (draft-safe, frozen at the
train boundary like every prior). Production already consumes this surface
twice: the 25-dim compact semantic group features and the learned semantic MoE
over the static/full-game/temporal encoder latents. Build-free `(champ, pos)`
coverage: 859 cells over 172 champions, median 949 train games per cell, 425
cells `>=1000`; the 254 cells `<200` are off-role picks handled by EB pooling
toward the champion level. Profiles are *stable*, not stale: train-window vs
val-window weighted correlation per behavioral dim is `0.96-0.998`
(cells with `n>100` in both windows).

**Probe (`semantic_profile_probe.py`).** Build-free `(champ, pos)` profiles:
10 behavioral dims from ClickHouse train rows (physical/magic/true damage
shares, damage to champions, damage taken, self-mitigation, CC time,
heals+shields on allies, turret damage, gold — per minute, EB strength 200
toward champion then global) plus 4 static dims (level-18 armor/MR/HP, attack
range). Features per match: blue-minus-red team-mean profile diff (14) and 11
explicit interactions (7 cross-team own-durability-vs-enemy-threat products,
2 within-team fits, 1 per-lane product, 1 damage-type balance). Head: ridge
logistic residual on the frozen warm4 base logits, fit on full train and on
the last-15% train window, val-only reads. Controls: shuffled
`(champ,pos) -> profile` assignment, intercept-only refit, standalone (no
base) fits, and a `(champ,pos)` one-hot champion-identity ablation.

Validation results (base `57.936%` / `0.673202`; best per variant):

| variant | val acc | NLL | acc lift vs base |
| --- | ---: | ---: | ---: |
| intercept-only, last15 window | `58.102%` | `0.672131` | `+0.166pp` |
| shuffled profiles, all features, last15 | `58.142%` | `0.672100` | `+0.206pp` |
| real profiles, diff features, last15 | `58.144%` | `0.672092` | `+0.208pp` |
| real profiles, diff+interactions, last15 | `58.137%` | `0.672098` | `+0.201pp` |

Standalone (no base offset, fit on train): real semantic features `53.67%`,
shuffled profiles `53.80%`, `(champ,pos)` one-hot `54.55%`.

**Decomposition.** ~80% of the residual lift (`+0.166pp` / `-0.00107` NLL) is
a single scalar: the frozen base is blue-side miscalibrated on the late-train
window and val (fitted intercept `-0.062` on the last 15% of train). The
remaining `~+0.04pp` comes from *any* fixed champion-keyed projection
(shuffled control), i.e. mild champion-composition recalibration, not
semantics. Real semantic content beyond the shuffled control: `+0.002pp` acc,
`-8e-6` NLL — zero. The interpretable cross-team interactions carry intuitive
signs on the full-train fit (sustain-vs-enemy-damage `+0.075`,
HP-vs-enemy-damage `+0.041`, MR-vs-enemy-magic `+0.035`,
armor-vs-enemy-physical `+0.014` in standardized space) but the armor/MR signs
flip on the last15 window and the shuffled-control equivalence marks the
whole block as noise-level. The standalone ordering (one-hot > shuffled ~=
real) confirms the semantic profiles are a lossy compression of champion
identity, which the base's champion embeddings already encode losslessly.

**Verdict.** Q1: simple semantic groups do contain absolute draft-time signal
(`53.7%` standalone vs `50%`) but zero incremental signal over the production
base — redundant with champion identity, not noisy and not stale. Q2: no,
mining richer encoder-derived semantic groups is not worth it for win
prediction: production already feeds the full encoder latents through the
semantic MoE, and the in-band ceiling decomposition showed the 900-dim union
of group features + context axes + sidecar latents plateaus at `~0.0014`
train-fit NLL with era freshness, not representation, as the binding
constraint. This probe independently reconfirms that conclusion on a
build-free `(champ, pos)` surface with causal controls. No candidate was
formed; no test read was taken. Historical 60% gate note: under that old
rolled-split protocol, the intercept-drift finding (`+0.166pp` from one scalar)
pointed at window freshness rather than richer semantic representation, and the
player-priors round showed late-window validation gains could flip sign on test.

Note: the probe's report `auc` values are invalid in the archived JSON — the
shared `_binary_auc(scores, targets)` helper was called with swapped
arguments, the same root cause as the known-buggy `eval_player.py` AUC
column. Conclusions use acc/NLL only; the probe script is fixed.

Artifacts were removed with the rolled-split experiment directory.
