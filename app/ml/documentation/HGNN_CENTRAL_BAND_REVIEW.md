# HGNN Central Band Review

Date: 2026-06-04

This audit reviews held-out validation/test games where the reference HGNN model
predicted `P(blue win)` in the central `0.475-0.525` band and the raw `0.5`
classification was wrong. The tight-band pass uses batches of 1,000 games; the
wider `0.425-0.575` follow-up uses a 2,000-game miss sample.

Allowed evidence was restricted to champion, role/position, build label,
historical champion/build/relationship performance, summoner spells, runes, and
patch/date. No player identity evidence is selected, emitted, aggregated, or used
for reasoning. The rune table uses `puuid` only inside the ClickHouse join that
aligns rune rows to participant slots; the analyzer does not materialize it.

## Artifacts

| Artifact | Purpose |
| --- | --- |
| `app/ml/data/experiments/hgnn_central_band_review_candidates_b1000_001.json` | Batch 1 candidates, 1,000 games. |
| `app/ml/data/experiments/hgnn_central_band_review_candidates_b1000_002.json` | Batch 2 candidates, 1,000 games. |
| `app/ml/data/experiments/hgnn_central_band_batch_b1000_001_analysis.json` | Batch 1 allowed-signal review. |
| `app/ml/data/experiments/hgnn_central_band_batch_b1000_002_analysis.json` | Batch 2 allowed-signal review. |
| `app/ml/data/experiments/hgnn_central_band_deep_b1000_review.json` | Second-pass deeper rune/build-profile review over the same 2,000 games. |
| `app/ml/data/experiments/hgnn_central_band_wide_425_575_candidates_b2000_001.json` | Wider `0.425-0.575` band miss sample, 2,000 games. |
| `app/ml/data/experiments/hgnn_central_band_wide_425_575_allowed_b2000_001_analysis.json` | Wider-band first-pass allowed-signal review. |
| `app/ml/data/experiments/hgnn_central_band_wide_425_575_deep_b2000_001_review.json` | Wider-band deeper rune/build-profile review. |
| `app/ml/data/experiments/hgnn_central_band_lift_estimate_475_525.json` | Accuracy-lift estimate for the `0.475-0.525` band. |
| `app/ml/data/experiments/hgnn_central_band_lift_estimate_425_575.json` | Accuracy-lift estimate for the `0.425-0.575` band. |
| `app/ml/experiments/hgnn_central_band_candidates.py` | Reproducible central-band candidate generator. |
| `app/ml/experiments/hgnn_central_band_allowed_review.py` | Reusable allowed-signal analyzer. |
| `app/ml/experiments/hgnn_central_band_deep_review.py` | Detailed rune page, secondary build, and build-margin analyzer. |
| `app/ml/experiments/hgnn_central_band_lift_estimate.py` | Reproducible current-vs-theoretical accuracy estimator. |

Earlier 100-game smoke artifacts remain in `app/ml/data/experiments`, but this
report supersedes them with the reproducible samples above.

## Model Context

Reference checkpoint:
`app/ml/data/experiments/semantic_focus_reference_w3000_cont6/model.pt`.

The checkpoint accounts for champion/build identity, 1vX champion-role-build
priors, support counts, and the learned semantic MoE/group context path. It does
not directly consume summoner spells, rune/perk choices, patch/date, or exact
1v1/2vX relationship arrays. The exact relationship arrays exist in the cache
and ClickHouse priors, but `use_relationship_integrations` is disabled for this
checkpoint.

Tight central-band held-out summary:

| Split | Central games | Central accuracy | Central misses | Blue WR |
| --- | ---: | ---: | ---: | ---: |
| validation | 82,346 | 52.56% | 39,067 | 46.31% |
| test | 81,564 | 52.24% | 38,958 | 46.58% |

## Batch Results

| Batch | Games | Correct at threshold 0.516 | Still wrong at threshold 0.516 | Games with unaccounted allowed signal | Patch mix |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 1,000 | 358 | 642 | 446 | S16.8: 177, S16.9: 496, S16.10: 327 |
| 2 | 1,000 | 396 | 604 | 460 | S16.8: 183, S16.9: 483, S16.10: 334 |
| total | 2,000 | 754 | 1,246 | 906 | S16.8: 360, S16.9: 979, S16.10: 661 |

In the first-pass analyzer, batch 2 repeated the same broad influence families
as batch 1. The deeper review below then checks finer loadout/build profile
signals on the same 2,000 games.

## Repeated Accounted Reasons

- Tuned `0.516` decision threshold fixes central blue-side overcalls/undercalls
  that are wrong only under a raw `0.5` cut (754).
- Still wrong after the tuned threshold because the model score remains central
  and the accounted champion/build 1vX priors plus semantic context do not move
  the logit enough (1,246).
- Patch/date is not an input; S16.9/S16.10 examples have no same-patch train
  solo prior in this chronological cache, so current model behavior cannot adapt
  directly to that patch shift (1,640).

## Unaccounted Influence Buckets

- Direct champion matchup/synergy priors are allowed but disabled (615). This
  combines exact team relationship edges (326) and strong individual matchup
  edges where the team-average relationship score stayed below the threshold
  (289).
- Rune/keystone-conditioned priors are absent (333). This combines team-level
  rune edge cases (278) and strong slot-level rune edge cases (55).
- Summoner-spell-conditioned priors are absent (121). This combines team-level
  spell edge cases (113) and strong slot-level spell edge cases (8).
- Patch-conditioned champion/build priors are absent where S16.8 has train
  overlap (49). S16.9/S16.10 should be treated as a rolling temporal-prior
  problem rather than a same-patch train lookup.

Counts overlap because one game can expose more than one missing signal.

## Second-Pass Deep Review

The second pass reviewed the same 2,000 games for more detailed allowed
loadout/build profile signals that the first pass did not test. It compares
train priors for the actual rune page, secondary rune pair, stat shards,
secondary build label, and build-label margin against the broader
champion-role-build or keystone/tree baseline.

| Signal | All 2,000 games | New-only vs first pass |
| --- | ---: | ---: |
| Secondary build profile beyond highest build label | 698 | 366 |
| Build-label margin profile | 645 | 348 |
| Full rune page beyond keystone/tree | 508 | 273 |
| Secondary rune pair beyond keystone/tree | 306 | 180 |
| Stat shard profile | 229 | 109 |
| Any deep signal | 1,480 | 803 |

The `new-only` column is the important second-pass result: 803 games had one of
these deeper profile edges even though the first-pass analyzer found no
relationship, broad rune, spell, or patch prior bucket.

This is not yet a serving-model prescription. Rune page, secondary rune, stat
shard, and summoner spell choices are pregame inputs and directly allowed.
Secondary build and build-margin signals come from
`participant_item_value_totals`, so they are useful forensic evidence about
build intent but would need a draft-safe source, declared build intent, or a
leakage-safe build-intent predictor before being used for pregame prediction.

## Representative Deep Games

| Match | Prediction | Actual | New deep signal | Evidence |
| --- | --- | --- | --- | --- |
| `NA1_5550830963` | blue, 52.24% blue | red | full rune + secondary rune + shards + secondary build + margin | No first-pass unaccounted bucket. Ornn TOP `ar_tank` full-rune profile was +0.93 pp over its broader baseline over 2,548 train games; the same game also had build-profile and margin edges. |
| `TW2_409309499` | blue, 50.18% blue | red | full rune + secondary rune + shards + secondary build + margin | No first-pass unaccounted bucket. Kaisa BOTTOM `on_hit` full-rune profile was +2.80 pp over baseline over 1,051 games; Ambessa TOP secondary-rune profile was +2.54 pp over baseline over 2,270 games. |
| `KR_8210891176` | red, 49.09% blue | blue | full rune + secondary rune + secondary build + margin | No first-pass unaccounted bucket. Yone MIDDLE `crit` full-rune profile was +4.93 pp over baseline over 229 games; secondary-rune profile was +2.85 pp over baseline over 1,866 games. |
| `LA2_1595081633` | red, 49.36% blue | blue | full rune + shards + secondary build + margin | No first-pass unaccounted bucket. Pantheon JUNGLE `ad_off_tank` full-rune profile was +7.56 pp over baseline over 125 games; Smolder BOTTOM stat-shard profile was +6.43 pp over baseline over 551 games. |

## Semantic Context Diagnostic

The current checkpoint does consume semantic group features through the learned
semantic MoE path, so generic composition claims are not "absent" in the same
way runes/spells/patch are absent. The deeper issue is whether the context logit
is specific and strong enough in central-band misses.

Among the 1,246 reviewed games that remained wrong after the `0.516` threshold,
`context_logit` favored the actual winner in only 102 games and favored the
wrong side in 1,144. Among the 715 threshold-still-wrong games with no
first-pass unaccounted bucket, `context_logit` favored the actual winner in only
41 games.

This supports a narrower direction: do not add broad hand-labeled composition
claims as conclusions yet. Instead, test role-restricted relationship and
composition features that the current semantic summaries can blur, especially
same-role lane matchups, `BOTTOM+UTILITY`, `JUNGLE+UTILITY`, damage skew into
enemy defensive builds, range plus siege, frontline thresholds, and hard-CC plus
burst interactions.

## Multi-Agent Critique

Three independent reviewers audited the same 2,000 games and then critiqued
each other's conclusions. The consensus was:

- Richer rune/loadout context is genuinely unaccounted: full rune pages,
  secondary runes, stat shards, spell pairs, and joint rune/spell/build priors
  should be tested with support-gated backoff.
- Role-restricted relationships are a stronger next step than generic
  composition labels. Same-lane, bot/support, jungle/support, and jungle/mid
  priors may preserve signal that all-pair averages dilute.
- Time should wrap every proposed prior. S16.9/S16.10 dominate this sample, the
  model has no patch/date input, and static train priors can become stale.
  Rolling, leakage-safe priors using only games before the candidate timestamp
  are required.
- Final-build profile evidence is useful for diagnosis, but unsafe as a direct
  pregame feature unless build intent is available before the outcome.
- Patch drift, loadout choices, and composition can explain each other if tested
  separately. Any claimed lift should survive patch-stratified ablations and
  calibration checks.

## Wider 0.425-0.575 Check

The requested `42.5/57.5` review was run as a wider `P(blue win)` band of
`0.425-0.575`, using the same non-identity evidence boundary and a 2,000-game
miss sample. This band contains almost the full validation/test distribution,
so its main use is to test whether the central-band findings still hold when
the model is a little less close to `0.5`.

| Split | Wide-band games | Wide-band accuracy | Wide-band misses | Blue WR |
| --- | ---: | ---: | ---: | ---: |
| validation | 138,837 | 55.35% | 61,993 | 46.23% |
| test | 138,815 | 54.99% | 62,485 | 46.50% |

The wider band produced 124,478 total raw `0.5` misses across validation/test.
The 2,000 sampled misses had this first-pass profile:

| Finding | `0.475-0.525` sample | `0.425-0.575` sample |
| --- | ---: | ---: |
| Correct at threshold `0.516` | 754 | 463 |
| Still wrong at threshold `0.516` | 1,246 | 1,537 |
| Games with first-pass unaccounted signal | 906 | 858 |
| Direct relationship/matchup bucket | 615 | 472 |
| Rune/keystone bucket | 333 | 377 |
| Summoner-spell bucket | 121 | 140 |
| S16.8 patch-conditioned bucket | 49 | 59 |
| S16.9/S16.10 temporal caveat | 1,640 | 1,624 |

The same influence families repeat; no new allowed influence family appeared in
the wider sample. The main difference is that the global threshold fixes fewer
misses, and rune/loadout detail becomes slightly more prominent than exact
relationship evidence.

The wider deep review also repeated the prior second-pass pattern:

| Deep signal | `0.475-0.525` sample | `0.425-0.575` sample |
| --- | ---: | ---: |
| Any deep signal | 1,480 | 1,511 |
| New-only vs first pass | 803 | 859 |
| Secondary build profile | 698 | 705 |
| Build-margin profile | 645 | 705 |
| Full rune page | 508 | 547 |
| Secondary rune pair | 306 | 337 |
| Stat shard profile | 229 | 230 |

Semantic context remains a residual concern. In the wider sample, `context_logit`
favored the actual winner in only 68 of the 1,537 games still wrong after the
`0.516` threshold. In the 921 threshold-still-wrong games with no first-pass
bucket, it favored the actual winner in only 38. This is better described as a
misdirected or too-generic semantic-context path, not merely an underweighted
one.

The wider sub-agent review agreed with the earlier critique: static train priors
are split-safe but not rolling, final-build profile evidence is diagnostic only,
and role-restricted relationships plus pregame loadout priors remain the best
next ablation targets.

## Accuracy-Lift Estimate

Reference checkpoint current accuracy from
`semantic_focus_reference_w3000_cont6/metrics.json`:

| Split | Raw `0.5` accuracy | Tuned `0.516` threshold accuracy |
| --- | ---: | ---: |
| validation | 55.82% | 56.63% |
| test | 55.45% | 56.22% |

The estimates below are diagnostic. They are computed from miss-only samples,
so the upper bounds assume sampled signal coverage extrapolates to all band
misses and that capturing those signals causes no regressions on currently
correct games. The margin-conditioned estimate is more conservative: it counts a
sampled miss only when the summed positive actual-side prior edges are large
enough to cross the current decision margin. It is still not a substitute for an
ablation.

| Band | Split | Current raw | Current threshold | All-signal miss-side upper bound | Post-threshold signal upper bound | Margin-conditioned raw | Margin-conditioned threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.475-0.525` | validation | 55.82% | 56.63% | 80.46% | 70.99% | 77.83% | 68.51% |
| `0.475-0.525` | test | 55.45% | 56.22% | 80.02% | 70.53% | 77.40% | 68.06% |
| `0.425-0.575` | validation | 55.82% | 56.63% | 94.13% | 84.92% | 86.18% | 76.47% |
| `0.425-0.575` | test | 55.45% | 56.22% | 94.06% | 84.72% | 86.05% | 76.21% |

For the validation set, the theoretical upper bound if all identified signal
families in the wider band were captured is 94.13% from the raw `0.5` baseline,
or 84.92% if starting from the current tuned threshold and only fixing
post-threshold signal-tagged misses. The margin-conditioned heuristic gives
86.18% raw and 76.47% threshold. These numbers clear the 60% target in theory,
but they should be read as "there is enough miss-side signal volume to justify
the ablations," not as expected production accuracy.

## Representative Games

| Batch | Match | Prediction | Actual | Unaccounted signal | Evidence |
| --- | --- | --- | --- | --- | --- |
| 1 | `LA1_1712392896` | red, 49.14% blue | blue | matchup + spell + rune + patch | Veigar MIDDLE `ability_power` vs Shyvana JUNGLE `on_hit` favored the actual side at 59.2% over 125 games; Leona UTILITY `utility_enchanter` was +15.7 pp in S16.8 patch prior. |
| 1 | `VN2_1404655488` | red, 48.42% blue | blue | relationship + spell + rune | Actual-side relationship edge was +4.24 pp; Xin Zhao JUNGLE Flash+Smite and keystone/tree priors were +12.3 pp and +12.5 pp over the base prior. |
| 2 | `JP1_581968596` | blue, 50.03% blue | red | relationship + spell + rune | Lee Sin JUNGLE `ad_off_tank` vs Samira BOTTOM `crit` favored the actual red side at 61.6% over 372 games; Xin Zhao loadout priors also favored red. |
| 2 | `LA1_1719584530` | blue, 50.87% blue | red | relationship + spell + rune | Brand UTILITY `ap_off_tank` vs Camille TOP `ar_tank` favored red at 60.7% over 135 games; Camille rune and spell priors were both strongly positive over base. |
| 2 | `EUW1_7832366830` | blue, 51.32% blue | red | relationship + patch | S16.8 patch priors favored the actual side: Amumu UTILITY `utility_protection` was +11.9 pp, LeBlanc TOP `ability_power` was +7.0 pp, and Seraphine MIDDLE `utility_protection` was +5.6 pp over aggregate priors. |
| 1 | `KR_8199751461` | red, 49.61% blue | blue | relationship + spell + rune | Yunara BOTTOM `crit` vs Jhin BOTTOM `lethality` favored blue at 68.5% over 355 games; Xin Zhao JUNGLE Flash+Smite and keystone/tree priors both favored blue. |

## Direction

The highest-signal expansion path is to reintroduce allowed, non-identity
relationship, loadout, and temporal inputs with support gating:

1. Add a leakage-safe temporal wrapper first: patch/date base-rate offsets,
   blue-side rate, role/meta drift, support aging, and rolling priors computed
   only from games before the candidate timestamp.
2. Enable or rebuild calibrated exact relationship paths for 1v1 and 2vX priors,
   then add role-restricted variants for same-lane, `BOTTOM+UTILITY`,
   `JUNGLE+UTILITY`, and `JUNGLE+MIDDLE` contexts.
3. Add pregame loadout features: summoner-spell pairs, full rune pages,
   secondary rune pairs, stat shards, and joint rune/spell/build priors with
   empirical-Bayes shrinkage and champion-role fallback.
4. Treat secondary build label and build-margin profile as research signals
   until a draft-safe build-intent source exists.
5. Audit semantic-context residuals with fixed composition features and
   patch-stratified ablations rather than relying on generic composition
   explanations.
6. Evaluate every addition as an ablation on validation/test central-band
   accuracy and overall `val_threshold_accuracy`/`test_threshold_accuracy`.

The tight `0.475-0.525` band contains 163,910 held-out games and sits near
52.4% accuracy. The wider `0.425-0.575` band contains 277,652 held-out games and
covers almost the whole validation/test distribution. The theoretical estimates
show enough miss-side signal volume to exceed 60% validation accuracy if the
signals are captured cleanly, but only leakage-safe ablations can turn that into
an expected production number.
