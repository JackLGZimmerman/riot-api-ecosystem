# Temporal Profile Model Audit

Date: 2026-05-20

Scope: `app/ml/model.py`, `app/ml/build_dataset.py`, `app/ml/dataset.py`, `app/ml/config.py`, ML documentation, current cache metadata/arrays, and the SQL that creates the split, build labels, and `synergy_1vx` profile table.

## Executive Judgement

**No-go for claiming the new representation is beneficial yet. Go for offline experiments only.**

The new 4 x 21 profile representation is promising but unproven. It is correctly wired into the model as a structured temporal profile encoder rather than a blind flatten, and the current cache is dense, finite, and highly covered. A simple confidence-weighted team win-rate prior from the new cache has held-out signal (`val AUC=0.5909`, `test AUC=0.5896`), and the 4-bin aggregate beats every individual bin on test.

However, there is no trained current-cache temporal model metric to compare against the previous single-scalar/pre-temporal baseline. The only live `best.pt`/`metrics.jsonl` I found is from 2026-05-19, before the May 20 temporal cache rebuild; its checkpoint has no `profile_encoder` state keys and `Vocab(profile_mean=(), profile_std=())`. That older run is roughly `test_loss=0.6753`, `test_auc=0.6019`, `test_acc=0.5754`, `test_brier=0.2413`, `test_ece=0.0173`, but it used `test_n=167,822` while the current cache has `test_n=194,776`. Existing documentation also marks earlier sweep evidence as stale for the new temporal encoder.

Most importantly, if the production prediction point is draft/pre-game, the current input still contains a critical post-outcome leakage path: `build` is derived from the current match's final item slots, then used as both an embedding input and the key into historical profile rows. The training profiles also include train rows themselves and later train games. Validation/test avoid direct label leakage via `WHERE split = 'train'`, but the training objective is not causal and can overfit rare profile rows.

## Evidence Summary

- Current cache shape is `[1,947,759, 10, 4, 22]`: 4 profile bins, 21 content metrics, plus `log_matchups` as confidence.
- Split sizes: train `1,558,207`, validation `194,776`, test `194,776`.
- Label drift is non-trivial: blue win rate is `49.56%` train, `47.90%` validation, `46.81%` test. This makes calibration-by-time mandatory.
- Profile coverage is excellent:
  - Test token with any profile: `99.992%`.
  - Test token with all 4 bins: `99.881%`.
  - Test game with all 10 tokens having any profile: `99.922%`.
  - No NaN or Inf values found in the current cache.
- Support is mostly high:
  - Test support median from `expm1(log_matchups)` is about `9,253`.
  - Only `1.61%` of test profile rows have support below `64`.
  - With `profile_confidence_prior_count=64`, test median confidence is `0.993`; the reliability gate is saturated for most rows.
- Simple profile-prior signal:
  - Confidence-weighted all-bin win-rate team difference AUC: train `0.5972`, val `0.5909`, test `0.5896`.
  - Test per-bin AUCs: bin 1 `0.5690`, bin 2 `0.5815`, bin 3 `0.5747`, bin 4 `0.5454`.
  - Interpretation: the aggregate temporal profile contains signal, but late-bin win-rate is weak and the prior alone is still below the old full-model AUC band around `0.602`.

## Feature Representation Findings

The model uses the 4 temporal dimensions, but only after compressing them into one vector per player. `_TemporalProfileEncoder` standardizes the 21 non-count metrics by train mean/std, applies a shared per-bin MLP, adds learned bin-position embeddings, masks zero-filled rows, attention-pools the bins, and adds an explicit late-minus-early delta. This is a reasonable first-pass structured encoder and is materially better than treating 84 values as an unordered flat vector.

The temporal structure is still limited. The 4-bin axis is collapsed before cross-player self-attention, so the match transformer cannot directly compare "blue early profile vs red late profile" token-by-token. It only sees the pooled level plus a coarse `[bins 3-4] - [bins 1-2]` delta. The model can distinguish bins through position embeddings, but there is no ordinal constraint, no adjacent-bin deltas, and no telemetry proving the attention pool or `delta_proj` learned a useful temporal decomposition.

The representation stores 22 fields, not just 21: `log_matchups` is correctly excluded from the content MLP and used only for confidence. That design prevents raw popularity/support from becoming a direct content signal. But the current confidence prior is too small for the observed support distribution: with median support near 9k, `k=64` changes almost nothing outside the rare tail.

Scaling and missing values are mostly handled correctly. Standardization is computed only over present train profile rows, zero-filled missing bins are masked before standardization, and the model has a finite CPU forward pass with `2,955,778` parameters. The risk is that standardization is global across all roles and bins. Several metrics are strongly role-coded, so the profile stream may duplicate role embeddings instead of learning performance quality.

## Redundancy And Noise Findings

Top sampled absolute correlations among the 21 content metrics show heavy redundancy:

| Pair | Correlation | Concern |
| --- | ---: | --- |
| `physical_damage_share` vs `magic_damage_share` | `-0.985` | compositional shares nearly determine each other |
| `avg_total_cs` vs `avg_vision_score` | `-0.957` | role confounding, not pure quality |
| `avg_gold` vs `avg_kills` | `0.910` | snowball/outcome proxy |
| `avg_gold` vs `avg_total_cs` | `0.906` | economy duplicate |
| `avg_epic_monster_takedowns` vs `avg_damage_to_objectives` | `0.903` | objective duplicate |
| `avg_damage_taken` vs `avg_durability` | `0.882` | survivability duplicate |
| `avg_vision_score` vs `avg_control_wards_bought` | `0.874` | vision duplicate |

Temporal correlations also show that many metrics are almost deterministic offsets across bins:

- `avg_vision_score` adjacent-bin correlations are `0.9992`, `0.9997`, `0.9995`.
- `avg_xp`, `avg_gold`, `avg_damage_taken`, `avg_total_damage_dealt`, and `damage_to_taken_ratio` are mostly `0.96+` from bin 1 to bin 4.
- `avg_item_completions` is essentially a restatement of the bin definition, with mean bin 4 minus bin 1 delta `2.24`.
- `win_rate` is the main metric with meaningfully different temporal behavior: bin 1 vs bin 4 correlation is only `0.126`, and the all-bin aggregate beats any single bin.

Conclusion: the new representation is rich, but much of the 21-metric payload is redundant, role-coded, or duration/snowball-coded. The useful temporal signal appears concentrated in outcome prior and a smaller set of scaling-sensitive metrics.

## Leakage And Temporal Consistency Findings

1. **Final build leakage is the highest-severity issue.** `participant_item_value_totals` derives `highest_value_label` from `item0..item6` in final participant stats, and `ml_game_player_pivot` passes that `build` into the model and profile join. If predictions are made before final builds are known, this is post-outcome information.

2. **Profile rows are not causal for train examples.** `synergy_1vx` is built from all train games, then `build_dataset.py` joins every split to `synergy_1vx WHERE split = 'train'`. Validation/test do not leak their own labels into profiles, but train rows see aggregate statistics that include themselves and future train games.

3. **The profile is conditioned on game duration/survival.** The 4 bins are assigned using historical game duration and final legendary item counts. That can be useful as a historical scaling profile, but it is not the same as a causal feature available at draft time.

4. **Current chronological split is good but insufficient.** The split is ordered by timestamps, which is the right validation direction. But because train profiles are all-train aggregates, the training side is still non-causal; use cumulative or out-of-fold profile generation.

5. **Train/test contamination checks are incomplete.** Cache metadata records counts and format, but not a stable split hash, profile source max timestamp, or feature-order hash. Those are needed for production-grade temporal guarantees.

## Model Suitability

The current architecture is suitable as an initial structured baseline for 4 x 21 tabular temporal inputs, but it is not yet a strong temporal model.

Keep this model as the baseline candidate because it has sensible mechanics: shared per-bin MLP, bin position embeddings, confidence gating, missing-bin masking, and antisymmetric blue/red scoring.

Do not assume it is the final architecture. The next architecture should compare:

- Current attention-pool plus early/late delta.
- Flattened 4 x 21 projection as a negative control.
- Engineered deltas and slopes in raw feature space.
- Tiny sequence model over 4 bins per player.
- Temporal attention with separate outputs for level, early strength, mid-game strength, late strength, and scaling slope.
- Grouped encoders by feature family, with role/bin-specific normalization.
- Dimensionality reduction or feature pruning for correlated metric groups.

## Performance Impact Assessment

The current evidence cannot establish improvement over the previous single-scalar/pre-temporal baseline.

Known baseline band from available runs:

- Live old checkpoint: `test_loss=0.6753`, `test_auc=0.6019`, `test_acc=57.54%`, `test_brier=0.2413`, `test_ece=0.0173`.
- Prior documented promoted schedule run: `test_loss=0.67504`, `test_auc=0.60273`, `test_acc=57.60%`, `test_brier=0.24114`, `test_ece=0.0176`.
- MoE dense final run: `test_loss=0.67490`, `test_acc=57.51%`, with no evidence it used the current temporal cache.

Current temporal representation evidence:

- No trained current-cache temporal run found.
- Profile-prior-only AUC is `0.5896` on current test, which proves signal but does not beat the full old model.
- Current cache test set differs from old metrics (`194,776` vs `167,822`), so raw old/new comparisons would be invalid unless both models are rerun on the same cache and split.

## Top 10 Risks

1. Final item-build label leaks post-prediction information if production prediction happens before final builds are known.
2. Train profiles include each train row and future train games, creating non-causal train signal.
3. No current trained temporal metrics exist, so the representation may harm performance despite looking richer.
4. Confidence prior is saturated; low-support reliability logic affects only a small tail.
5. Global normalization lets role identity and duration patterns dominate profile metrics.
6. High metric redundancy increases overfit risk and makes importance unstable.
7. Temporal axis is compressed before team/player interactions, limiting sequence learning.
8. Late-bin win-rate prior is weak (`test AUC=0.5454`) and may inject noise.
9. Label/base-rate drift across chronological splits can cause calibration drift.
10. Evaluation currently lacks support, cohort, temporal, and edge-case slices needed to catch regressions.

## Top 10 Improvement Opportunities

1. Replace final-build inputs with features available at the prediction point, or explicitly split the product into "draft-only" and "build-known" models.
2. Build causal or out-of-fold historical profiles so every train row sees only past or fold-external data.
3. Rerun the previous scalar baseline and the new temporal model on the exact May 20 cache with matched seeds.
4. Add explicit ablation switches for profile gate, bins, metric groups, confidence, delta, and temporal order.
5. Add role/bin-specific normalization and compare it against global normalization.
6. Add empirical-Bayes shrinkage toward champion/role/global priors instead of shrink-to-zero profile vectors.
7. Prune or transform redundant groups, especially damage shares and duplicated economy/objective/vision metrics.
8. Log `profile_gate`, `delta_proj` norm, bin attention weights, confidence distributions, and grouped permutation importance.
9. Add segmented validation by support, bin coverage, role, build, champion frequency, time decile, and temporal archetype.
10. Compare sequence-aware and grouped encoders against the current pooled temporal encoder.

## Prioritized Experiment Plan

### P0: Leakage Gate

Define the exact production prediction time. If final items are not known, run a no-build or pre-game-build model before any promotion. Also add metadata checks that profile source timestamps are earlier than prediction timestamps.

### P1: Matched Baseline Rebuild

On the current May 20 cache, run at least seeds `42, 43, 44` for:

- Previous single-scalar representation.
- Current 4 x 21 temporal representation.
- Identity-only baseline: `profile_gate=0`.
- Profile-only or profile-dominant probe.
- No-build model.

Select by paired validation loss, then confirm on test. Require no regression in Brier/ECE and no material slice regression.

### P2: Temporal Ablation Matrix

Run targeted ablations to determine whether the temporal axis helps:

- Keep only bin 1, only bin 2, only bin 3, only bin 4.
- Prefix bins: bins 1, 1-2, 1-3, all 4.
- Remove each bin from the full model.
- Reverse bin order.
- Shuffle bin order per sample at evaluation.
- Collapse bins by confidence-weighted average and compare with sequence encoder.
- Disable `delta_proj`.
- Keep delta only, remove attention level.

### P3: Metric Group Ablation Matrix

Remove or isolate groups:

- `win_rate` only.
- No `win_rate`.
- Economy only.
- Threat/combat only.
- Damage shares only, and damage shares removed.
- Survivability only.
- Utility/objective/vision only.
- No `avg_item_completions`.
- No `log_matchups` confidence weighting.
- Confidence prior sweep: `k in {16, 64, 256, 1024, 4096}`.

### P4: Architecture Sweep

Compare current encoder against:

- Flattened 84-value projection.
- Raw engineered features: per-metric mean, early, late, late-early, adjacent deltas, slope.
- Tiny GRU/TCN/Transformer over 4 bins.
- Grouped encoders by metric family.
- Role-specific profile encoders or role-conditioned normalization.

### P5: Production Monitoring

Before launch, create dashboards for feature drift, calibration drift, segment degradation, profile support, missing-bin patterns, and feature-importance shifts.

## Recommended Metrics

Use paired, split-consistent metrics:

- Primary: validation/test log loss, Brier score, ROC-AUC.
- Calibration: ECE, adaptive-bin ECE, calibration intercept/slope, reliability curve.
- Decision quality: accuracy at 0.5, lift by confidence decile, top/bottom confidence hit rate.
- Stability: mean and standard deviation over matched seeds, paired deltas per seed.
- Statistical tests: paired bootstrap over matches for loss/Brier/accuracy, DeLong or bootstrap for AUC, McNemar for thresholded accuracy, and seed-paired t-test only as secondary evidence.
- Segment metrics: the same metrics by time decile, role, build, champion frequency, support quantile, bin coverage pattern, temporal slope archetype, and confidence bucket.

## Recommended Plots

- Learning curves: train vs validation loss/Brier/AUC/ECE.
- Reliability diagrams globally and by chronological test decile.
- Prediction probability histograms by split.
- Per-bin attention weight distributions and entropy.
- `profile_gate` and `delta_proj` norm over training.
- Grouped permutation importance with confidence intervals.
- Ablation waterfall: full model minus each time bin and metric group.
- Heatmaps by support quantile x build frequency and role x temporal archetype.
- Drift plots for feature means/stds, missing rates, support, and bin-presence rates.
- Cumulative chronological test performance to detect patch/time degradation.

## Production Checks

- Assert input shape `[B, 10, 4, 22]` and exact feature order.
- Store a feature-order hash and cache format in every checkpoint.
- Store split matchid hash and profile-source query/hash in `cache_meta.json`.
- Verify train/validation/test matchid disjointness.
- Verify profile source max timestamp is earlier than prediction timestamp.
- Track missing profile bins by role/build/champion.
- Track support quantiles and confidence quantiles.
- Alert on PSI/KS drift for every metric group.
- Track calibration and Brier by time window.
- Monitor grouped feature importance and temporal-bin attention shifts.

## Final Verdict

The new representation is technically plausible and data-rich, and the cache-level diagnostics show that it contains real held-out signal. But it is not production-ready evidence-wise. The correct judgement today is:

**No-go for promotion or for claiming performance improvement.**

Promotion should require a matched, multi-seed current-cache experiment against the previous scalar baseline, plus a leakage-safe build-input story. If final build labels are not available at prediction time, the current model should be treated as leaky regardless of its offline score.
