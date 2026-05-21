# Temporal/Profile Data Usage Audit

Transformer over 10 fixed player tokens. Current model input includes champion, role, build, side, and `player_profile` shaped `[B, 10, 4, 22]`: four temporal/scaling bins, 21 content profile metrics, and `log_matchups` as support confidence. Current hyperparameters are in [README.md](README.md); experiment evidence is in [INVESTIGATION.md](INVESTIGATION.md) and [AUDIT.md](AUDIT.md).

## Executive Summary

Confirmed: the profile branch is wired into the model and materially affects predictions. The current checkpoint has profile encoder weights, `profile_gate=1.1137`, and `delta_proj` moved off zero-init with weight norm `2.0365`. Test-time perturbations from the investigation show original profile test AUC about `0.6013`, reversed temporal bins about `0.5943`, and zero profile tensor about `0.5345`. That means the branch is alive and order-sensitive.

Confirmed: richer input is not yet a proven improvement over simpler controls. Observed-build identity without profiles is competitive across seeds `42-44`, and seed-42 temporal ablations show no reliable win for the full four-bin representation over single-bin or no-delta variants. The current local `best.pt` records epoch `14`, `val_loss=0.6751`, and final test around `test_loss=0.6762`, `test_auc=0.6012`, `test_brier=0.2416`, `test_ece=0.0192`.

Highest-severity caveat: final observed build is used as both an embedding input and as the join key into historical profiles. If prediction happens at draft or pre-game time, this is an oracle input, not a deployable feature. Train profiles also use all train aggregates, so train rows can see themselves and later train games through profile statistics.

## Data Path Review

1. Feature construction: `build_dataset.py` streams games from `game_data_filtered.ml_game_player_pivot`, expands fixed slots `0..4` blue and `5..9` red, and joins `game_data_filtered.synergy_1vx WHERE split = 'train'` by `(championid, teamposition, build)`. The source profile bins `1..4` are stored as indices `0..3`.

2. Cache shape and dtype: `player_profile.npy` is written as `(games, 10, 4, 22)` in `float16`. The 22 fields are listed in `PROFILE_FEATURE_COLUMNS`; `log_matchups` is support, and the other 21 fields are content metrics.

3. Missing values: missing profile rows are zero-filled in the cache. The model computes `present = any_nonzero_feature(player_profile)` before standardization and masks absent bins from attention pooling and early/late delta.

4. Scaling: `profile_standardization` is computed over present train profile rows only and stored in `cache_meta.json`. The model loads mean/std as buffers and standardizes the 21 content fields globally. `log_matchups` is excluded from the content MLP and used raw for confidence.

5. Tensor path: `player_profile [B, 10, 4, 22] -> content [B, 10, 4, 21] -> shared per-bin MLP [B, 10, 4, d] -> learned bin-position embedding -> confidence-weighted attention pool [B, 10, d]`.

6. Temporal compression: the four bins are collapsed to one profile vector per player before the 10-token match transformer. The only explicit trajectory feature is `late - early`, where bins `0..1` are early and `2..3` are late.

7. Token composition: `champ_emb + role_emb + build_emb + side_emb + profile_gate * profile_token`, then `LayerNorm`, dropout, and a pre-norm transformer encoder.

8. Output head and loss: blue and red token groups are pooled separately, same-role lane differences are attention-pooled, and the head receives `concat(b, r, b-r, abs(b-r), b*r, lane)`. The same head is applied in both orientations, and final logit is `score_bvr - score_rvb`. Training uses `BCEWithLogitsLoss` with smoothed targets; evaluation uses hard labels.

## Temporal Preservation Assessment

Confirmed preserved:

- Time-step identity exists through learned bin-position embeddings.
- Temporal order matters empirically: reversing bins hurts held-out performance.
- Early vs late direction is represented by the explicit `late - early` delta.
- Missing temporal halves are guarded: the delta is zero unless both early and late halves have present bins.

Confirmed weakened or compressed:

- The temporal axis is collapsed before cross-player self-attention. The match transformer cannot directly attend from "blue bin 1" to "red bin 4"; it only sees one fused player vector.
- Trends are coarse. There is no adjacent-bin delta, slope, acceleration, decay curve, or monotonic/ordinal inductive bias.
- The current delta path is not justified by results. Seed-42 no-delta retraining is effectively tied with or slightly better than the full model.
- Single-bin retrains are competitive; bin 4 alone produced the best seed-42 temporal ablation in the investigation. This argues that the model may be using one strong profile summary more than the intended four-step temporal structure.

Hypothesis:

- The learned attention pool may mostly select high-signal bins rather than learn a stable trajectory. This needs bin-attention logging or attribution, not just architecture inspection.

## Profile Preservation Assessment

Confirmed preserved:

- Individual metric order is fixed by `PROFILE_FEATURE_COLUMNS`; the model validates `n_features == len(PROFILE_FEATURE_COLUMNS)`.
- Content metrics keep absolute scale after global train mean/std standardization. A per-row layer norm is not used, so metric levels are not erased inside each profile row.
- Cross-metric interactions can be learned by the shared MLP and downstream transformer/head.
- `log_matchups` is not allowed to become a direct popularity/content feature; it only modulates reliability.

Confirmed weakened or unproven:

- Metric group structure is not encoded. Economy, combat, damage-share, objective, vision, durability, and win-rate metrics are all mixed by one shared MLP.
- Normalization is global across roles and bins. Role-coded metrics can dominate or duplicate role/build embeddings instead of representing player/profile quality.
- Sparse signals are mostly not sparse in the current cache: coverage is very high and support confidence is saturated for most rows with `profile_confidence_prior_count=64`. The confidence gate mainly affects a small tail.
- High-impact metrics are not yet stable. The investigation found `win_rate only` underperforms, while `no win_rate` improves seed-42 metrics, so the apparent profile signal may be redundant or noisy.
- Redundant metric families remain. Prior audit evidence found strong correlations among damage shares, economy, objective, vision, and survivability metrics.

## Signal Loss Points

1. Temporal bottleneck: `[4, 22]` per player is compressed to one `[d]` vector before team/player interaction. This is the main architecture-level information bottleneck.

2. Coarse trajectory: `late - early` can preserve a broad scaling direction but loses bin-local shape, adjacent changes, acceleration, and decay.

3. Shared per-bin MLP: sharing is parameter-efficient and keeps bin semantics comparable, but it can collapse bin-specific meaning unless the learned bin embedding and attention scores carry enough distinction.

4. Global standardization: role/bin-specific distributions are mixed. This can turn profile values into role or duration proxies and can under-scale rare but meaningful deviations.

5. Missing encoding: zero-filled rows are masked correctly, but absence itself is not separately embedded as a learned signal beyond returning a zero profile contribution.

6. Confidence saturation: support median is high, so `confidence = n / (n + 64)` is near one for most rows. Reliability weighting is unlikely to explain most prediction behavior.

7. Oracle build path: final observed build carries large signal and can reduce the incremental value of temporal/profile inputs. It also risks leakage if unavailable at prediction time.

8. Train profile causality: `synergy_1vx WHERE split = 'train'` protects validation/test from direct self-label profile leakage, but train examples can still use aggregate rows containing themselves and future train games.

9. Regularization/optimization: no current evidence of gradient failure; investigation grad norms were stable. Dropout and AdamW may still suppress small profile deltas, but this is a hypothesis.

## Critical Diagnostics

Already run or documented:

- Test-time zero-profile ablation: confirms the profile branch carries signal.
- Test-time reverse-bin ablation: confirms temporal order affects predictions.
- Matched observed-build identity control: shows profiles are not a decisive win over identity/build inputs.
- Train-mode build controls: show deterministic deployable build fallback loses most observed-build oracle lift.
- Single-bin and prefix-bin retrains: show the full four-bin temporal encoder is not yet justified.
- No-delta retrain: shows the explicit trajectory delta is not currently valuable.
- `win_rate only` and `no win_rate` retrains: show metric utility is non-obvious and likely redundant/noisy.
- Calibration and chronological decile checks: show global calibration helps but time drift remains.

Highest-value remaining checks:

- Leakage gate: define prediction time, remove or marginalize final observed build, and assert feature timestamps are prior to prediction timestamps.
- Out-of-fold or causal profile cache: every train row should see only past or fold-external profile aggregates.
- Previous scalar baseline on the exact current cache/split, with matched seeds.
- Permutation importance by `time x metric`, grouped into profile families, using the same checkpoint and held-out split.
- Gradient/input attribution by `time x metric` on held-out batches, reported as heatmaps and group sums.
- Representation similarity across bins before pooling to test whether bin representations collapse.
- Mask/remove one time bin and one metric group at evaluation and retrain time; prefer small paired tests over broad sweeps.

## Highest-Impact Fixes

1. Resolve build leakage first. Replace final observed build with no-build, train-only `p(build | champion, role)` marginalization, or a real pre-game build-intent model.

2. Rebuild profiles causally or out-of-fold. This is required before trusting training improvements from profile features.

3. Re-run matched controls on the same cache: scalar baseline, identity/build-only, profile-disabled, temporal full, no-delta, and strongest single-bin/no-`win_rate` candidates.

4. Add durable metadata checks: feature-order hash, split matchid hash, profile source query/hash, and profile source max timestamp.

5. Add profile diagnostics to training/evaluation output: `profile_gate`, `delta_proj` norm, bin attention distributions, confidence quantiles, bin coverage, and grouped permutation importance.

## Recommended Architecture Changes

Near-term conservative changes:

- Keep the current encoder as an oracle-build baseline, but do not promote it as the final temporal architecture.
- Add an explicit flattened or engineered-feature control: per-metric mean, early, late, late-minus-early, adjacent deltas, and slope.
- Try a level-only encoder and a late-bin/no-`win_rate` variant under deployable build handling before increasing capacity.
- Add role/bin-conditioned normalization or role-conditioned affine parameters.
- Group profile metrics by family before fusion, so redundant families can be isolated and ablated cleanly.

Conditional changes after data validity is fixed:

- Represent each player as separate temporal tokens, or use a tiny per-player sequence model over four bins before player fusion.
- Preserve separate outputs for level, early strength, late strength, and slope instead of forcing all temporal evidence into one vector.
- Add explicit team-composition summaries such as damage mix and frontline/backline features only if residual slices continue to justify them.

## Next Experiment

Run a deployable-input, leakage-gated comparison before more architecture work:

1. Build a no-final-build or marginalized-build cache/input path.
2. Build out-of-fold or causal profile aggregates.
3. Train matched seeds `42, 43, 44` for identity-only, previous scalar/profile baseline, current full temporal, no-delta, bin-4-only, and no-`win_rate`.
4. Select by paired validation loss and Brier, then confirm on test with calibration and chronological decile metrics.
5. Promote only if the temporal/profile arm beats identity/scalar controls under deployable inputs without ECE/Brier or time-slice regressions.
