# Training Session Investigation

Date: 2026-05-20

Scope: current run in `app/ml/data/metrics.jsonl`, checkpoint `app/ml/data/best.pt`, cache metadata/arrays in `app/ml/data/cache`, current model/data/training code, and existing ML documentation.

## Executive Verdict

Do not promote this run as a performance improvement yet.

The current 4 x 21 temporal profile branch is alive and order-sensitive, but it is not a decision-grade improvement under a deployable, temporally valid setup. The best checkpoint reaches `val_loss=0.6747`, `val_auc=0.6043`, and `val_acc=57.42%` at epoch 14, but final test is only `test_loss=0.6761`, `test_auc=0.6013`, `test_acc=57.64%`, `test_brier=0.2416`, and `test_ece=0.0189`.

Matched controls strengthen the no-promotion verdict:

- observed-build identity without profiles is competitive with full temporal over seeds `42-44`;
- deterministic train-mode build recovers only a small part of the observed final-build oracle lift;
- no-delta and single-bin retrains match or beat the full four-bin encoder on seed `42`;
- removing `win_rate` improves seed-42 metrics;
- removing side embeddings preserves ranking but worsens ECE;
- pairwise support rarity is not the main residual pattern.

The remaining high-value work is deployable build-direction input and leakage-safe profile generation, not another broad architecture sweep.

## Current Run

| Item | Evidence |
| --- | --- |
| Dataset | train `1,558,207`, validation `194,776`, test `194,776` |
| Model | `2,955,778` params, `d_model=256`, `n_layers=3`, `n_heads=4`, `team_mean`, temporal profile encoder |
| Best validation | epoch `14`, `val_loss=0.6747`, `val_auc=0.6043`, `val_brier=0.2410`, `val_ece=0.0085` |
| Final test | `test_loss=0.6761`, `test_auc=0.6013`, `test_acc=57.64%`, `test_brier=0.2416`, `test_ece=0.0189` |
| Early stop | epoch `24` after 10 epochs without validation-loss improvement |
| Optimization | grad norm median `0.0416`, range `0.0257..0.1701`; no exploding/vanishing-gradient evidence |
| Prediction density | `72.0%` of test predictions are in `40-60%`; `42.1%` are in `45-55%`; only `4.6%` are outside `30-70%` |

Checkpoint perturbations show that the profile branch is used:

| Test-time condition | Loss | Accuracy | AUC | Brier | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Original profile | `0.6761` | `57.64%` | `0.6013` | `0.2416` | matches final test |
| Reversed temporal bins | `0.6786` | `57.21%` | `0.5943` | `0.2428` | temporal order is used |
| Zero profile tensor | `0.6912` | `52.66%` | `0.5345` | `0.2490` | profile branch carries substantial signal |

Checkpoint internals also show profile usage: `profile_gate=1.10`, and the zero-initialised `delta_proj` moved to weight norm `1.89`.

## Data Validity

Static lineage check:

- `participant_item_value_totals` derives `highest_value_label` from final participant item slots `item0..item6`.
- `ml_game_player_pivot` passes `highest_value_label` into each player tuple as `build`.
- `build_dataset.py` uses `build` both as an embedding input and as the join key into `synergy_1vx`.
- `synergy_1vx` is built from train participants only, but its grain is still `(champion, role, final build, bin)`.

Implication: final observed build may be a useful proxy for latent pre-game intent, but it is not directly available to the system at draft time and may contain outcome-conditioned corrections. Treat observed-build results as oracle upper bounds until a deployable build-direction policy is tested.

Train profiles are also all-train aggregates. Validation/test avoid direct self-label leakage via `WHERE split = 'train'`, but train rows can still see aggregate statistics that include themselves and later train games.

## Result Summary

Artifacts live under `app/ml/data/investigations/20260520/`.

### Build Policy And Calibration

| Check | Artifact | Result | Verdict |
| --- | --- | --- | --- |
| Train-only build priors | `build_policy_stats.json` | deterministic train-mode build matches observed build for `77.94%` of validation tokens and `76.95%` of test tokens | use `p(build | champion, role)` marginalization or a real intent model; train-mode is only a fallback |
| Calibration | `calibration_deciles.json` | intercept+scale improves test loss `0.6761 -> 0.6756`, Brier `0.2416 -> 0.2413`, ECE `0.0189 -> 0.0112` | keep calibration artifact, but decile drift remains |
| Drift | current checkpoint | train/validation/test blue rates are `49.56%` / `47.90%` / `46.81%`; test pred mean is `48.60%` uncalibrated | global intercept helps, but time drift is still real |

Profile coverage is not the bottleneck: test tokens have any profile `99.992%`, all 4 bins `99.881%`, and support median about `9,253`; confidence with `k=64` is saturated for most rows.

### Matched Runs

| Run | Seed | Best Epoch | Val Loss | Test Loss | Test AUC | Test Brier | Test ECE | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full temporal oracle-build | `42` | `14` | `0.6747` | `0.6761` | `0.6013` | `0.2416` | `0.0189` | `app/ml/data/best.pt` |
| Full temporal oracle-build | `43` | `15` | `0.6751` | `0.6762` | `0.6023` | `0.2416` | `0.0261` | `runs/full_seed43/` |
| Full temporal oracle-build | `44` | `11` | `0.6747` | `0.6759` | `0.6015` | `0.2415` | `0.0216` | `runs/full_seed44/` |
| Observed-build identity, no profiles | `42` | `14` | `0.6754` | `0.6754` | `0.6018` | `0.2413` | `0.0169` | `runs/identity_build_only_seed42/` |
| Observed-build identity, no profiles | `43` | `21` | `0.6756` | `0.6759` | `0.6016` | `0.2415` | `0.0232` | `runs/identity_build_only_seed43/` |
| Observed-build identity, no profiles | `44` | `8` | `0.6755` | `0.6759` | `0.6001` | `0.2416` | `0.0212` | `runs/identity_build_only_seed44/` |
| Draft-only no-build/no-profile | `42` | `14` | `0.6833` | `0.6828` | `0.5761` | `0.2449` | `0.0200` | `runs/draft_no_build_no_profile_seed42/` |
| Train-mode build temporal | `42` | `14` | `0.6817` | `0.6813` | `0.5831` | `0.2442` | `0.0222` | `runs/train_mode_temporal_seed42/` |
| Train-mode build identity | `42` | `14` | `0.6822` | `0.6817` | `0.5820` | `0.2443` | `0.0210` | `runs/train_mode_identity_seed42/` |

Verdict: temporal profiles are not a decision-grade win over observed-build identity, and deterministic train-mode build loses most of the observed-build oracle lift.

### Temporal And Feature Ablations

All rows below are seed `42` retrains on the current cache unless noted.

| Run | Best Epoch | Val Loss | Test Loss | Test AUC | Test Brier | Test ECE | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full temporal oracle-build | `14` | `0.6747` | `0.6761` | `0.6013` | `0.2416` | `0.0189` | `app/ml/data/best.pt` |
| No `delta_proj` | `13` | `0.6746` | `0.6760` | `0.6014` | `0.2415` | `0.0187` | `runs/no_delta_full_seed42/` |
| Single bin 1 only | `14` | `0.6746` | `0.6752` | `0.6016` | `0.2412` | `0.0176` | `runs/single_bin_1_seed42/` |
| Single bin 2 only | `14` | `0.6747` | `0.6761` | `0.6010` | `0.2416` | `0.0207` | `runs/single_bin_2_seed42/` |
| Single bin 3 only | `14` | `0.6748` | `0.6753` | `0.6018` | `0.2412` | `0.0163` | `runs/single_bin_3_seed42/` |
| Single bin 4 only | `14` | `0.6746` | `0.6751` | `0.6026` | `0.2411` | `0.0171` | `runs/single_bin_4_seed42/` |
| Prefix bins 1-2 | `13` | `0.6745` | `0.6758` | `0.6017` | `0.2414` | `0.0193` | `runs/prefix_bins_1_2_seed42/` |
| Prefix bins 1-3 | `14` | `0.6747` | `0.6761` | `0.6011` | `0.2416` | `0.0191` | `runs/prefix_bins_1_2_3_seed42/` |
| Win rate only | `14` | `0.6753` | `0.6766` | `0.5994` | `0.2418` | `0.0190` | `runs/win_rate_only_seed42/` |
| No win rate | `14` | `0.6743` | `0.6750` | `0.6030` | `0.2411` | `0.0185` | `runs/no_win_rate_seed42/` |
| No side embeddings | `10` | `0.6749` | `0.6763` | `0.6024` | `0.2417` | `0.0289` | `runs/no_side_full_seed42/` |

Verdict: the current four-bin compression adds no reliable held-out value in these seed-42 ablations. Bin `4` alone is the strongest temporal variant, `delta_proj` is not useful as-is, `win_rate` may be noisy/redundant, and side embeddings are not the ranking source but help ECE.

### Residual Probe

Artifact: `pairwise_residual_probe.json`.

| Slice, Lowest vs Highest Quartile | Low Range | High Range | Low Brier | High Brier | Low AUC | High AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lane min support | `0-67` | `615-5811` | `0.2424` | `0.2413` | `0.5972` | `0.6032` |
| Team-pair min support | `1-394` | `992-4911` | `0.2420` | `0.2418` | `0.5987` | `0.6010` |
| Bot-duo min support | `0-507` | `4091-36063` | `0.2417` | `0.2416` | `0.6012` | `0.6004` |
| Damage-mix extreme | `0.001-0.148` | `0.291-0.899` | `0.2422` | `0.2403` | `0.5976` | `0.6088` |
| Frontline gap | `0.000-0.326` | `1.176-4.898` | `0.2423` | `0.2408` | `0.5979` | `0.6057` |

Verdict: support rarity is not strong enough to justify broad pairwise feature expansion before build-policy validity is fixed. Composition slices are more promising for monitoring or future summary features.

## Root Causes

| Area | Evidence | Current Status |
| --- | --- | --- |
| Build direction | final item slots define `build`; train-mode build matches only `76.95%` of test tokens | unresolved; observed-build runs are oracle upper bounds |
| Profile causality | `synergy_1vx WHERE split = 'train'` protects validation/test but train rows see all-train aggregates | unresolved; OOF/causal cache not tested |
| Temporal compression | `[10, 4, 22]` collapses to one vector per player before match attention | current design not justified; single-bin/no-delta variants are competitive |
| Feature redundancy | inference says `win_rate` is influential, but no-`win_rate` retrain improves seed-42 metrics | remaining grouped ablations needed |
| Side symmetry | side embeddings break full-input blue/red complementarity; no-side worsens ECE | scalar side-bias remains untested |
| Calibration drift | test blue rate is `46.81%`, uncalibrated pred mean is `48.60%` | validation calibration helps global metrics but not time drift |
| Pairwise context | profile grain is marginal, not pairwise | residual probe does not justify pairwise expansion yet |
| Optimization | stable gradients and normal early stopping | lower priority than data validity and profile/input controls |

## Remaining Work

Keep these items because they are not yet tested or are only partially tested.

| Priority | Work | Status |
| --- | --- | --- |
| Highest | Marginalized `p(build | champion, role)` policy or real build-intent model | not tested; deterministic train-mode fallback performs poorly |
| Highest | OOF or causal profile cache | not performed; requires rebuilding profile inputs |
| High | Previous scalar-profile baseline rebuilt on this exact cache | not tested; requires compatible historical representation or rebuild |
| High | Calibration reported for every future deployable-input variant | current oracle-build checkpoint only |
| Medium | Scalar side-bias control | not tested; no-side control alone is not enough because ECE worsened |
| Medium | Level-only vs delta-only temporal retrains | not tested |
| Medium | Reversed-bin smoke tests for every retrained temporal variant | not tested |
| Medium | Grouped retrained ablations: economy, combat/threat, damage shares, survivability, utility/objective/vision, no `avg_item_completions` | not tested |
| Medium | Role/bin-specific normalization or grouped profile encoders | not tested |
| Medium | Simple late-bin or no-`win_rate` profile baseline under deployable build policy | suggested by seed-42 results, not tested under deployable build inputs |
| Conditional | Pairwise/synergy feature probe | not justified by current residual probe; reopen only with stronger residual evidence |
| Conditional | Confidence-prior sweep | support slices do not show low-support harm; remains conditional |
| Conditional | `team_attention`, explicit team-tempo/head features, per-player sequence encoder, or separate temporal tokens | not tested; wait for deployable-input and scalar-baseline verdicts |
| Evaluation | Hard-label train monitor, split hash, feature-order hash, profile-source hash, profile max timestamp | not implemented here |
| Monitoring | Prediction mean, ECE/Brier, support/missing-bin rates, feature drift, calibration intercept drift by time window | production checklist, not implemented here |

## Stop/Go Criteria

Go to the next training iteration only if:

- build-direction inference/proxy validity is resolved beyond deterministic train-mode fallback;
- a deployable-input temporal/profile variant beats identity/profile-disabled and scalar controls on matched current-cache seeds;
- calibrated Brier/ECE do not regress;
- chronological deciles, support slices, and central prediction bands do not degrade materially.

Stop and redesign data if:

- marginalized build loses most of the observed-build oracle gain, as deterministic train-mode already does;
- profile-disabled control continues to match full temporal after deployable build handling is fixed;
- no-side or scalar-side control matches full temporal while improving calibration;
- temporal ablations continue to show one noisy bin or group dominating without stable held-out gain;
- calibration keeps improving global metrics but not time-decile probability drift.

## Final Recommendation

Freeze the current checkpoint as an active oracle-build temporal baseline, not a promoted deployable model.

The next training iteration should keep the architecture mostly unchanged and focus on:

1. marginalized build-direction variant or a real build-intent model;
2. causal or out-of-fold profile cache;
3. previous scalar-profile baseline rebuilt on this cache;
4. scalar-side-bias control if side handling remains under review;
5. validation-fitted calibration reported for each future variant.

Only after those verdicts should architecture work resume. Start with simple late-bin or no-`win_rate` profile baselines, scalar side-bias, and explicit team-tempo/head features before broader capacity changes such as `team_attention`, grouped encoders, or separate temporal tokens.
