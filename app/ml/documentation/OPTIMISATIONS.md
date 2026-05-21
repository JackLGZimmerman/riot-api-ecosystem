# ML Optimisations

Maintenance: update this file with decision-grade experiment evidence only: protocol, compact table, conclusion, and the recommended setting. Keep reusable testing procedure in `TESTING.md` and current defaults in `README.md`.

## Optimizer Default

The 2026-05-13 optimizer evidence tuned Lion against itself but did not include a matched AdamW control. With no decision-grade evidence that Lion improves held-out metrics, and with PyTorch AdamW offering a fused CUDA implementation without a third-party package, the active default is AdamW. Base LR / schedule values come from the 2026-05-18 LR-schedule sweep (see below):

```text
optimizer = "adamw"
lr = 2e-4
weight_decay = 5e-3
adamw_betas = (0.9, 0.999)
```

Recommendation: use fused PyTorch AdamW on CUDA and keep Lion retired unless a fresh isolated sweep beats this AdamW baseline on validation loss first and does not regress samples/s.

<!-- adamw-lower-lr-150epoch-20260515:start -->
## AdamW Lower Learning Rate 150-Epoch Sweep, 2026-05-15

> **Stale architecture (2026-05-20):** trials predate the temporal profile encoder and the `lane` head feature now in `model.py`. Superseded for LR selection by the 2026-05-18 sweep below; retained only for the schedule-knob sensitivities, which should be re-confirmed against the current architecture.

Status: `completed`. Run directory:

```text
/home/jack/projects/riot-api-ecosystem/app/ml/data/checkpoints/adamw_lower_lr_150epoch_20260515_0630
```

Protocol: isolated fresh Python process per trial, seed `42`, `epochs=150`, current model/training defaults except LR, BF16 AMP, fused AdamW, `torch.compile(mode="reduce-overhead")`, heavy attention diagnostics and TensorBoard disabled, final test evaluation from each trial's best validation-loss checkpoint.

Memory gate: parent runner requires at least `8.0 GiB` available system memory and `8.0 GiB` free GPU memory before starting each trial.

Learning rates:

```text
5.00e-5, 4.00e-5, 3.20e-5, 2.50e-5, 2.00e-5, 1.60e-5, 1.25e-5, 1.00e-5, 8.00e-6, 6.40e-6, 5.00e-6, 4.00e-6, 3.20e-6, 2.50e-6
```

| lr | status | epochs | best epoch | val loss | val AUC | test loss | test AUC | test acc | median samples/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.00e-5 | completed | 150 | 111 | 0.675675 | 0.601425 | 0.675649 | 0.600723 | 0.573685 | 183,879 |
| 4.00e-5 | completed | 150 | 111 | 0.675760 | 0.601502 | 0.675673 | 0.601119 | 0.574406 | 184,180 |
| 3.20e-5 | completed | 150 | 111 | 0.676041 | 0.600974 | 0.675866 | 0.600999 | 0.574519 | 184,181 |
| 2.50e-5 | completed | 150 | 111 | 0.676492 | 0.599908 | 0.676229 | 0.600351 | 0.573894 | 184,194 |
| 2.00e-5 | completed | 150 | 111 | 0.677016 | 0.598591 | 0.676695 | 0.599360 | 0.573483 | 184,176 |
| 1.60e-5 | completed | 150 | 111 | 0.677623 | 0.597036 | 0.677247 | 0.598110 | 0.572857 | 184,184 |
| 1.25e-5 | completed | 150 | 111 | 0.678329 | 0.595264 | 0.677923 | 0.596517 | 0.571707 | 184,186 |
| 1.00e-5 | completed | 150 | 121 | 0.678928 | 0.593893 | 0.678536 | 0.595196 | 0.569937 | 184,184 |
| 8.00e-6 | completed | 150 | 121 | 0.679505 | 0.592404 | 0.679094 | 0.593699 | 0.569031 | 184,184 |
| 6.40e-6 | completed | 150 | 121 | 0.680115 | 0.590818 | 0.679662 | 0.592138 | 0.568185 | 184,184 |
| 5.00e-6 | completed | 150 | 121 | 0.680838 | 0.588870 | 0.680352 | 0.590162 | 0.567524 | 184,184 |
| 4.00e-6 | completed | 150 | 129 | 0.681530 | 0.586946 | 0.681067 | 0.588153 | 0.566076 | 184,182 |
| 3.20e-6 | completed | 150 | 129 | 0.682241 | 0.584655 | 0.681770 | 0.585736 | 0.564890 | 184,180 |
| 2.50e-6 | completed | 150 | 129 | 0.683103 | 0.581748 | 0.682647 | 0.582573 | 0.563329 | 184,180 |

Conclusion: the current recommended setting from this sweep is `lr=5.00e-5`, selected by lowest validation loss (`0.675675`). **Superseded by the 2026-05-18 LR-schedule sweep below**, which paired a higher base LR (`2e-4`) with an aggressive heavy-tail schedule and beats this on every test metric while cutting wall time ~44%.
<!-- adamw-lower-lr-150epoch-20260515:end -->

<!-- lr-schedule-20260518:start -->
## LR Schedule Tuning, 2026-05-18

> **Stale architecture (2026-05-20):** ran before the temporal profile encoder and `lane` head feature were added to `model.py`. The schedule knobs and their durable sensitivities are likely still directionally valid, but re-baseline the exact values against the current architecture before relying on them.

Status: `completed` and promoted. Run directories: `app/ml/data/checkpoints/{iter,baseline_multiseed,final}/`.

Protocol: one-knob-at-a-time iter sweeps over `lr`, `lr_center_epoch`, `lr_sharpness`, `lr_tail_strength`, each via `sweep one --phase iter`. Fixed `epochs=40`, `early_stop_patience=0`, all other live defaults. Best-of-run val_loss recorded; winner re-validated against baseline with paired multi-seed (42/43/44), then promoted via `sweep final --epochs 60 --patience 20` with the full diagnostic suite for the head-to-head test comparison.

### Iter knob results (single seed=42)

| Run | Config (vs defaults) | best val_loss | best ep | notes |
| --- | --- | ---: | ---: | --- |
| baseline | defaults: lr=5e-5, center=20, sharp=4.0, tail=0.3 | 0.67558 | 45 | stable, ~0.6757 plateau |
| iter/r1 | lr=2e-4, center=10, sharp=4.0, tail=0.5 | 0.67503 | 11 | post-peak oscillates ~0.6766 |
| iter/r2 | lr=2e-4, center=10, sharp=8.0, tail=0.5 | **0.67500** | 11 | tighter ~0.6758 plateau |
| iter/r3 | r2 + center=12 | 0.67513 | 11 | higher LR at peak (1.87e-4 vs 1.56e-4), no gain |
| iter/r4 | r2 + tail=0.3 | 0.67504 | 11 | noisier plateau (~0.6763) than r2 (~0.6757) |

### Paired multi-seed validation (r2 config vs baseline, same seed in both arms)

| seed | baseline vloss | iter vloss | Δ (iter − baseline) |
| --- | ---: | ---: | ---: |
| 42 | 0.67544 | 0.67500 | −0.00045 |
| 43 | 0.67584 | 0.67562 | −0.00022 |
| 44 | 0.67590 | 0.67576 | −0.00014 |
| **mean** | **0.67573** | **0.67546** | **−0.00027** |

Iter wins 3/3 paired seeds (sign-consistent). Paired t ≈ −2.9 (df=2). Baseline seed std = 2.0e-4; iter seed std = 3.3e-4.

### Final-run test comparison (seed=42, `sweep final --epochs 60 --patience 20`)

| Metric | Baseline final | Iter final | Δ |
| --- | ---: | ---: | ---: |
| test_loss | 0.67523 | **0.67504** | −0.00019 |
| test_accuracy | 0.5753 | **0.5760** | +0.0007 |
| test_auc | 0.60253 | **0.60273** | +0.0002 |
| test_brier | 0.24123 | **0.24114** | −0.00009 |
| test_ece | 0.0194 | **0.0176** | −0.0018 (better calibration) |
| wall_s | 457 | 258 | **−44%** |

### Conclusion

Promoted: `lr=2e-4`, `lr_center_epoch=10`, `lr_sharpness=8.0`, `lr_tail_strength=0.5`. `TrainConfig` defaults updated; README "Training Defaults" table updated to match.

### Schedule-knob sensitivities (durable)

1. **`lr` dominates.** Sweet spot 2e-4 to 1e-3. `lr<1e-4` trains too slowly; `lr>=2e-3` overshoots and degrades val_acc sharply.
2. **`lr_center_epoch` must align with the peak.** At `lr=2e-4` the model peaks at ep ~11. `center=10` aligns. `center=5` decays too fast (underfit). `center=12` leaves LR too high through the peak (no gain). Default `center=20` from a 500-epoch schedule leaves LR at full strength too long → post-peak collapse.
3. **`lr_sharpness` controls post-peak stability.** `sharp=4` → noisy plateau (~0.6766). `sharp=8` → tighter plateau (~0.6758). Peak value unchanged.
4. **`lr_tail_strength=0.5`** is the sweet spot. `0.3` gives a noisier plateau (~0.6763 mean vs r2's ~0.6757). `1.0` (untested in iter; tried in r0) decays too fast.
5. **`warmup_steps`**: at default `lr=5e-5` has no effect. At high LR, `warmup_500` hurts by ~4e-4. Keep at default 125 unless investigating.
6. **`weight_decay`, `lr_eta_min_ratio`, `lr_sharpness>=2`**: all sub-noise. Don't sweep these further unless lr/schedule are locked.

### Methodology note

The first verdict from this sweep was "no signal" — based on comparing iter's 3-seed mean to a single-seed baseline. Re-running baseline at the same seeds inverted the conclusion: paired Δ is consistent in sign and magnitude across all seeds. **Always multi-seed both arms before declaring a tie**; sub-noise gains in the mean can still be real if they appear paired.
<!-- lr-schedule-20260518:end -->

## Model Structure Notes

Pick model size by held-out behavior, not by parameter count alone. If train and validation improve together, add capacity. If train improves while validation loss/Brier/ECE stall or worsen, reduce capacity or add regularization. If validation keeps improving and `gen_*` gaps stay small, the model can probably use more capacity.

| Decision | Effect |
| --- | --- |
| `d_model` | Widens every token representation and drives most parameter/memory growth. Use when features look under-expressed, but expect slower training and easier overfit. |
| `n_layers` | Adds repeated token mixing. More layers can model higher-order draft interactions, but redundant layers show up as weak validation gains and similar attention heads. |
| `dim_feedforward` | Adds per-token nonlinear capacity. For this tabular/token setup, widen FFN before adding much depth. |
| `n_heads` | Changes how attention is partitioned, with little parameter change at fixed `d_model`. Too many heads can become redundant; too few can bottleneck distinct role/team patterns. |
| `dropout` / `attention_dropout` | Regularizes memorization. Raise when train metrics pull ahead; lower if both train and validation underfit. |
| `pooling` | Controls the per-team summary sent to the head: `team_mean` (unweighted) or `team_attention` (softmax-weighted). The head input is a 6-way concat `(b, r, b-r, abs(b-r), b*r, lane)` — see [MODEL.md](MODEL.md). |
| `head_hidden` | Capacity after pooling. Usually a small lever compared with encoder width/depth. |
| `profile_confidence_prior_count` | Pseudo-count for profile reliability, `confidence = n / (n + k)`, where `n = expm1(log_matchups)`. Larger values shrink sparse profile rows harder. |

### Profile Confidence Follow-Up, 2026-05-20

Current implementation: `log_matchups` is excluded from the profile content MLP and used strictly as reliability. The temporal encoder recovers `n = expm1(log_matchups)`, computes `confidence = n / (n + profile_confidence_prior_count)`, adds `log(confidence)` to bin attention scores, scales pooled bin values by confidence, and scales the early/late delta by the weaker half's confidence.

Future options to evaluate once this baseline has support-sliced metrics:

- Sweep `profile_confidence_prior_count` over plausible pseudo-counts such as `16, 32, 64, 128, 256`; select by validation loss plus rare-profile Brier/ECE, not headline AUC alone.
- Replace shrink-to-zero with explicit empirical-Bayes profile shrinkage at build time: blend each `(champion, role, build, bin)` metric toward `(champion, role, bin)`, then `(role, bin)`, then `(global, bin)` priors.
- Use per-feature shrinkage strengths. Win rate, damage shares, ratios, and high-variance per-minute rates likely need different pseudo-counts.
- Store raw `matchups` beside `log_matchups` in a future cache format to avoid recovering support through `expm1` on float16 cache values.
- Add support-sliced diagnostics: tokens with all bins below 16/32/64 games, rare champion/build rows, and rows with one low-confidence half in the early/late delta.
- Test explicit prior embeddings or prior profile rows instead of zero-profile shrinkage, so low-support rows fall back to role/champion priors inside the model rather than only at cache-build time.

### CUDA Graph Follow-Up, 2026-05-14

Manual CUDA graph probes used `batch_size=10240`, current model shape, static copied batches, BF16 autocast, and fused AdamW.

| path | result | median samples/s | note |
| --- | --- | ---: | --- |
| eager + AdamW `capturable=False` | failed | n/a | AdamW refuses graph capture unless the param group is capturable. |
| eager + AdamW `capturable=True` | captured | 44,866 | Works, but is slower than compiled training. |
| compiled + manual CUDA graph | failed | n/a | Fails with Inductor CUDA graph replay inside an active capture. |

Conclusion: do not add manual CUDA graphs to the training loop. `torch.compile(mode="reduce-overhead")` remains the graph/launch-overhead lever.
