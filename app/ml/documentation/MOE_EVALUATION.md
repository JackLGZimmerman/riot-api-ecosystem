# Match-Level MoE Evaluation

Maintenance: keep first-pass and MoE-specific evidence here. Promote only decision-grade, paired or multi-seed conclusions into [OPTIMISATIONS.md](OPTIMISATIONS.md).

## Current Architecture

Status: `dense head promoted as default after one-by-one optimization`, 2026-05-19.

The model still contains the full match-level MoE head over the shared trunk, but the active default is now the dense antisymmetric head (`use_moe=false`). The one-by-one optimization below found that the MoE routing diagnostics were useful, but the enabled MoE head did not beat the dense head on validation loss under hyperparameter-only changes.

- shared transformer trunk and team pooling stay unchanged
- the dense antisymmetric head still emits `baseline_logit = score(b,r) - score(r,b)` for diagnostics
- setting `use_moe=true` activates `_MatchLevelMoEHead`, which routes from `match_features` plus three antisymmetric scalars derived from `baseline_logit`: `baseline_logit` for the `m(b, r)` call, `-baseline_logit` for `m(r, b)`, plus the implied `prob` and `|prob - 0.5|`
- in MoE mode, only selected expert/sample pairs are evaluated after routing; top-k router weights are applied, and the final logit is `score(b, r) - score(r, b)` after adding the orientation-local MoE correction
- MoE training-stability bundle: router final linear is zero-init (p0), Switch-style load-balance aux loss summed across both orientations (p1), and the first `moe_warmup_steps` use dense routing (k=`n_experts`) before switching to top-k (p2)
- MoE routing noise (p4): Shazeer-style Gaussian noise on routing logits in train mode only, disabled during the dense-routing warmup
- MoE route diagnostics (p5-p7): `matched_diagnostic_tensors` returns selected weights, full router softmax, and weighted expert corrections for both `m(b, r)` and `m(r, b)`; `matched_moe_diagnostics` reports route entropy, top-k margin, per-expert selected share/weight/correction/outcome slices, combined orientation stats, and matched baseline-vs-final metrics across central and tail confidence bands
- default config: `use_moe=false`; retained MoE defaults are `n_experts=8`, `moe_top_k=2`, `expert_hidden=128`, `router_hidden=64`, `router_temperature=1.0`, `moe_aux_loss_coef=0.01`, `moe_warmup_steps=81`, `moe_router_noise=0.354`

## Critical Issue and Proposed Solutions

The major issue is now explicit: **the router is not learning decisive specialization, and the MoE behaves like a tiny confidence-shaping layer rather than a different decision surface**. In the p5-p8 run, the full-router entropy is `2.0794`, effectively `ln(8)`, and the top-k margin is only `0.0005`; selected top-k weights are almost exactly `0.5/0.5`. The selected experts differ by tiny logit noise, not by confident routing.

The matched central 40-60% slice confirms that this does not buy scalar quality: BCE/Brier/AUC are unchanged at displayed precision, ECE worsens by `+0.0003`, and hard accuracy moves `-0.0002`. Tail coverage shows the correction is mostly monotonic with baseline confidence: lower-probability bands get pushed down and higher-probability bands get pushed up. That improves ECE below `45%` baseline probability, but worsens ECE from `47.5%` upward and especially in the `60-100%` tail.

Likely causes and proposed solutions, ordered by expected leverage:

1. **Keep the dense-vs-MoE gate as the promotion check.** The 2026-05-19 pass found `use_moe=false` ahead on validation loss. Future MoE work should clear that same paired same-seed/split gate before becoming the default again.
2. **Make routing non-degenerate.** Try lowering/annealing `moe_aux_loss_coef`, lowering/annealing `moe_router_noise`, lowering `router_temperature`, or adding a small negative-entropy pressure after warmup. The gate should be route entropy and top-k margin, not just selected-count balance.
3. **Give the router and experts richer specialization inputs.** Current experts all see the same pooled `match_features`. Add lane matchup summaries, player-token attention summaries, patch id, player mastery/recent form, and queue-relative skill bracket so the router has real context on which to specialize.
4. **Differentiate experts deliberately.** Try per-expert dropout schedules, an expert-orthogonality penalty on hidden activations, or upcycling experts from a trained dense head with small per-expert perturbations.
5. **Put gradient on the uncertain cases.** BCE spreads signal across all confidence bands; a small focal or hinge term on baseline `0.45-0.55` examples may help the head learn separation instead of a global confidence stretch.
6. **Guard tail calibration.** The new band table shows high-side tail ECE regressions. Any future objective change should track central and tail ECE/Brier separately, not only headline loss.

## One-by-One Hyperparameter Optimization, 2026-05-19

Status: `complete for this pass`. The dense head is the validation-loss winner and has been promoted to the default model config. The run did not hit the 15-experiment fail-safe; it stopped because the first dense gate improved loss and each follow-up hyperparameter probe either regressed or only produced a tiny validation-accuracy movement without a loss win.

Protocol:

- Same seed/split/training schedule for every run: seed `42`, CUDA bfloat16 AMP, `lr=0.0002`, `weight_decay=0.005`, `warmup_steps=125`, early-stop patience `10`.
- Iteration metric: best validation loss from the early-stopped checkpoint; validation accuracy is tracked as a secondary tie-breaker.
- Final confirmation: reran the winning dense config with final test enabled as `e07_dense_final`.
- TensorBoard: all runs are under `app/ml/data/tensorboard/moe_hparam_20260519/<run>/`.
- Concise summary utility: `uv run python -m app.ml.utils.diagnostics_summary ... --sort best_val_loss --details`.

TensorBoard review command:

```bash
uv run tensorboard --logdir app/ml/data/tensorboard
```

Experiment table:

| run | one-by-one change | best val_loss | best val_acc | max val_acc | test_loss | test_acc | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `e01_dense_no_moe` | `use_moe=false` | 0.67500 | 57.370% | 57.370% | - | - | first validation-loss win |
| `e07_dense_final` | final-test rerun of `use_moe=false` | 0.67500 | 57.380% | 57.380% | 0.67490 | 57.510% | promoted default |
| `e03_router_noise_0` | MoE, `moe_router_noise=0.0` | 0.67530 | 57.390% | 57.390% | - | - | best MoE loss probe, still behind dense |
| `e04_noise0_aux_0p001` | MoE, noise off + `moe_aux_loss_coef=0.001` | 0.67540 | 57.410% | 57.420% | - | - | tiny validation-accuracy edge, worse loss |
| `e05_noise0_topk1` | MoE, noise off + `moe_top_k=1` | 0.67540 | 57.370% | 57.370% | - | - | no gain |
| `e06_dense_dropout_0p10` | dense, `dropout=0.10` | 0.67550 | 57.310% | 57.310% | - | - | more overfit than default dropout |
| current MoE baseline | `use_moe=true`, prior default | 0.67560 | 57.370% | 57.370% | 0.67530 | 57.540% | router still near-uniform |
| `e02_aux_0p001` | MoE, `moe_aux_loss_coef=0.001` | 0.67560 | 57.370% | 57.370% | - | - | tied prior MoE loss |

Final promoted dense test metrics from `e07_dense_final`: `test_loss=0.67490`, `test_accuracy=57.510%`, `test_auc=0.60288`, `test_brier=0.24107`, `test_ece=0.01695`, `n=167822`.

Decision:

- Promote `ModelConfig(use_moe=false)` as the current default.
- Keep the MoE implementation and diagnostics available for targeted ablations, but do not enable it by default until routing specialization improves.
- Treat the `e04` validation-accuracy edge as non-decision-grade: it is `+0.04 pp` over dense and comes with `+0.00040` worse validation loss. The current MoE baseline also has `+0.03 pp` higher test accuracy than dense, but worse validation loss, test loss, AUC, Brier, and ECE.
- The most useful MoE diagnostic remains router degeneracy: the current MoE baseline has entropy `2.0794` and top-k margin `0.0005`, so the router is balanced but not decisive.

## P1 + p3-p8 Run, 2026-05-19

Status: `first-pass single-seed with full p5-p8 diagnostics`. Useful architecture evidence, not promotion evidence.

Protocol:

- Command: `CLICKHOUSE_HOST=localhost .venv/bin/python -m app.ml.train`
- Device: CUDA, RTX 5070 Ti, bfloat16 AMP, `compile_mode=reduce-overhead`
- Data: train `1,342,579`, validation `167,822`, test `167,822`
- Best checkpoint: epoch `13`, step `1053`, validation loss `0.67562`
- Early stop: epoch `23`, step `1863`, patience `10`
- Final test: best validation-loss checkpoint reloaded

### Scalar Result

Rows are not paired current-code comparisons. The dense baseline, old-MoE, and p3/p4 rows are historical anchors; the p5-p8 row is the current full-head run.

| metric | dense baseline anchor | old residual MoE | P1 full MoE head | P1 + p3/p4 | P1 + p3-p8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| test_loss | 0.67504 | 0.67490 | 0.67495 | 0.67488 | 0.67527 |
| test_accuracy | 0.57600 | 0.57531 | 0.57520 | 0.57555 | 0.57540 |
| test_auc | 0.60273 | 0.60287 | 0.60282 | 0.60282 | 0.60189 |
| test_brier | 0.24114 | 0.24108 | 0.24110 | 0.24107 | 0.24125 |
| test_ece | 0.01760 | 0.01696 | 0.01739 | 0.01641 | 0.01733 |

Read: the p5-p8 code adds visibility and sparse dispatch, not a scalar win. This run is worse than p3/p4 on loss, AUC, Brier, and ECE, and still below the dense historical accuracy anchor. Treat it as diagnostic evidence only.

### Prediction Density

Final prediction density from the test run:

| region | count | share | accuracy |
| --- | ---: | ---: | ---: |
| all test examples | 167822 | 100.0000% | 57.5400% |
| central 40-60% | 122292 | 72.8701% | 54.8319% |
| central 45-55% | 71229 | 42.4432% | 52.8282% |
| tails <=40% or >=60% | 45529 | 27.1293% | 64.8115% |
| confident <=30% or >=70% | 6310 | 3.7599% | 73.3598% |

Folded confidence view:

| folded confidence band | count | share | accuracy |
| --- | ---: | ---: | ---: |
| 50-55% | 71229 | 42.4432% | 52.8282% |
| 55-60% | 51063 | 30.4269% | 57.6268% |
| 60-65% | 27382 | 16.3161% | 62.2088% |
| 65-70% | 11837 | 7.0533% | 66.2753% |
| 70-75% | 4269 | 2.5438% | 70.5318% |
| 75-80% | 1403 | 0.8360% | 76.7641% |
| 80-85% | 441 | 0.2628% | 80.9524% |
| 85-90% | 148 | 0.0882% | 93.9189% |
| 90-100% | 49 | 0.0292% | 91.8367% |

The model is slightly less central than p3/p4: central 40-60% density drops from `74.34%` to `72.87%`, while tails rise from `25.66%` to `27.13%`. The added confidence does not translate into better headline accuracy.

### Matched Diagnostics

Matched diagnostics compare the dense baseline head and final MoE logit on the same test examples. Numbers below are from the p5-p8 run (zero-init router, Switch-style aux loss `0.01`, dense-routing warmup `81` steps, antisymmetric route input via `baseline_logit`, Gaussian routing noise `0.354` in train mode, both-orientation route telemetry, sparse expert dispatch).

| central 40-60% metric | dense baseline | post-p5-p8 final | delta |
| --- | ---: | ---: | ---: |
| count | 123269 | 123269 | 0 |
| BCE | 0.6869 | 0.6869 | +0.0000 |
| Brier | 0.2469 | 0.2469 | +0.0000 |
| ECE | 0.0173 | 0.0177 | +0.0003 |
| AUC | 0.5626 | 0.5626 | +0.0000 |
| hard accuracy | 0.5488 | 0.5486 | -0.0002 |

Folded confidence transition counts:

| baseline -> final confidence | 50.0-52.5% | 52.5-55.0% | 55.0-57.5% | 57.5-60.0% | 60.0-100.0% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50.0-52.5% | 36472 | 1189 | 0 | 0 | 0 |
| 52.5-55.0% | 720 | 32383 | 1379 | 0 | 0 |
| 55.0-57.5% | 0 | 472 | 26767 | 1365 | 0 |
| 57.5-60.0% | 0 | 0 | 286 | 21085 | 1151 |
| 60.0-100.0% | 0 | 0 | 0 | 166 | 44387 |

Matched metric deltas by dense-baseline band:

| baseline band | count | BCE delta | Brier delta | ECE delta | AUC delta | acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0%-30.0% | 3636 | +0.0000 | +0.0000 | -0.0033 | +0.0000 | +0.0000 |
| 30.0%-40.0% | 23644 | -0.0001 | -0.0000 | -0.0015 | +0.0002 | +0.0000 |
| 40.0%-42.5% | 13141 | -0.0001 | -0.0000 | -0.0009 | +0.0007 | +0.0000 |
| 42.5%-45.0% | 15890 | -0.0001 | -0.0000 | -0.0006 | +0.0007 | +0.0000 |
| 45.0%-47.5% | 18443 | +0.0000 | +0.0000 | -0.0002 | -0.0009 | +0.0000 |
| 47.5%-50.0% | 19168 | -0.0000 | -0.0000 | +0.0002 | +0.0008 | +0.0005 |
| 50.0%-52.5% | 18493 | +0.0000 | +0.0000 | +0.0006 | -0.0001 | -0.0018 |
| 52.5%-55.0% | 16039 | +0.0001 | +0.0000 | +0.0010 | +0.0009 | +0.0000 |
| 55.0%-57.5% | 12714 | +0.0001 | +0.0001 | +0.0014 | -0.0008 | +0.0000 |
| 57.5%-60.0% | 9381 | +0.0002 | +0.0001 | +0.0018 | -0.0018 | +0.0000 |
| 60.0%-70.0% | 14884 | +0.0002 | +0.0001 | +0.0023 | -0.0000 | +0.0000 |
| 70.0%-100.0% | 2389 | +0.0004 | +0.0001 | +0.0028 | +0.0008 | +0.0000 |

Correction rows by dense-baseline band:

| baseline band | count | mean logit delta | mean prob delta | abs logit p50 | abs logit p90 | abs logit p99 | sign agreement | combined top-2 mix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0%-30.0% | 3636 | -0.0121 | -0.0023 | 0.0115 | 0.0250 | 0.0366 | 70.21% | E5 0.231 / E6 0.228 |
| 30.0%-40.0% | 23644 | -0.0067 | -0.0015 | 0.0070 | 0.0169 | 0.0263 | 59.80% | E5 0.226 / E6 0.184 |
| 40.0%-42.5% | 13141 | -0.0038 | -0.0009 | 0.0054 | 0.0130 | 0.0203 | 54.92% | E5 0.215 / E6 0.170 |
| 42.5%-45.0% | 15890 | -0.0023 | -0.0006 | 0.0047 | 0.0118 | 0.0193 | 52.54% | E5 0.215 / E6 0.169 |
| 45.0%-47.5% | 18443 | -0.0008 | -0.0002 | 0.0044 | 0.0110 | 0.0179 | 50.30% | E5 0.212 / E6 0.163 |
| 47.5%-50.0% | 19168 | +0.0009 | +0.0002 | 0.0043 | 0.0108 | 0.0189 | 50.30% | E5 0.211 / E6 0.167 |
| 50.0%-52.5% | 18493 | +0.0025 | +0.0006 | 0.0045 | 0.0116 | 0.0195 | 50.28% | E5 0.211 / E6 0.165 |
| 52.5%-55.0% | 16039 | +0.0042 | +0.0010 | 0.0051 | 0.0133 | 0.0221 | 51.24% | E5 0.213 / E6 0.169 |
| 55.0%-57.5% | 12714 | +0.0059 | +0.0014 | 0.0060 | 0.0153 | 0.0241 | 52.86% | E5 0.217 / E6 0.180 |
| 57.5%-60.0% | 9381 | +0.0074 | +0.0018 | 0.0072 | 0.0171 | 0.0266 | 54.27% | E5 0.213 / E6 0.187 |
| 60.0%-70.0% | 14884 | +0.0101 | +0.0023 | 0.0097 | 0.0210 | 0.0307 | 59.16% | E5 0.218 / E6 0.207 |
| 70.0%-100.0% | 2389 | +0.0150 | +0.0027 | 0.0147 | 0.0284 | 0.0398 | 69.61% | E6 0.264 / E5 0.213 |

Route telemetry:

| view | entropy | top-k margin | top selected experts | correction p90 leaders |
| --- | ---: | ---: | --- | --- |
| `m(b,r)` | 2.0794 | 0.0005 | E5 0.210 / E6 0.189 / E1 0.149 | E5 0.0090 / E1 0.0072 / E7 0.0054 |
| `m(r,b)` | 2.0794 | 0.0005 | E5 0.222 / E0 0.175 / E6 0.167 | E5 0.0111 / E4 0.0077 / E7 0.0077 |
| combined | 2.0794 | 0.0005 | E5 0.638 / E6 0.554 / E0 0.491 | E5 0.0125 / E4 0.0071 / E7 0.0064 |

Matched read: the MoE correction now acts mostly as a monotonic confidence stretch. It pushes low baseline probabilities lower and high baseline probabilities higher, with p50 logit deltas from `0.0043` in the center to `0.0147` in the high tail. That helps low-side ECE but hurts high-side ECE, and the central slice does not improve.

Router behaviour: the full softmax is almost perfectly uniform (`entropy=2.0794`, `ln(8)=2.0794`) and the top-k margin is effectively zero (`0.0005`). The selected-count shares are not uniform because minute logit differences still determine top-k membership, but selected weights are nearly `0.5/0.5`. This is balanced routing without meaningful routing confidence.

### Implementation Review Notes

- p5-p7 are implemented in the matched diagnostics payload and doc tables: both orientations, route telemetry, combined view, and full baseline confidence band coverage are now available.
- p8 is implemented as sparse selected expert/sample dispatch. At `n_experts=8`, this run took `295.9s` to early stop versus `278.1s` for the prior all-expert p3/p4 run, so sparse dispatch is not automatically faster at this small expert count. It is still the right scaling direction if expert count grows.
- The next model decision should not be another blind MoE capacity increase. The route entropy and margin telemetry should be promotion gates for any routing change.

## Historical Residual Takeaway

The old prediction-band residual was a safe first pass: it slightly improved loss, AUC, Brier, and ECE versus the historical dense anchor, slightly hurt hard accuracy, and mostly moved mass from 50-55% into 55-65% confidence. Its limits are now addressed architecturally: routing is learned from match features, the residual cap is removed, and the head scores both orientations through the same MoE module.

## Research Anchors

- [Shazeer et al. 2017](https://arxiv.org/abs/1701.06538): sparse MoE relies on learned routing and conditional capacity.
- [GShard](https://arxiv.org/abs/2006.16668), [Switch Transformer](https://arxiv.org/abs/2101.03961), [V-MoE](https://arxiv.org/abs/2106.05974), and [ST-MoE](https://arxiv.org/abs/2202.08906): routing stability and load management are core design concerns.
- [BASE Layers](https://proceedings.mlr.press/v139/lewis21a.html) and [Expert Choice Routing](https://arxiv.org/abs/2202.09368): route balance affects training quality, not just efficiency.
- [Soft Merging of Experts with Adaptive Routing](https://arxiv.org/abs/2306.03745): soft or semi-sparse routing can be a useful bridge before hard top-k routing.
- [Sparse Upcycling](https://arxiv.org/abs/2212.05055) and [Upcycling Large Language Models into Mixture of Experts](https://arxiv.org/abs/2410.07524): initializing experts from a dense checkpoint can reduce sparse-MoE instability.
- [Muller et al. 2019](https://arxiv.org/abs/1906.02629) and [Guo et al. 2017](https://arxiv.org/abs/1706.04599): density movement must be separated from accuracy and calibration.

## Next Gates

1. Keep `use_moe=false` as the live default unless a future MoE run beats dense on validation loss and does not regress Brier/ECE.
2. If revisiting MoE, prioritize routing degeneracy directly: lower/anneal `router_temperature`, add a small post-warmup entropy-sharpening pressure, or add richer route inputs. Gate on route entropy, top-k margin, central ECE, and tail ECE.
3. Add new information before adding more expert capacity: patch id, player mastery/recent form, queue-relative skill bracket, and lane matchup summaries.
4. For dense-only improvement, sweep `d_model` / `n_layers` / `dim_feedforward` and training schedule after preserving the `e07_dense_final` TensorBoard run as the comparison anchor.
5. Promotion criterion: require a repeatable validation-loss win plus no calibration regression; use hard accuracy only as a secondary metric unless the gain is at least `+0.10 pp`.
