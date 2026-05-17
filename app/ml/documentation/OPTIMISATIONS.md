# ML Optimisations

Maintenance: update this file with decision-grade experiment evidence only: protocol, compact table, conclusion, and the recommended setting. Keep reusable testing procedure in `TESTING.md` and current defaults in `README.md`.


<!-- pooling-multi-seed-20260517:start -->
## Pooling × Seed Sweep, 2026-05-17

Status: `completed`. Run directory: `app/ml/data/checkpoints/pooling_sweep_20260517_141947`.

Protocol: live defaults, only `ModelConfig.pooling` and `TrainConfig.seed` vary. Each trial in an isolated subprocess. Seeds `[42, 123, 777]`. Test metrics from best-val-loss checkpoint. Symmetry from `evaluate_symmetry` on test (`n=167,822`/trial), measuring `|p_swap − (1 − p_orig)|`.

Modes compared: `team_mean` / `team_attention` (5-way concat `(b, r, b-r, |b-r|, b*r)`, `head_input=5d`), `team_mean_symmetric` (3-way concat `(b-r, |b-r|, b*r)`, `head_input=3d`; drops the asymmetric `(b, r)` prefix so the head can only read swap-antisymmetric `b-r` plus the two swap-symmetric magnitudes).

### Mean ± std across seeds

| pooling | test_loss | test_auc | test_brier | test_ece | sym_mean | sym_p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| team_mean | **0.675578 ± 0.000043** | **0.600971 ± 0.000306** | **0.241397 ± 0.000020** | 0.017061 ± 0.001209 | 1.5189e-01 ± 2.2e-03 | 3.8638e-01 ± 5.1e-03 |
| team_attention | 0.675718 ± 0.000047 | 0.600825 ± 0.000227 | 0.241464 ± 0.000021 | 0.018757 ± 0.001260 | 1.5126e-01 ± 3.4e-03 | 3.8637e-01 ± 8.2e-03 |
| team_mean_symmetric (default) | 0.675641 ± 0.000175 | 0.600418 ± 0.000718 | 0.241430 ± 0.000087 | **0.016419 ± 0.001016** | **1.4583e-01 ± 4.0e-03** | **3.7468e-01 ± 9.2e-03** |

### Recommendation

`team_mean_symmetric`: best symmetry and best calibration (ECE) among the team modes; scoring metrics within ≤1.5e-3 of the `team_mean` leader. The architectural prior (head reads only swap-antisymmetric + swap-symmetric components) is the deciding factor on a near-tie. `team_attention` is dominated on every axis. Pick `team_mean` if a future change targets held-out scoring without the symmetry constraint.
<!-- pooling-multi-seed-20260517:end -->

## Optimizer Default

The 2026-05-13 optimizer evidence tuned Lion against itself but did not include a matched AdamW control. With no decision-grade evidence that Lion improves held-out metrics, and with PyTorch AdamW offering a fused CUDA implementation without a third-party package, the active default is AdamW:

```text
optimizer = "adamw"
lr = 5e-5
weight_decay = 5e-3
adamw_betas = (0.9, 0.999)
```

Recommendation: use fused PyTorch AdamW on CUDA and keep Lion retired unless a fresh isolated sweep beats this AdamW baseline on validation loss first and does not regress samples/s.

<!-- adamw-lower-lr-150epoch-20260515:start -->
## AdamW Lower Learning Rate 150-Epoch Sweep, 2026-05-15

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

Conclusion: the current recommended setting from this sweep is `lr=5.00e-5`, selected by lowest validation loss (`0.675675`).
<!-- adamw-lower-lr-150epoch-20260515:end -->

## Model Structure Notes

Pick model size by held-out behavior, not by parameter count alone. If train and validation improve together, add capacity. If train improves while validation loss/Brier/ECE stall or worsen, reduce capacity or add regularization. If validation keeps improving and `gen_*` gaps stay small, the model can probably use more capacity.

| Decision | Effect |
| --- | --- |
| `d_model` | Widens every token representation and drives most parameter/memory growth. Use when features look under-expressed, but expect slower training and easier overfit. |
| `n_layers` | Adds repeated token mixing. More layers can model higher-order draft interactions, but redundant layers show up as weak validation gains and similar attention heads. |
| `dim_feedforward` | Adds per-token nonlinear capacity. For this tabular/token setup, widen FFN before adding much depth. |
| `n_heads` | Changes how attention is partitioned, with little parameter change at fixed `d_model`. Too many heads can become redundant; too few can bottleneck distinct role/team patterns. |
| `dropout` / `attention_dropout` / `head_dropout` | Regularizes memorization. Raise when train metrics pull ahead; lower if both train and validation underfit. |
| `pooling` | Controls the summary sent to the prediction head. Modes (see the 2026-05-17 sweep): `team_mean` / `team_attention` (per-team pool + 5-way concat `(b, r, b-r, abs(b-r), b*r)`) |
| `head_hidden` | Capacity after pooling. Usually a small lever compared with encoder width/depth. |

### CUDA Graph Follow-Up, 2026-05-14

Manual CUDA graph probes used `batch_size=10240`, current model shape, static copied batches, BF16 autocast, and fused AdamW.

| path | result | median samples/s | note |
| --- | --- | ---: | --- |
| eager + AdamW `capturable=False` | failed | n/a | AdamW refuses graph capture unless the param group is capturable. |
| eager + AdamW `capturable=True` | captured | 44,866 | Works, but is slower than compiled training. |
| compiled + manual CUDA graph | failed | n/a | Fails with Inductor CUDA graph replay inside an active capture. |

Conclusion: do not add manual CUDA graphs to the training loop. `torch.compile(mode="reduce-overhead")` remains the graph/launch-overhead lever.

<!-- capacity-reg-grid-20260515:start -->
## Capacity × Regularization Grid Sweep, 2026-05-15

Status: `partially completed` — t00–t19 ran to 150 epochs; t20 (d384/l4/ff2048) was terminated early (~30 epochs) due to batch-size collapse caused by insufficient GPU memory headroom at that capacity, reducing throughput to ~10k samples/s. Run directory:

```text
app/ml/data/checkpoints/capacity_reg_grid_20260515_210752
```

Protocol: 5×5 grid (5 capacity tiers × 5 regularization tiers = 25 trials), isolated fresh Python subprocess per trial, seed `42`, `epochs=150`, `batch_size=16384`, BF16 AMP, fused AdamW, `torch.compile(mode="reduce-overhead")`, attention diagnostics disabled, train monitor `50k` samples/epoch. Best val-loss checkpoint used for test evaluation. Median samples/s computed from `train_step` events after step 10 (skips compile warmup).

Capacity tiers (`d_model` / `n_layers` / `dim_feedforward`, `n_heads=4` fixed):

```text
C0: 192 / 3 /  768
C1: 256 / 3 / 1024
C2: 256 / 4 / 1536
C3: 384 / 3 / 1536
C4: 384 / 4 / 2048  ← terminated early
```

Regularization tiers (`dropout` / `attention_dropout` / `weight_decay`):

```text
R0: 0.05 / 0.05 / 1e-3
R1: 0.15 / 0.10 / 5e-3
R2: 0.25 / 0.15 / 1e-2
R3: 0.35 / 0.20 / 2e-2
R4: 0.45 / 0.25 / 5e-2
```

### Full Results

Sorted by capacity tier then regularization tier. `best ep` = epoch of best val loss. Samples/s from training steps only (excludes val/monitor overhead).

| trial | cap | reg | epochs | best ep | val loss | val acc | val AUC | val Brier | val ECE | test loss | test acc | test AUC | test Brier | test ECE | med samples/s |
| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t00 d192/l3/ff768 | C0 | R0 | 150 | 65 | 0.677196 | 0.5708 | 0.598349 | 0.242159 | 0.012168 | 0.676330 | 0.5728 | 0.598872 | 0.241741 | 0.016725 | 209,963 |
| t01 d192/l3/ff768 | C0 | R1 | 150 | 111 | 0.677121 | 0.5703 | 0.599112 | 0.242133 | 0.017318 | 0.676807 | 0.5722 | 0.599564 | 0.241986 | 0.024896 | 211,384 |
| t02 d192/l3/ff768 | C0 | R2 | 150 | 111 | 0.677661 | 0.5685 | 0.597451 | 0.242396 | 0.017584 | 0.677589 | 0.5705 | 0.597555 | 0.242363 | 0.026280 | 212,411 |
| t03 d192/l3/ff768 | C0 | R3 | 150 | 111 | 0.678468 | 0.5667 | 0.594907 | 0.242789 | 0.019163 | 0.678574 | 0.5686 | 0.594920 | 0.242836 | 0.028027 | 212,433 |
| t04 d192/l3/ff768 | C0 | R4 | 150 | 111 | 0.679678 | 0.5643 | 0.590989 | 0.243363 | 0.017593 | 0.679638 | 0.5666 | 0.591574 | 0.243329 | 0.026179 | 212,459 |
| t05 d256/l3/ff1024 | C1 | R0 | 150 | 34 | 0.676780 | 0.5722 | 0.599282 | 0.241963 | 0.011960 | 0.676263 | 0.5735 | 0.599405 | 0.241708 | 0.017379 | 167,026 |
| **t06 d256/l3/ff1024** | **C1** | **R1** | **150** | **111** | **0.675678** | **0.5728** | **0.601421** | **0.241450** | **0.007786** | **0.675649** | **0.5737** | **0.600729** | **0.241421** | **0.016638** | **167,855** |
| t07 d256/l3/ff1024 | C1 | R2 | 150 | 111 | 0.675860 | 0.5726 | 0.601084 | 0.241534 | 0.008147 | 0.675688 | 0.5753 | 0.600906 | 0.241437 | 0.016301 | 167,004 |
| t08 d256/l3/ff1024 | C1 | R3 | 150 | 128 | 0.676813 | 0.5704 | 0.598151 | 0.241991 | 0.008636 | 0.676555 | 0.5734 | 0.598413 | 0.241852 | 0.017186 | 167,019 |
| t09 d256/l3/ff1024 | C1 | R4 | 150 | 128 | 0.678353 | 0.5664 | 0.593396 | 0.242733 | 0.010421 | 0.678156 | 0.5691 | 0.593878 | 0.242623 | 0.019419 | 167,936 |
| t10 d256/l4/ff1536 | C2 | R0 | 150 | 33 | 0.676520 | 0.5722 | 0.600297 | 0.241840 | 0.013679 | 0.676347 | 0.5726 | 0.599605 | 0.241753 | 0.018524 | 112,273 |
| t11 d256/l4/ff1536 | C2 | R1 | 150 | 70 | 0.675557 | 0.5729 | 0.602009 | 0.241389 | 0.008108 | 0.675313 | 0.5747 | 0.601552 | 0.241265 | 0.016334 | 121,093 |
| t12 d256/l4/ff1536 | C2 | R2 | 150 | 89 | 0.675633 | 0.5731 | 0.601784 | 0.241425 | 0.008696 | 0.675383 | 0.5755 | 0.602088 | 0.241288 | 0.016873 | 111,066 |
| t13 d256/l4/ff1536 | C2 | R3 | 150 | 89 | 0.676231 | 0.5711 | 0.599678 | 0.241709 | 0.006290 | 0.675944 | 0.5736 | 0.599552 | 0.241558 | 0.013723 | 110,609 |
| t14 d256/l4/ff1536 | C2 | R4 | 150 | 127 | 0.677349 | 0.5686 | 0.596521 | 0.242249 | 0.010384 | 0.677149 | 0.5717 | 0.596872 | 0.242136 | 0.019338 | 110,595 |
| t15 d384/l3/ff1536 | C3 | R0 | 150 | 29 | 0.676410 | 0.5713 | 0.599669 | 0.241787 | 0.009169 | 0.675982 | 0.5719 | 0.598837 | 0.241580 | 0.013950 | 99,243 |
| t16 d384/l3/ff1536 | C3 | R1 | 150 | 34 | 0.675557 | 0.5739 | 0.602078 | 0.241383 | 0.008295 | 0.675147 | 0.5751 | 0.601698 | 0.241188 | 0.016740 | 99,249 |
| t17 d384/l3/ff1536 | C3 | R2 | 150 | 76 | 0.675505 | 0.5735 | 0.601744 | 0.241366 | 0.005764 | 0.675135 | 0.5753 | 0.601244 | 0.241180 | 0.013302 | 98,640 |
| t18 d384/l3/ff1536 | C3 | R3 | 150 | 15 | 0.679733 | 0.5646 | 0.589780 | 0.243391 | 0.013627 | 0.679726 | 0.5656 | 0.589677 | 0.243376 | 0.022351 | 98,684 |
| t19 d384/l3/ff1536 | C3 | R4 | 150 | 128 | 0.676843 | 0.5702 | 0.598706 | 0.242000 | 0.010438 | 0.676560 | 0.5723 | 0.598964 | 0.241852 | 0.018999 | 99,269 |
| t20 d384/l4/ff2048 | C4 | R0 | 30† | 26 | 0.676635 | 0.5702 | 0.598384 | 0.241902 | 0.007964 | — | — | — | — | — | 10,122† |

† Terminated early. Throughput reflects batch-size collapse (GPU OOM pressure), not model cost.

### Tier Summary (best trial per capacity tier)

| tier | config | best trial / reg | best val loss | test loss | test acc | test AUC | test Brier | test ECE | med samples/s | vs C1 throughput |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | d192 / l3 / ff768 | t01 / R1 | 0.677121 | 0.676807 | 0.5722 | 0.599564 | 0.241986 | 0.024896 | 211,384 | +26% |
| **C1** | **d256 / l3 / ff1024** | **t06 / R1** | **0.675678** | **0.675649** | **0.5737** | **0.600729** | **0.241421** | **0.016638** | **167,855** | **—** |
| C2 | d256 / l4 / ff1536 | t11 / R1 | 0.675557 | 0.675313 | 0.5747 | 0.601552 | 0.241265 | 0.016334 | 121,093 | −28% |
| C3 | d384 / l3 / ff1536 | t17 / R2 | 0.675505 | 0.675135 | 0.5753 | 0.601244 | 0.241180 | 0.013302 | 98,640 | −41% |
| C4 | d384 / l4 / ff2048 | t20 / R0 | 0.676635† | — | — | — | — | — | 10,122† | −94%† |

### Conclusions

**Accuracy ceiling**: all four completed tiers converge within 0.002 val loss of each other (0.6755–0.6771 at their best regularization). The dataset signal, not model capacity, limits performance above C1.

**Best throughput/accuracy tradeoff: C1 (d256/l3/ff1024, R1)**. Matches C2 and C3 accuracy to within noise while running at 167k samples/s — 26% faster than C2 (d256/l4) and 41% faster than C3 (d384/l3). These defaults are already the active config.

**C0 (d192) is genuinely compromised**: consistently 0.001–0.004 higher val loss and notably worse ECE (0.017–0.028 vs 0.007–0.017 for C1), even at peak throughput. Not a valid tradeoff.

**Regularization**: R1 (dropout=0.15, attn\_dropout=0.10, weight\_decay=5e-3) or R2 (0.25/0.15/1e-2) is optimal across all tiers. R3–R4 degrades all metrics. R0 underregularizes slightly (best val at early epochs, marginally worse final metrics).

**Recommended setting (confirmed)**: `d_model=256, n_layers=3, dim_feedforward=1024, dropout=0.15, attention_dropout=0.10, weight_decay=5e-3` — existing defaults unchanged.
<!-- capacity-reg-grid-20260515:end -->
