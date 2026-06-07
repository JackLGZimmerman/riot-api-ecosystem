# HGNN Context Examples Audit

Updated: 2026-06-07.

This audit joins the empirical focus-side context examples to the trained semantic HGNN predictions for the same cached games. Each audit is its own table: one row per threshold bin reporting `n / empirical WR / HGNN WR / gap / accuracy`, with a per-table Gap MSE, accuracy, and the accuracy headroom from perfect calibration (`Calibration lift`) above it. Gap is `HGNN WR - empirical WR`; zero gap is the target.

## Scope And Threshold Definitions

- Context source: `app/ml/data/cache` side-row arrays, `val` split only.
- HGNN model: `app/ml/data/hgnn_production_model.pt`.
- HGNN cache: `app/ml/data/cache`.
- Encoder sidecar artifact: `app/ml/data/experiments/semantic_identity_sidecar_compact.npz`.
- HGNN WR uses focus-slot semantic MoE probabilities when a checkpoint exposes slot deltas; older checkpoints fall back to raw `final_logit` probabilities.
- Semantic group feature schema: v1, 17 compact per-slot features; used only by checkpoints trained with `--use-semantic-group-features`.
- Games audited: 143,131.
- Focus-slot rows audited: 1,431,310.
- Model-alignment rows score each slot with its focus-side win probability; blue-side slots use the blue-team frame and red-side slots use the mirrored red-team frame.
- Continuous thresholds are global side-row team-average percentiles.
- Count thresholds use explicit enemy-team counts.
- WR, effects, and gaps are focus-side win-rate percentage points.
- Accuracy is focus-row classification accuracy at the 0.5 threshold (HGNN focus WR >= 0.5 predicts a focus-side win); the per-table value is bin-n weighted.
- `Acc if calibrated` shifts each bin's predictions so the bin mean equals the empirical WR (perfect calibration) while keeping the model's within-bin ranking, then re-thresholds at 0.5; `Calibration lift` is that minus current accuracy -- the true accuracy impact of closing the gap. It is near zero because accuracy is limited by ranking, not calibration.
- Selected-enchanter probe uses Sona, Karma, Lulu, and Zilean in `UTILITY` with `utility_enchanter` or `utility_protection`.
- Low own-damage probe is anchored once per team side, then compared against the enemy heal/shield context.
- Effect shrinkage is `HGNN effect / empirical effect`; values below 1.0 mean the model under-expresses the observed context effect.

| Axis | Low threshold | High threshold | Notes |
|---|---|---|---|
| Physical share | `<= 0.387` | `>= 0.557` | Team-average identity-context physical share. |
| Magic share | `<= 0.373` | `>= 0.549` | Team-average identity-context magic share. |
| Damage pressure | `<= 0.739` | `>= 0.813` | Team-average champion damage pressure. |
| Damage-taken pressure | `<= 0.639` | `>= 0.721` | Team-average damage-taken pressure. |
| Heal/shield pressure | `<= 0.028` | `>= 0.202` | Team-average ally heal/shield pressure. |
| CC pressure | `<= 0.374` | `>= 0.539` | Team-average crowd-control pressure. |
| Siege pressure | `<= 0.441` | `>= 0.530` | Team-average siege and structure pressure. |
| Scaling pressure | `<= 0.829` | `>= 0.863` | Team-average scaling pressure. |
| Burst-proxy count | `0` | `>= 3` | Enemy slots with slot damage pressure `>= 0.952` and a non-tank build. |
| Hard-CC count | `0` | `>= 3` | Enemy slots with slot CC pressure `>= 0.696`. |
| Tank/frontline count | `0` | `>= 3` | Enemy builds in `ar_tank`, `mr_tank`, `ad_off_tank`, or `ap_off_tank`. |
| Heavy damage-taken count | `0` | `>= 3` | Enemy slots with slot damage-taken pressure `>= 0.822`. |
| High-HP count | `0` | `>= 3` | Enemy champions with static level-18 HP `>= 2478.5`. |
| Focus HP tier | `<= 2309.0` | `>= 2478.5` | Static champion level-18 HP. |
| Ranged count | `<= 1` | `>= 4` | Static `attackRange_flat > 250` as ranged. |
| Same-role range | `<= 250` | `> 250` | Static attack range for the lane opponent. |
| Skirmish-ally count | `0` | `>= 2` | Gwen, Jax, Irelia, Fiora, Udyr, and XinZhao on the focus team. |

## Gap Summary

| Section | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Headline Trajectory Audit Tables | 10 | 47 | 1.70 pp | 5.56 pp | 4.65 pp^2 | 58.03% | 58.06% | +0.03 pp |
| Richer Composition Trajectory Tables | 13 | 52 | 1.84 pp | 10.12 pp | 8.01 pp^2 | 57.38% | 57.49% | +0.11 pp |
| Retained Prior And User-Requested Trajectory Tables | 12 | 53 | 1.88 pp | 6.62 pp | 5.75 pp^2 | 57.74% | 57.86% | +0.12 pp |
| Inspected Lower-Signal Trajectory Tables | 4 | 16 | 0.81 pp | 2.86 pp | 1.38 pp^2 | 57.48% | 57.49% | +0.00 pp |
| Top-20 Matchup And Synergy Audits | 7 | 32 | 1.55 pp | 4.54 pp | 3.63 pp^2 | 57.03% | 57.20% | +0.17 pp |

## Train, Validation, And Test Summary

These rows reuse the same audit specs and prediction cache, but evaluate the cached train, validation, and test ranges separately.

| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,145,051 | 11,450,510 | 46 | 200 | 1.14 pp | 8.23 pp | 2.57 pp^2 | 57.66% | 57.69% | +0.03 pp |
| Validation | 143,131 | 1,431,310 | 46 | 200 | 1.69 pp | 10.12 pp | 5.39 pp^2 | 57.50% | 57.54% | +0.05 pp |
| Test | 143,131 | 1,431,310 | 46 | 200 | 1.66 pp | 8.37 pp | 4.95 pp^2 | 56.98% | 57.05% | +0.07 pp |

## Retired MoE Expert-Grid Note

A seed-4 MoE expert-count / `top_k` ablation was run against this audit and
then removed from the maintained experiment workspace. The production checkpoint
remains `hgnn_production_model.pt`; none of these seed-4-only variants were
promoted.

Primary ranking used the mean of validation/test flagged support-weighted mean
absolute gap over the six required flagged audit examples. The best completed
variant was `convex_encoder_mix` with 128 experts and `top_k=32`: flagged MAE
`1.7027 pp`, flagged MSE `4.8201 pp^2`, validation/test accuracy
`57.8547%` / `57.3433%`, and validation/test NLL `0.6729` / `0.6759`. The
`8x2` in-sweep control reported flagged MAE `1.9989 pp`, flagged MSE
`6.2255 pp^2`, validation/test accuracy `57.8575%` / `57.3489%`, and
validation/test NLL `0.6730` / `0.6760`.

The result suggests larger expert capacity can reduce the flagged context gap,
but it was not monotonic (`64x*` and `128x16` were weaker than the best `32x*`
and `128x32` runs). The current implementation evaluates all experts before
applying the top-k mask, so larger-capacity follow-up should first implement
true sparse dispatch. See `HGNN_CURRENT.md` for the full retired ablation table
and sparse-dispatch plan.

## Scaling Time-Bin Calibration

The persistent scaling-time analytics are built by `database/clickhouse/schema/analytics_builds/8005_scaling_item_time_bins.sql`. They use only `game_data_filtered.participant_stats`: completed items are final `item0` through `item6` slots that exist in the generic item-value map key `(championid = 0, teamposition = '')`, match time is `max(timeplayed) / 60`, eligible matches have 10 participant rows, and the first bin starts at the observed minimum eligible game length.

These bins are empirical analytics aids for reading scaling examples. They do not change the current HGNN scaling-pressure feature unless a later model-input change wires `game_data_filtered.match_scaling_time_bins` into training or inference.

| Bin | Item-count interval | Calibration matches | From min | To min | Centroid min |
|---:|---|---:|---:|---:|---:|
| 1 | `2-3 items` | 106,855 | 16.52 | 20.97 | 18.75 |
| 2 | `3-4 items` | 579,784 | 20.97 | 26.48 | 23.73 |
| 3 | `4-5 items` | 575,504 | 26.48 | 31.88 | 29.18 |
| 4 | `5+ items` | 165,891 | 31.88 | 38.81 | 35.35 |

Hard match assignment uses adjacent centroids and a normal-weighted deterministic hash draw. The materialized assignment table has 1,431,313 rows, 1,431,313 distinct matches, zero duplicate match rows, and maximum probability-sum error `2.98e-8`.

| Assigned bin | Matches | Avg game min | Avg completed items |
|---|---:|---:|---:|
| `2-3 items` | 66,711 | 19.72 | 2.49 |
| `3-4 items` | 336,998 | 24.63 | 3.30 |
| `4-5 items` | 611,641 | 29.29 | 3.98 |
| `5+ items` | 415,963 | 35.83 | 4.85 |

Early-skew examples pool bins 1 and 2 as "early" and compare them with bin 4 as "late"; rows require at least 500 early examples and 200 late examples.

| Champion role build | Early n | Early WR | Late n | Late WR | Early - late |
|---|---:|---:|---:|---:|---:|
| Irelia TOP `attack_damage` | 958 | 73.90% | 3,717 | 51.09% | +22.81 pp |
| DrMundo TOP `ad_off_tank` | 1,236 | 76.21% | 2,094 | 54.20% | +22.01 pp |
| Irelia MIDDLE `attack_damage` | 716 | 69.13% | 1,882 | 48.25% | +20.89 pp |
| Belveth JUNGLE `attack_damage` | 890 | 71.46% | 888 | 52.82% | +18.65 pp |
| Illaoi TOP `attack_damage` | 516 | 61.63% | 767 | 45.37% | +16.26 pp |
| Pyke UTILITY `attack_damage` | 694 | 63.11% | 2,373 | 47.91% | +15.20 pp |
| Yorick TOP `attack_damage` | 1,723 | 58.56% | 2,266 | 43.78% | +14.78 pp |
| Belveth JUNGLE `crit` | 710 | 63.10% | 431 | 48.72% | +14.37 pp |
| Sion TOP `ad_off_tank` | 1,229 | 61.19% | 1,454 | 48.07% | +13.11 pp |
| Bard UTILITY `ap_off_tank` | 904 | 58.96% | 1,616 | 45.92% | +13.04 pp |

Late-skew examples use the same support threshold and show champion/build profiles whose win rate is materially higher in the `5+ items` bin.

| Champion role build | Early n | Early WR | Late n | Late WR | Late - early |
|---|---:|---:|---:|---:|---:|
| Shen TOP `ad_off_tank` | 3,375 | 34.81% | 360 | 51.94% | +17.13 pp |
| Yone TOP `on_hit` | 3,377 | 31.51% | 392 | 45.92% | +14.41 pp |
| Kayle MIDDLE `ability_power` | 3,813 | 48.60% | 3,800 | 62.55% | +13.96 pp |
| Yuumi UTILITY `utility_protection` | 19,521 | 40.72% | 16,695 | 54.21% | +13.49 pp |
| Yone MIDDLE `on_hit` | 3,459 | 30.82% | 669 | 44.10% | +13.28 pp |
| Thresh UTILITY `utility_enchanter` | 3,967 | 37.74% | 456 | 50.88% | +13.14 pp |
| Draven BOTTOM `attack_damage` | 4,082 | 36.65% | 868 | 49.77% | +13.12 pp |
| MissFortune BOTTOM `attack_damage` | 5,796 | 39.80% | 2,539 | 52.86% | +13.05 pp |
| Jhin BOTTOM `lethality` | 4,341 | 31.49% | 542 | 44.46% | +12.97 pp |
| Yasuo MIDDLE `on_hit` | 3,535 | 32.59% | 524 | 45.23% | +12.64 pp |

## Enemy Count Tail Shrinkage

| Audit | Axis | Baseline bin | Tail bin | Empirical tail effect | HGNN tail effect | Shrinkage |
|---|---|---:|---:|---:|---:|---:|
| Sylas MIDDLE `ability_power` vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -7.26 pp | -2.66 pp | 0.37x |
| Nilah BOTTOM any build vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -6.51 pp | -3.64 pp | 0.56x |
| Kaisa BOTTOM any build vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | +0.74 pp | -2.37 pp | -3.21x |
| Kaisa BOTTOM `on_hit` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +6.59 pp | +5.48 pp | 0.83x |
| Ahri MIDDLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +7.64 pp | +4.66 pp | 0.61x |
| Sylas JUNGLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +9.37 pp | +5.08 pp | 0.54x |
| Sylas MIDDLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +11.72 pp | +3.01 pp | 0.26x |
| Karma UTILITY any build vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +1.56 pp | +6.08 pp | 3.90x |
| Vayne BOTTOM `on_hit` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +15.19 pp | +7.42 pp | 0.49x |
| Thresh UTILITY `ar_tank` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.39 pp | -4.15 pp | 0.94x |
| Nautilus UTILITY `mr_tank` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | +3.26 pp | -0.87 pp | -0.27x |
| Zed MIDDLE `lethality` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -17.04 pp | -5.08 pp | 0.30x |
| Nami UTILITY `utility_protection` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -1.85 pp | -2.94 pp | 1.59x |
| Jinx BOTTOM `crit` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.86 pp | -4.08 pp | 0.84x |
| Malphite TOP `ar_tank` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -12.88 pp | -8.26 pp | 0.64x |
| Viego JUNGLE any build vs enemy high-HP count | `enemy_high_hp_count` | `0` | `>= 3` | +1.45 pp | +2.17 pp | 1.50x |
| Darius TOP any build vs enemy range count | `enemy_ranged_count` | `<= 1` | `>= 4` | -4.82 pp | -3.47 pp | 0.72x |
| MasterYi JUNGLE any build vs enemy hard CC | `enemy_hard_cc_count` | `0` | `>= 3` | -0.26 pp | -3.23 pp | 12.45x |
| Focus HP `<= 2309` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.89 pp | -3.87 pp | 0.79x |
| Focus HP `>= 2478` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -3.84 pp | -3.84 pp | 1.00x |
| Ahri MIDDLE `ability_power` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | +1.05 pp | -1.43 pp | -1.36x |
| Kaisa BOTTOM `on_hit` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -3.34 pp | -1.50 pp | 0.45x |
| Ezreal BOTTOM `attack_damage` vs enemy hard CC | `enemy_hard_cc_count` | `0` | `>= 3` | -0.62 pp | -1.54 pp | 2.48x |
| Jayce TOP `attack_damage` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +5.93 pp | +5.41 pp | 0.91x |
| Caitlyn BOTTOM `crit` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -2.29 pp | -2.32 pp | 1.01x |

## Headline Trajectory Audit Tables

### Yasuo TOP `crit` vs enemy siege

Melee crit carry punished by poke and siege.

**Gap MSE** 3.50 pp^2 | **Mean abs gap** 1.58 pp | **Accuracy** 56.56% | **Accuracy if calibrated** 56.62% | **Calibration lift** +0.05 pp | **Empirical effect** -0.46 pp | **HGNN effect** -3.57 pp | **Shrinkage** 7.80x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.441` | 782 | 51.53% | 53.80% | +2.26 pp | 58.31% |
| `0.441-0.471` | 797 | 49.44% | 52.27% | +2.83 pp | 57.34% |
| `0.471-0.499` | 759 | 51.38% | 51.43% | +0.04 pp | 54.02% |
| `0.499-0.530` | 714 | 49.30% | 51.20% | +1.90 pp | 57.28% |
| `>= 0.530` | 650 | 51.08% | 50.23% | -0.85 pp | 55.69% |

### Graves JUNGLE `lethality` vs enemy damage

Burst jungler into high enemy damage.

**Gap MSE** 13.75 pp^2 | **Mean abs gap** 3.33 pp | **Accuracy** 68.57% | **Accuracy if calibrated** 68.35% | **Calibration lift** -0.21 pp | **Empirical effect** -11.09 pp | **HGNN effect** -10.06 pp | **Shrinkage** 0.91x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 483 | 41.20% | 35.64% | -5.56 pp | 63.56% |
| `0.739-0.764` | 408 | 33.09% | 29.63% | -3.46 pp | 67.89% |
| `0.764-0.785` | 340 | 30.29% | 28.23% | -2.06 pp | 69.12% |
| `0.785-0.813` | 351 | 27.92% | 26.90% | -1.02 pp | 73.50% |
| `>= 0.813` | 279 | 30.11% | 25.58% | -4.53 pp | 71.33% |

### Yasuo MIDDLE `crit` vs enemy siege

Same melee-carry-into-poke pattern across lane.

**Gap MSE** 2.10 pp^2 | **Mean abs gap** 1.26 pp | **Accuracy** 57.16% | **Accuracy if calibrated** 57.11% | **Calibration lift** -0.05 pp | **Empirical effect** -1.65 pp | **HGNN effect** -2.10 pp | **Shrinkage** 1.27x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.441` | 1,844 | 51.03% | 50.68% | -0.35 pp | 60.30% |
| `0.441-0.471` | 1,720 | 52.21% | 49.98% | -2.23 pp | 56.80% |
| `0.471-0.499` | 1,758 | 50.40% | 49.47% | -0.93 pp | 57.05% |
| `0.499-0.530` | 1,674 | 51.73% | 49.75% | -1.98 pp | 55.79% |
| `>= 0.530` | 1,687 | 49.38% | 48.59% | -0.79 pp | 55.54% |

### Ahri MIDDLE `ability_power` vs enemy scaling

AP mid into scaling enemy compositions.

**Gap MSE** 0.92 pp^2 | **Mean abs gap** 0.90 pp | **Accuracy** 57.43% | **Accuracy if calibrated** 57.37% | **Calibration lift** -0.06 pp | **Empirical effect** -4.27 pp | **HGNN effect** -1.91 pp | **Shrinkage** 0.45x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.829` | 3,162 | 52.88% | 51.91% | -0.97 pp | 57.84% |
| `0.829-0.841` | 3,036 | 50.49% | 50.04% | -0.45 pp | 58.20% |
| `0.841-0.852` | 3,198 | 48.69% | 49.79% | +1.11 pp | 57.35% |
| `0.852-0.863` | 2,784 | 48.89% | 49.45% | +0.56 pp | 56.54% |
| `>= 0.863` | 2,876 | 48.61% | 50.00% | +1.39 pp | 57.13% |

### Nautilus UTILITY `mr_tank` with ally damage

Engage support with damage behind it.

**Gap MSE** 3.15 pp^2 | **Mean abs gap** 1.53 pp | **Accuracy** 58.55% | **Accuracy if calibrated** 59.21% | **Calibration lift** +0.65 pp | **Empirical effect** +7.33 pp | **HGNN effect** +8.62 pp | **Shrinkage** 1.18x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 684 | 44.74% | 44.51% | -0.23 pp | 60.67% |
| `0.739-0.764` | 679 | 48.90% | 47.31% | -1.59 pp | 59.35% |
| `0.764-0.785` | 722 | 50.69% | 48.89% | -1.80 pp | 55.82% |
| `0.785-0.813` | 682 | 53.37% | 50.40% | -2.97 pp | 57.92% |
| `>= 0.813` | 290 | 52.07% | 53.13% | +1.06 pp | 60.00% |

### Galio MIDDLE `mr_tank` vs enemy magic

Anti-magic tank itemization (kept off-list MR-tank).

**Gap MSE** 3.12 pp^2 | **Mean abs gap** 1.07 pp | **Accuracy** 61.20% | **Accuracy if calibrated** 61.20% | **Calibration lift** +0.00 pp | **Empirical effect** +7.58 pp | **HGNN effect** +3.88 pp | **Shrinkage** 0.51x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 158 | 38.61% | 42.38% | +3.78 pp | 64.56% |
| `0.373-0.423` | 213 | 43.19% | 42.07% | -1.12 pp | 63.85% |
| `0.423-0.486` | 235 | 42.98% | 43.20% | +0.22 pp | 60.43% |
| `0.486-0.549` | 322 | 43.48% | 43.33% | -0.15 pp | 60.25% |
| `>= 0.549` | 433 | 46.19% | 46.26% | +0.07 pp | 59.82% |

### Malphite TOP `ar_tank` vs enemy physical

Armor tank into AD-heavy enemies.

**Gap MSE** 2.01 pp^2 | **Mean abs gap** 1.25 pp | **Accuracy** 57.54% | **Accuracy if calibrated** 57.40% | **Calibration lift** -0.13 pp | **Empirical effect** +10.33 pp | **HGNN effect** +9.25 pp | **Shrinkage** 0.90x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 669 | 45.44% | 46.29% | +0.85 pp | 59.49% |
| `0.387-0.448` | 806 | 45.91% | 47.25% | +1.34 pp | 54.09% |
| `0.448-0.508` | 1,238 | 49.19% | 50.78% | +1.59 pp | 56.87% |
| `0.508-0.557` | 1,544 | 50.71% | 52.94% | +2.23 pp | 57.45% |
| `>= 0.557` | 1,741 | 55.77% | 55.54% | -0.23 pp | 58.93% |

### Sylas MIDDLE `ability_power` vs enemy range

Short-range AP battlemage into enemy range pressure.

**Gap MSE** 3.88 pp^2 | **Mean abs gap** 1.79 pp | **Accuracy** 56.97% | **Accuracy if calibrated** 57.00% | **Calibration lift** +0.03 pp | **Empirical effect** -7.26 pp | **HGNN effect** -2.66 pp | **Shrinkage** 0.37x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 529 | 56.71% | 54.89% | -1.82 pp | 60.49% |
| `2` | 2,307 | 54.92% | 52.86% | -2.06 pp | 56.78% |
| `3` | 2,805 | 51.98% | 52.48% | +0.50 pp | 56.43% |
| `>= 4` | 813 | 49.45% | 52.23% | +2.78 pp | 57.07% |

### Nilah BOTTOM any build vs enemy range

Melee bot lane into range-heavy teams (kept off-list melee-ADC).

**Gap MSE** 7.36 pp^2 | **Mean abs gap** 2.39 pp | **Accuracy** 56.17% | **Accuracy if calibrated** 57.43% | **Calibration lift** +1.26 pp | **Empirical effect** -6.51 pp | **HGNN effect** -3.64 pp | **Shrinkage** 0.56x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 168 | 53.57% | 54.60% | +1.03 pp | 55.95% |
| `2` | 644 | 56.21% | 52.79% | -3.42 pp | 56.83% |
| `3` | 807 | 52.66% | 51.47% | -1.19 pp | 57.25% |
| `>= 4` | 204 | 47.06% | 50.96% | +3.90 pp | 50.00% |

### Kaisa BOTTOM any build vs enemy range

High-sample marksman vs enemy range pressure; large n keeps bins low-noise.

**Gap MSE** 7.68 pp^2 | **Mean abs gap** 2.21 pp | **Accuracy** 58.48% | **Accuracy if calibrated** 58.48% | **Calibration lift** +0.00 pp | **Empirical effect** +0.74 pp | **HGNN effect** -2.37 pp | **Shrinkage** -3.21x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 1,793 | 45.73% | 50.66% | +4.93 pp | 60.01% |
| `2` | 7,744 | 47.87% | 49.60% | +1.73 pp | 59.08% |
| `3` | 9,357 | 48.54% | 48.89% | +0.35 pp | 57.59% |
| `>= 4` | 2,707 | 46.47% | 48.29% | +1.82 pp | 58.81% |


## Richer Composition Trajectory Tables

### Kaisa BOTTOM `on_hit` vs enemy frontline count

On-hit marksman shreds added enemy frontline.

**Gap MSE** 1.04 pp^2 | **Mean abs gap** 0.90 pp | **Accuracy** 58.85% | **Accuracy if calibrated** 58.50% | **Calibration lift** -0.34 pp | **Empirical effect** +6.59 pp | **HGNN effect** +5.48 pp | **Shrinkage** 0.83x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,710 | 45.80% | 47.09% | +1.29 pp | 58.85% |
| `1` | 7,899 | 46.42% | 47.79% | +1.37 pp | 58.72% |
| `2` | 3,915 | 48.51% | 49.29% | +0.78 pp | 58.49% |
| `>= 3` | 733 | 52.39% | 52.57% | +0.18 pp | 62.07% |

### Ahri MIDDLE `ability_power` vs enemy frontline count

AP mid improves as enemies stack durable targets.

**Gap MSE** 1.67 pp^2 | **Mean abs gap** 0.95 pp | **Accuracy** 57.43% | **Accuracy if calibrated** 57.35% | **Calibration lift** -0.09 pp | **Empirical effect** +7.64 pp | **HGNN effect** +4.66 pp | **Shrinkage** 0.61x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,287 | 48.89% | 49.43% | +0.53 pp | 56.73% |
| `1` | 6,903 | 49.76% | 50.08% | +0.32 pp | 57.58% |
| `2` | 3,277 | 50.56% | 51.06% | +0.50 pp | 57.67% |
| `>= 3` | 589 | 56.54% | 54.08% | -2.45 pp | 59.42% |

### Sylas JUNGLE `ability_power` vs enemy frontline count

Sustained AP skirmisher into beefy teams.

**Gap MSE** 2.92 pp^2 | **Mean abs gap** 1.14 pp | **Accuracy** 56.61% | **Accuracy if calibrated** 56.35% | **Calibration lift** -0.25 pp | **Empirical effect** +9.37 pp | **HGNN effect** +5.08 pp | **Shrinkage** 0.54x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,480 | 49.86% | 50.91% | +1.04 pp | 56.22% |
| `1` | 2,359 | 51.17% | 51.29% | +0.12 pp | 57.19% |
| `2` | 1,123 | 52.89% | 53.03% | +0.14 pp | 54.50% |
| `>= 3` | 184 | 59.24% | 55.99% | -3.25 pp | 65.22% |

### Sylas MIDDLE `ability_power` vs enemy frontline count

Same AP anti-frontline pattern from lane.

**Gap MSE** 12.77 pp^2 | **Mean abs gap** 2.58 pp | **Accuracy** 56.97% | **Accuracy if calibrated** 57.33% | **Calibration lift** +0.36 pp | **Empirical effect** +11.72 pp | **HGNN effect** +3.01 pp | **Shrinkage** 0.26x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,781 | 50.42% | 52.34% | +1.92 pp | 55.53% |
| `1` | 2,941 | 53.55% | 52.58% | -0.98 pp | 57.40% |
| `2` | 1,489 | 53.93% | 53.30% | -0.62 pp | 56.68% |
| `>= 3` | 243 | 62.14% | 55.35% | -6.79 pp | 64.20% |

### Karma UTILITY any build vs enemy frontline count

Utility support gains value as enemies stack frontline to zone.

**Gap MSE** 9.85 pp^2 | **Mean abs gap** 2.65 pp | **Accuracy** 57.22% | **Accuracy if calibrated** 57.29% | **Calibration lift** +0.07 pp | **Empirical effect** +1.56 pp | **HGNN effect** +6.08 pp | **Shrinkage** 3.90x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 5,678 | 49.10% | 49.92% | +0.81 pp | 56.99% |
| `1` | 8,586 | 48.26% | 50.93% | +2.67 pp | 57.40% |
| `2` | 3,819 | 50.62% | 52.40% | +1.78 pp | 57.03% |
| `>= 3` | 679 | 50.66% | 56.00% | +5.33 pp | 58.03% |

### Vayne BOTTOM `on_hit` vs enemy frontline count

Classic anti-tank marksman pattern.

**Gap MSE** 22.14 pp^2 | **Mean abs gap** 3.47 pp | **Accuracy** 57.83% | **Accuracy if calibrated** 58.49% | **Calibration lift** +0.66 pp | **Empirical effect** +15.19 pp | **HGNN effect** +7.42 pp | **Shrinkage** 0.49x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 896 | 46.43% | 45.40% | -1.03 pp | 54.02% |
| `1` | 1,770 | 49.60% | 46.61% | -3.00 pp | 58.14% |
| `2` | 918 | 50.33% | 49.29% | -1.04 pp | 59.91% |
| `>= 3` | 198 | 61.62% | 52.82% | -8.80 pp | 62.63% |

### Thresh UTILITY `ar_tank` vs enemy burst count

Durable engage support punished by multiple burst threats.

**Gap MSE** 0.88 pp^2 | **Mean abs gap** 0.61 pp | **Accuracy** 57.05% | **Accuracy if calibrated** 57.03% | **Calibration lift** -0.03 pp | **Empirical effect** -4.39 pp | **HGNN effect** -4.15 pp | **Shrinkage** 0.94x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 995 | 49.35% | 49.38% | +0.04 pp | 56.98% |
| `1` | 1,769 | 48.95% | 48.67% | -0.29 pp | 56.59% |
| `2` | 778 | 45.76% | 47.59% | +1.83 pp | 57.46% |
| `>= 3` | 109 | 44.95% | 45.24% | +0.28 pp | 62.39% |

### Nautilus UTILITY `mr_tank` vs enemy burst count

High-HP engage tank loses into concentrated burst.

**Gap MSE** 10.54 pp^2 | **Mean abs gap** 2.53 pp | **Accuracy** 58.55% | **Accuracy if calibrated** 59.44% | **Calibration lift** +0.88 pp | **Empirical effect** +3.26 pp | **HGNN effect** -0.87 pp | **Shrinkage** -0.27x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 852 | 50.59% | 48.73% | -1.85 pp | 57.75% |
| `1` | 1,472 | 48.91% | 48.12% | -0.80 pp | 60.05% |
| `2` | 642 | 49.69% | 48.19% | -1.49 pp | 56.85% |
| `>= 3` | 91 | 53.85% | 47.86% | -5.99 pp | 53.85% |

### Zed MIDDLE `lethality` vs enemy burst count

Assassin into enemy burst stacking.

**Gap MSE** 27.57 pp^2 | **Mean abs gap** 3.69 pp | **Accuracy** 56.85% | **Accuracy if calibrated** 57.18% | **Calibration lift** +0.32 pp | **Empirical effect** -17.04 pp | **HGNN effect** -5.08 pp | **Shrinkage** 0.30x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,816 | 53.69% | 51.84% | -1.85 pp | 58.15% |
| `1` | 3,080 | 51.46% | 50.59% | -0.87 pp | 55.88% |
| `2` | 1,131 | 51.72% | 49.79% | -1.93 pp | 56.94% |
| `>= 3` | 161 | 36.65% | 46.76% | +10.12 pp | 60.25% |

### Nami UTILITY `utility_protection` vs enemy burst count

Protective enchanter punished by burst-heavy enemies.

**Gap MSE** 0.36 pp^2 | **Mean abs gap** 0.49 pp | **Accuracy** 57.20% | **Accuracy if calibrated** 57.41% | **Calibration lift** +0.21 pp | **Empirical effect** -1.85 pp | **HGNN effect** -2.94 pp | **Shrinkage** 1.59x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,342 | 51.27% | 51.41% | +0.15 pp | 56.96% |
| `1` | 7,172 | 49.85% | 50.56% | +0.71 pp | 57.10% |
| `2` | 3,063 | 49.76% | 49.59% | -0.17 pp | 57.98% |
| `>= 3` | 429 | 49.42% | 48.47% | -0.95 pp | 55.94% |

### Jinx BOTTOM `crit` vs enemy burst count

Fragile crit carry into burst-heavy enemies.

**Gap MSE** 0.79 pp^2 | **Mean abs gap** 0.78 pp | **Accuracy** 56.74% | **Accuracy if calibrated** 56.98% | **Calibration lift** +0.23 pp | **Empirical effect** -4.86 pp | **HGNN effect** -4.08 pp | **Shrinkage** 0.84x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,968 | 53.45% | 52.86% | -0.59 pp | 57.18% |
| `1` | 7,062 | 52.36% | 51.23% | -1.14 pp | 56.73% |
| `2` | 3,015 | 51.18% | 49.96% | -1.21 pp | 56.68% |
| `>= 3` | 426 | 48.59% | 48.79% | +0.20 pp | 53.29% |

### Malphite TOP `ar_tank` vs heavy damage-taken count

Armor tank loses into teams with multiple high-soak targets.

**Gap MSE** 8.91 pp^2 | **Mean abs gap** 2.14 pp | **Accuracy** 57.54% | **Accuracy if calibrated** 57.87% | **Calibration lift** +0.33 pp | **Empirical effect** -12.88 pp | **HGNN effect** -8.26 pp | **Shrinkage** 0.64x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,616 | 53.71% | 54.82% | +1.11 pp | 58.35% |
| `1` | 2,903 | 50.40% | 51.44% | +1.04 pp | 56.84% |
| `2` | 1,310 | 48.63% | 49.29% | +0.67 pp | 57.63% |
| `>= 3` | 169 | 40.83% | 46.56% | +5.74 pp | 60.95% |

### Viego JUNGLE any build vs enemy high-HP count

On-hit bruiser jungler into high-HP enemy teams.

**Gap MSE** 4.65 pp^2 | **Mean abs gap** 2.03 pp | **Accuracy** 56.95% | **Accuracy if calibrated** 57.16% | **Calibration lift** +0.22 pp | **Empirical effect** +1.45 pp | **HGNN effect** +2.17 pp | **Shrinkage** 1.50x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,051 | 48.48% | 45.79% | -2.69 pp | 57.52% |
| `1` | 5,723 | 48.87% | 46.28% | -2.60 pp | 56.84% |
| `2` | 3,742 | 47.35% | 46.47% | -0.89 pp | 56.36% |
| `>= 3` | 1,290 | 49.92% | 47.96% | -1.97 pp | 57.75% |


## Retained Prior And User-Requested Trajectory Tables

### Malphite all roles `ar_tank` vs enemy physical

Original armor-stack audit, retained beyond TOP-only.

**Gap MSE** 2.08 pp^2 | **Mean abs gap** 1.08 pp | **Accuracy** 57.18% | **Accuracy if calibrated** 57.17% | **Calibration lift** -0.01 pp | **Empirical effect** +10.24 pp | **HGNN effect** +9.26 pp | **Shrinkage** 0.90x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 728 | 45.47% | 46.26% | +0.79 pp | 58.93% |
| `0.387-0.448` | 920 | 47.17% | 47.14% | -0.03 pp | 53.48% |
| `0.448-0.508` | 1,376 | 48.69% | 50.61% | +1.92 pp | 56.83% |
| `0.508-0.557` | 1,720 | 50.23% | 52.69% | +2.45 pp | 57.09% |
| `>= 0.557` | 2,041 | 55.71% | 55.52% | -0.19 pp | 58.55% |

### Galio all roles `mr_tank` vs enemy magic

Original anti-magic tank family, broader than MIDDLE-only.

**Gap MSE** 4.26 pp^2 | **Mean abs gap** 1.68 pp | **Accuracy** 60.01% | **Accuracy if calibrated** 60.26% | **Calibration lift** +0.25 pp | **Empirical effect** +6.08 pp | **HGNN effect** +4.95 pp | **Shrinkage** 0.81x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 173 | 39.88% | 43.26% | +3.38 pp | 64.16% |
| `0.373-0.423` | 228 | 44.74% | 42.58% | -2.16 pp | 61.84% |
| `0.423-0.486` | 270 | 44.81% | 44.54% | -0.28 pp | 58.89% |
| `0.486-0.549` | 387 | 44.96% | 44.61% | -0.35 pp | 60.21% |
| `>= 0.549` | 570 | 45.96% | 48.21% | +2.25 pp | 58.42% |

### Nautilus all roles `mr_tank` vs enemy magic

Top-20 MR-tank anti-magic case alongside Galio.

**Gap MSE** 4.11 pp^2 | **Mean abs gap** 1.49 pp | **Accuracy** 58.67% | **Accuracy if calibrated** 59.35% | **Calibration lift** +0.68 pp | **Empirical effect** -1.91 pp | **HGNN effect** -2.08 pp | **Shrinkage** 1.09x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 218 | 51.38% | 50.85% | -0.53 pp | 59.17% |
| `0.373-0.423` | 399 | 52.13% | 48.54% | -3.59 pp | 55.14% |
| `0.423-0.486` | 507 | 48.52% | 48.53% | +0.01 pp | 58.19% |
| `0.486-0.549` | 757 | 49.01% | 46.38% | -2.63 pp | 56.67% |
| `>= 0.549` | 1,221 | 49.47% | 48.76% | -0.70 pp | 61.18% |

### Nautilus all roles `ar_tank` vs enemy physical

Physical-heavy enemy teams remain a support-tank check.

**Gap MSE** 0.80 pp^2 | **Mean abs gap** 0.85 pp | **Accuracy** 57.75% | **Accuracy if calibrated** 57.71% | **Calibration lift** -0.04 pp | **Empirical effect** +5.85 pp | **HGNN effect** +7.74 pp | **Shrinkage** 1.32x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 915 | 46.23% | 45.04% | -1.18 pp | 57.81% |
| `0.387-0.448` | 1,287 | 45.53% | 46.22% | +0.69 pp | 56.33% |
| `0.448-0.508` | 1,749 | 50.20% | 49.00% | -1.20 pp | 57.92% |
| `0.508-0.557` | 1,908 | 49.21% | 49.66% | +0.45 pp | 57.02% |
| `>= 0.557` | 2,210 | 52.08% | 52.79% | +0.71 pp | 59.05% |

### Darius TOP any build vs enemy range count

Static team range pressure, stronger than lane-only range.

**Gap MSE** 2.80 pp^2 | **Mean abs gap** 1.57 pp | **Accuracy** 57.94% | **Accuracy if calibrated** 57.81% | **Calibration lift** -0.12 pp | **Empirical effect** -4.82 pp | **HGNN effect** -3.47 pp | **Shrinkage** 0.72x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 662 | 50.91% | 52.10% | +1.19 pp | 58.31% |
| `2` | 2,985 | 48.84% | 49.91% | +1.07 pp | 58.43% |
| `3` | 3,478 | 50.60% | 49.13% | -1.48 pp | 57.50% |
| `>= 4` | 972 | 46.09% | 48.63% | +2.54 pp | 57.72% |

### Darius TOP any build vs same-role range

User-requested static melee/ranged lane audit.

**Gap MSE** 2.65 pp^2 | **Mean abs gap** 1.28 pp | **Accuracy** 57.94% | **Accuracy if calibrated** 57.96% | **Calibration lift** +0.02 pp | **Empirical effect** -3.15 pp | **HGNN effect** -0.60 pp | **Shrinkage** 0.19x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 250` | 6,747 | 49.96% | 49.70% | -0.26 pp | 58.22% |
| `> 250` | 1,350 | 46.81% | 49.10% | +2.29 pp | 56.52% |

### MasterYi JUNGLE any build vs enemy hard CC

User-requested low-CC audit; unique even though gap is modest.

**Gap MSE** 8.70 pp^2 | **Mean abs gap** 2.70 pp | **Accuracy** 56.67% | **Accuracy if calibrated** 56.93% | **Calibration lift** +0.26 pp | **Empirical effect** -0.26 pp | **HGNN effect** -3.23 pp | **Shrinkage** 12.45x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,050 | 53.17% | 51.41% | -1.76 pp | 57.80% |
| `1` | 3,261 | 51.64% | 49.72% | -1.92 pp | 56.61% |
| `2` | 1,636 | 51.22% | 48.86% | -2.36 pp | 55.62% |
| `>= 3` | 395 | 52.91% | 48.18% | -4.73 pp | 55.70% |

### Selected enchanters UTILITY with skirmish allies

Original enchanter-with-skirmishers synergy probe.

**Gap MSE** 0.93 pp^2 | **Mean abs gap** 0.80 pp | **Accuracy** 56.97% | **Accuracy if calibrated** 56.99% | **Calibration lift** +0.02 pp | **Empirical effect** +3.14 pp | **HGNN effect** +0.89 pp | **Shrinkage** 0.28x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 36,795 | 50.16% | 51.63% | +1.48 pp | 57.18% |
| `1` | 7,128 | 52.30% | 52.14% | -0.16 pp | 55.92% |
| `>= 2` | 319 | 53.29% | 52.52% | -0.77 pp | 55.80% |

### Low own-damage teams vs enemy heal/shield

Original low-damage into sustain audit.

**Gap MSE** 0.83 pp^2 | **Mean abs gap** 0.81 pp | **Accuracy** 58.62% | **Accuracy if calibrated** 58.72% | **Calibration lift** +0.10 pp | **Empirical effect** -1.63 pp | **HGNN effect** -2.74 pp | **Shrinkage** 1.68x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.028` | 11,038 | 49.66% | 50.62% | +0.97 pp | 59.27% |
| `0.028-0.077` | 10,736 | 48.64% | 49.27% | +0.62 pp | 58.39% |
| `0.077-0.200` | 10,473 | 46.57% | 47.97% | +1.40 pp | 58.72% |
| `0.200-0.202` | 11,951 | 47.11% | 48.03% | +0.92 pp | 58.77% |
| `>= 0.202` | 11,263 | 48.02% | 47.89% | -0.14 pp | 57.95% |

### Ambessa TOP `attack_damage` vs enemy damage

Durable bruiser into enemy damage pressure.

**Gap MSE** 7.74 pp^2 | **Mean abs gap** 2.71 pp | **Accuracy** 55.51% | **Accuracy if calibrated** 56.68% | **Calibration lift** +1.17 pp | **Empirical effect** -2.91 pp | **HGNN effect** -2.93 pp | **Shrinkage** 1.01x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 1,164 | 50.69% | 52.83% | +2.14 pp | 57.39% |
| `0.739-0.764` | 1,204 | 48.67% | 50.97% | +2.30 pp | 54.32% |
| `0.764-0.785` | 1,140 | 46.40% | 50.01% | +3.61 pp | 55.26% |
| `0.785-0.813` | 1,195 | 46.03% | 49.39% | +3.37 pp | 56.90% |
| `>= 0.813` | 1,260 | 47.78% | 49.90% | +2.12 pp | 53.81% |

### LeeSin JUNGLE `ad_off_tank` vs enemy magic

Bruiser jungler resisting magic-heavy enemies.

**Gap MSE** 14.90 pp^2 | **Mean abs gap** 3.18 pp | **Accuracy** 58.29% | **Accuracy if calibrated** 59.09% | **Calibration lift** +0.80 pp | **Empirical effect** +3.43 pp | **HGNN effect** -3.68 pp | **Shrinkage** -1.07x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 647 | 42.35% | 48.97% | +6.62 pp | 58.27% |
| `0.373-0.423` | 573 | 45.20% | 47.19% | +1.99 pp | 58.12% |
| `0.423-0.486` | 541 | 45.10% | 47.20% | +2.10 pp | 57.12% |
| `0.486-0.549` | 485 | 40.21% | 44.90% | +4.70 pp | 60.41% |
| `>= 0.549` | 367 | 45.78% | 45.30% | -0.48 pp | 57.49% |

### Thresh UTILITY `mr_tank` vs enemy magic

MR-tank support anti-magic case.

**Gap MSE** 15.46 pp^2 | **Mean abs gap** 3.68 pp | **Accuracy** 57.02% | **Accuracy if calibrated** 56.17% | **Calibration lift** -0.85 pp | **Empirical effect** +6.44 pp | **HGNN effect** +6.51 pp | **Shrinkage** 1.01x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 56 | 44.64% | 42.54% | -2.10 pp | 50.00% |
| `0.373-0.423` | 76 | 40.79% | 44.80% | +4.01 pp | 65.79% |
| `0.423-0.486` | 110 | 50.00% | 45.00% | -5.00 pp | 59.09% |
| `0.486-0.549` | 140 | 50.71% | 45.46% | -5.26 pp | 48.57% |
| `>= 0.549` | 323 | 51.08% | 49.05% | -2.03 pp | 59.13% |


## Inspected Lower-Signal Trajectory Tables

### Focus HP `<= 2309` vs enemy burst count

Broad HP-vs-burst check; useful but lower signal than champion-specific rows.

**Gap MSE** 0.17 pp^2 | **Mean abs gap** 0.34 pp | **Accuracy** 57.35% | **Accuracy if calibrated** 57.35% | **Calibration lift** +0.00 pp | **Empirical effect** -4.89 pp | **HGNN effect** -3.87 pp | **Shrinkage** 0.79x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 92,697 | 51.98% | 51.62% | -0.36 pp | 57.74% |
| `1` | 157,900 | 50.54% | 50.53% | -0.00 pp | 57.30% |
| `2` | 67,915 | 49.80% | 49.46% | -0.34 pp | 56.96% |
| `>= 3` | 9,477 | 47.08% | 47.74% | +0.66 pp | 56.98% |

### Focus HP `>= 2478` vs enemy burst count

High-HP slots also drop into burst stacks, so champion/build specificity matters.

**Gap MSE** 0.05 pp^2 | **Mean abs gap** 0.19 pp | **Accuracy** 57.55% | **Accuracy if calibrated** 57.57% | **Calibration lift** +0.02 pp | **Empirical effect** -3.84 pp | **HGNN effect** -3.84 pp | **Shrinkage** 1.00x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 103,541 | 50.77% | 50.84% | +0.07 pp | 57.63% |
| `1` | 174,445 | 49.39% | 49.60% | +0.21 pp | 57.61% |
| `2` | 74,569 | 48.20% | 48.60% | +0.40 pp | 57.18% |
| `>= 3` | 10,375 | 46.93% | 47.00% | +0.07 pp | 58.18% |

### Ahri MIDDLE `ability_power` vs heavy damage-taken count

AP mid vs multiple high-soak enemies; weaker axis than frontline count.

**Gap MSE** 2.31 pp^2 | **Mean abs gap** 1.15 pp | **Accuracy** 57.43% | **Accuracy if calibrated** 57.21% | **Calibration lift** -0.23 pp | **Empirical effect** +1.05 pp | **HGNN effect** -1.43 pp | **Shrinkage** -1.36x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,740 | 50.94% | 50.61% | -0.33 pp | 57.43% |
| `1` | 7,431 | 49.74% | 50.21% | +0.47 pp | 57.69% |
| `2` | 3,508 | 49.14% | 50.13% | +0.99 pp | 56.96% |
| `>= 3` | 377 | 51.99% | 49.17% | -2.82 pp | 56.76% |

### Kaisa BOTTOM `on_hit` vs heavy damage-taken count

On-hit marksman vs high-soak enemies; frontline count is the stronger cut.

**Gap MSE** 2.97 pp^2 | **Mean abs gap** 1.55 pp | **Accuracy** 58.85% | **Accuracy if calibrated** 58.71% | **Calibration lift** -0.13 pp | **Empirical effect** -3.34 pp | **HGNN effect** -1.50 pp | **Shrinkage** 0.45x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,269 | 47.72% | 48.74% | +1.02 pp | 58.09% |
| `1` | 8,375 | 47.03% | 48.17% | +1.14 pp | 58.90% |
| `2` | 4,151 | 46.40% | 47.56% | +1.16 pp | 59.43% |
| `>= 3` | 462 | 44.37% | 47.24% | +2.86 pp | 59.52% |


## Top-20 Matchup And Synergy Audits

### Yasuo MIDDLE `crit` with ally CC

Yasuo's ult chains off ally knock-ups; scales with team CC.

**Gap MSE** 5.10 pp^2 | **Mean abs gap** 1.82 pp | **Accuracy** 57.16% | **Accuracy if calibrated** 57.40% | **Calibration lift** +0.24 pp | **Empirical effect** +1.79 pp | **HGNN effect** +5.46 pp | **Shrinkage** 3.05x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.374` | 1,135 | 51.10% | 46.84% | -4.26 pp | 55.95% |
| `0.374-0.429` | 1,597 | 49.28% | 47.86% | -1.42 pp | 55.85% |
| `0.429-0.479` | 1,817 | 51.35% | 49.23% | -2.11 pp | 57.68% |
| `0.479-0.539` | 2,041 | 49.83% | 50.53% | +0.70 pp | 57.18% |
| `>= 0.539` | 2,093 | 52.89% | 52.30% | -0.59 pp | 58.34% |

### Jhin BOTTOM `crit` with ally CC

Immobile crit marksman; measured synergy with team CC is near flat.

**Gap MSE** 2.20 pp^2 | **Mean abs gap** 1.19 pp | **Accuracy** 57.01% | **Accuracy if calibrated** 56.88% | **Calibration lift** -0.14 pp | **Empirical effect** +0.04 pp | **HGNN effect** +3.44 pp | **Shrinkage** 91.18x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.374` | 889 | 48.93% | 46.04% | -2.89 pp | 58.72% |
| `0.374-0.429` | 2,204 | 48.28% | 47.08% | -1.20 pp | 56.35% |
| `0.429-0.479` | 3,424 | 48.19% | 47.63% | -0.56 pp | 57.13% |
| `0.479-0.539` | 4,940 | 47.89% | 48.69% | +0.80 pp | 58.12% |
| `>= 0.539` | 7,421 | 48.97% | 49.48% | +0.52 pp | 56.22% |

### Lulu UTILITY `utility_protection` with ally damage

Enchanter value rises with carry damage to amplify and peel for.

**Gap MSE** 3.59 pp^2 | **Mean abs gap** 1.75 pp | **Accuracy** 56.40% | **Accuracy if calibrated** 56.76% | **Calibration lift** +0.36 pp | **Empirical effect** +4.03 pp | **HGNN effect** +5.39 pp | **Shrinkage** 1.34x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 4,473 | 48.25% | 49.95% | +1.70 pp | 56.16% |
| `0.739-0.764` | 3,793 | 50.07% | 51.40% | +1.34 pp | 55.89% |
| `0.764-0.785` | 2,907 | 51.26% | 52.11% | +0.86 pp | 58.10% |
| `0.785-0.813` | 1,841 | 51.11% | 52.88% | +1.77 pp | 54.21% |
| `>= 0.813` | 329 | 52.28% | 55.34% | +3.06 pp | 62.92% |

### Ezreal BOTTOM `attack_damage` vs enemy hard CC

Skillshot poke marksman punished as enemy hard CC stacks.

**Gap MSE** 11.19 pp^2 | **Mean abs gap** 3.20 pp | **Accuracy** 57.22% | **Accuracy if calibrated** 57.71% | **Calibration lift** +0.49 pp | **Empirical effect** -0.62 pp | **HGNN effect** -1.54 pp | **Shrinkage** 2.48x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 8,028 | 47.58% | 52.13% | +4.54 pp | 56.45% |
| `1` | 12,577 | 48.25% | 50.97% | +2.72 pp | 57.69% |
| `2` | 6,129 | 48.51% | 50.40% | +1.89 pp | 57.12% |
| `>= 3` | 1,250 | 46.96% | 50.58% | +3.62 pp | 57.84% |

### Jayce TOP `attack_damage` vs enemy frontline count

Poke bruiser empirically holds up into frontline-heavy teams; model heavily shrinks the effect.

**Gap MSE** 0.54 pp^2 | **Mean abs gap** 0.65 pp | **Accuracy** 57.88% | **Accuracy if calibrated** 57.89% | **Calibration lift** +0.01 pp | **Empirical effect** +5.93 pp | **HGNN effect** +5.41 pp | **Shrinkage** 0.91x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,276 | 47.58% | 46.92% | -0.67 pp | 56.59% |
| `1` | 3,860 | 47.90% | 47.65% | -0.25 pp | 57.25% |
| `2` | 1,862 | 48.71% | 49.20% | +0.48 pp | 61.12% |
| `>= 3` | 342 | 53.51% | 52.32% | -1.19 pp | 55.85% |

### LeeSin JUNGLE `attack_damage` vs enemy scaling

Early-tempo bruiser jungler fades as enemy scaling rises.

**Gap MSE** 0.87 pp^2 | **Mean abs gap** 0.78 pp | **Accuracy** 56.78% | **Accuracy if calibrated** 57.03% | **Calibration lift** +0.25 pp | **Empirical effect** -4.30 pp | **HGNN effect** -3.34 pp | **Shrinkage** 0.78x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.829` | 2,907 | 52.98% | 53.22% | +0.25 pp | 56.90% |
| `0.829-0.841` | 2,882 | 49.31% | 50.77% | +1.46 pp | 57.88% |
| `0.841-0.852` | 3,073 | 49.66% | 49.83% | +0.18 pp | 57.63% |
| `0.852-0.863` | 2,808 | 48.72% | 49.54% | +0.82 pp | 55.88% |
| `>= 0.863` | 3,211 | 48.68% | 49.88% | +1.20 pp | 55.65% |

### Caitlyn BOTTOM `crit` vs enemy burst count

Immobile siege ADC punished by multiple burst and dive threats.

**Gap MSE** 2.62 pp^2 | **Mean abs gap** 1.60 pp | **Accuracy** 56.96% | **Accuracy if calibrated** 56.78% | **Calibration lift** -0.18 pp | **Empirical effect** -2.29 pp | **HGNN effect** -2.32 pp | **Shrinkage** 1.01x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,569 | 49.49% | 51.06% | +1.58 pp | 56.31% |
| `1` | 8,114 | 48.62% | 50.56% | +1.94 pp | 57.32% |
| `2` | 3,628 | 48.70% | 50.05% | +1.34 pp | 56.67% |
| `>= 3` | 517 | 47.20% | 48.75% | +1.55 pp | 59.19% |


## Overall Summary

Detailed audit tables above are rendered from the `val` split.

| Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 46 | 200 | 1.69 pp | 10.12 pp | 5.39 pp^2 | 57.50% | 57.54% | +0.05 pp |

| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,145,051 | 11,450,510 | 46 | 200 | 1.14 pp | 8.23 pp | 2.57 pp^2 | 57.66% | 57.69% | +0.03 pp |
| Validation | 143,131 | 1,431,310 | 46 | 200 | 1.69 pp | 10.12 pp | 5.39 pp^2 | 57.50% | 57.54% | +0.05 pp |
| Test | 143,131 | 1,431,310 | 46 | 200 | 1.66 pp | 8.37 pp | 4.95 pp^2 | 56.98% | 57.05% | +0.07 pp |

Gap MSE is `mean((HGNN_focus_WR - empirical_focus_WR)^2)` across populated threshold bins, rendered as percentage-points squared.

## Reproduction Commands

The checked-in report uses the focus-slot audit path. Checkpoints with semantic MoE slot deltas are scored with per-slot focus-side probabilities instead of one repeated match-level probability. Regenerate predictions from the selected checkpoint with `--refresh-predictions`; omit it to reuse the prediction cache for report-only updates.

```bash
uv run python -m app.ml.context_examples_audit \
  --context-cache-dir app/ml/data/cache \
  --model-cache-dir app/ml/data/cache \
  --model-path app/ml/data/hgnn_production_model.pt \
  --encoder-sidecar-path app/ml/data/experiments/semantic_identity_sidecar_compact.npz \
  --prediction-cache app/ml/data/experiments/semantic_architecture_compact_w10_freeze_seed4/convex_encoder_mix_seed4/audit_focus_side_probability.npy \
  --audit-split val \
  --output app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md \
  --refresh-predictions
```
