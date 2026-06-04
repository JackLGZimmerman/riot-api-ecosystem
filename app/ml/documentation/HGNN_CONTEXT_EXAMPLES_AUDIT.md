# HGNN Context Examples Audit

Updated: 2026-06-04.

This audit joins the empirical focus-side context examples to the trained semantic HGNN predictions for the same cached games. Each audit is its own table: one row per threshold bin reporting `n / empirical WR / HGNN WR / gap / accuracy`, with a per-table Gap MSE, accuracy, and the accuracy headroom from perfect calibration (`Calibration lift`) above it. Gap is `HGNN WR - empirical WR`; zero gap is the target.

## Scope And Threshold Definitions

- Context source: `app/ml/data/cache` side-row arrays, `val` split only.
- HGNN model: `app/ml/data/experiments/semantic_focus_reference_w3000_cont6/model.pt`.
- HGNN cache: `app/ml/data/cache`.
- Encoder sidecar artifact: `app/ml/data/experiments/semantic_identity_sidecar_full.npz`.
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
| Headline Trajectory Audit Tables | 10 | 47 | 1.90 pp | 6.33 pp | 5.94 pp^2 | 54.33% | 54.90% | +0.57 pp |
| Richer Composition Trajectory Tables | 13 | 52 | 1.77 pp | 11.64 pp | 7.31 pp^2 | 53.89% | 54.41% | +0.53 pp |
| Retained Prior And User-Requested Trajectory Tables | 12 | 53 | 1.29 pp | 9.51 pp | 4.22 pp^2 | 54.51% | 54.56% | +0.05 pp |
| Inspected Lower-Signal Trajectory Tables | 4 | 16 | 1.25 pp | 6.49 pp | 4.14 pp^2 | 53.97% | 54.01% | +0.04 pp |
| Top-20 Matchup And Synergy Audits | 7 | 32 | 1.73 pp | 5.82 pp | 4.96 pp^2 | 54.01% | 54.28% | +0.28 pp |

## Train, Validation, And Test Summary

These rows reuse the same audit specs and prediction cache, but evaluate the cached train, validation, and test ranges separately.

| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,145,051 | 11,450,510 | 46 | 200 | 1.08 pp | 6.56 pp | 2.66 pp^2 | 54.67% | 54.75% | +0.08 pp |
| Validation | 143,131 | 1,431,310 | 46 | 200 | 1.63 pp | 11.64 pp | 5.54 pp^2 | 54.06% | 54.20% | +0.15 pp |
| Test | 143,131 | 1,431,310 | 46 | 200 | 1.99 pp | 11.97 pp | 6.77 pp^2 | 53.91% | 54.04% | +0.13 pp |

## Enemy Count Tail Shrinkage

| Audit | Axis | Baseline bin | Tail bin | Empirical tail effect | HGNN tail effect | Shrinkage |
|---|---|---:|---:|---:|---:|---:|
| Sylas MIDDLE `ability_power` vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -7.26 pp | -0.05 pp | 0.01x |
| Nilah BOTTOM any build vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -6.51 pp | -7.94 pp | 1.22x |
| Kaisa BOTTOM any build vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | +0.74 pp | -3.44 pp | -4.65x |
| Kaisa BOTTOM `on_hit` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +6.59 pp | +6.34 pp | 0.96x |
| Ahri MIDDLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +7.64 pp | +4.88 pp | 0.64x |
| Sylas JUNGLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +9.37 pp | +5.18 pp | 0.55x |
| Sylas MIDDLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +11.72 pp | +2.59 pp | 0.22x |
| Karma UTILITY any build vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +1.56 pp | +3.37 pp | 2.16x |
| Vayne BOTTOM `on_hit` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +15.19 pp | +10.58 pp | 0.70x |
| Thresh UTILITY `ar_tank` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.39 pp | -6.87 pp | 1.56x |
| Nautilus UTILITY `mr_tank` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | +3.26 pp | -3.56 pp | -1.09x |
| Zed MIDDLE `lethality` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -17.04 pp | -5.16 pp | 0.30x |
| Nami UTILITY `utility_protection` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -1.85 pp | -6.02 pp | 3.26x |
| Jinx BOTTOM `crit` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.86 pp | -3.71 pp | 0.76x |
| Malphite TOP `ar_tank` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -12.88 pp | -11.07 pp | 0.86x |
| Viego JUNGLE any build vs enemy high-HP count | `enemy_high_hp_count` | `0` | `>= 3` | +1.45 pp | +2.93 pp | 2.03x |
| Darius TOP any build vs enemy range count | `enemy_ranged_count` | `<= 1` | `>= 4` | -4.82 pp | -5.06 pp | 1.05x |
| MasterYi JUNGLE any build vs enemy hard CC | `enemy_hard_cc_count` | `0` | `>= 3` | -0.26 pp | -2.73 pp | 10.51x |
| Focus HP `<= 2309` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.89 pp | -3.56 pp | 0.73x |
| Focus HP `>= 2478` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -3.84 pp | -4.07 pp | 1.06x |
| Ahri MIDDLE `ability_power` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | +1.05 pp | -0.63 pp | -0.60x |
| Kaisa BOTTOM `on_hit` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -3.34 pp | +2.47 pp | -0.74x |
| Ezreal BOTTOM `attack_damage` vs enemy hard CC | `enemy_hard_cc_count` | `0` | `>= 3` | -0.62 pp | +1.35 pp | -2.17x |
| Jayce TOP `attack_damage` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +5.93 pp | +1.38 pp | 0.23x |
| Caitlyn BOTTOM `crit` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -2.29 pp | -1.83 pp | 0.80x |

## Headline Trajectory Audit Tables

### Yasuo TOP `crit` vs enemy siege

Melee crit carry punished by poke and siege.

**Gap MSE** 4.10 pp^2 | **Mean abs gap** 1.69 pp | **Accuracy** 53.92% | **Accuracy if calibrated** 53.97% | **Calibration lift** +0.05 pp | **Empirical effect** -0.46 pp | **HGNN effect** -2.14 pp | **Shrinkage** 4.67x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.441` | 782 | 51.53% | 50.05% | -1.49 pp | 57.03% |
| `0.441-0.471` | 797 | 49.44% | 49.20% | -0.23 pp | 53.70% |
| `0.471-0.499` | 759 | 51.38% | 48.65% | -2.74 pp | 53.36% |
| `0.499-0.530` | 714 | 49.30% | 48.45% | -0.85 pp | 52.94% |
| `>= 0.530` | 650 | 51.08% | 47.91% | -3.17 pp | 52.15% |

### Graves JUNGLE `lethality` vs enemy damage

Burst jungler into high enemy damage.

**Gap MSE** 6.42 pp^2 | **Mean abs gap** 2.12 pp | **Accuracy** 67.17% | **Accuracy if calibrated** 67.11% | **Calibration lift** -0.05 pp | **Empirical effect** -11.09 pp | **HGNN effect** -14.11 pp | **Shrinkage** 1.27x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 483 | 41.20% | 39.36% | -1.84 pp | 60.46% |
| `0.739-0.764` | 408 | 33.09% | 34.34% | +1.25 pp | 66.91% |
| `0.764-0.785` | 340 | 30.29% | 31.54% | +1.25 pp | 69.71% |
| `0.785-0.813` | 351 | 27.92% | 29.32% | +1.40 pp | 72.08% |
| `>= 0.813` | 279 | 30.11% | 25.25% | -4.86 pp | 69.89% |

### Yasuo MIDDLE `crit` vs enemy siege

Same melee-carry-into-poke pattern across lane.

**Gap MSE** 11.25 pp^2 | **Mean abs gap** 3.23 pp | **Accuracy** 52.61% | **Accuracy if calibrated** 53.97% | **Calibration lift** +1.36 pp | **Empirical effect** -1.65 pp | **HGNN effect** -0.90 pp | **Shrinkage** 0.54x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.441` | 1,844 | 51.03% | 48.26% | -2.77 pp | 53.52% |
| `0.441-0.471` | 1,720 | 52.21% | 47.83% | -4.38 pp | 51.98% |
| `0.471-0.499` | 1,758 | 50.40% | 47.51% | -2.88 pp | 52.50% |
| `0.499-0.530` | 1,674 | 51.73% | 47.61% | -4.13 pp | 52.57% |
| `>= 0.530` | 1,687 | 49.38% | 47.36% | -2.01 pp | 52.40% |

### Ahri MIDDLE `ability_power` vs enemy scaling

AP mid into scaling enemy compositions.

**Gap MSE** 1.10 pp^2 | **Mean abs gap** 0.95 pp | **Accuracy** 53.92% | **Accuracy if calibrated** 54.20% | **Calibration lift** +0.28 pp | **Empirical effect** -4.27 pp | **HGNN effect** -2.15 pp | **Shrinkage** 0.50x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.829` | 3,162 | 52.88% | 51.75% | -1.12 pp | 54.40% |
| `0.829-0.841` | 3,036 | 50.49% | 50.66% | +0.16 pp | 53.72% |
| `0.841-0.852` | 3,198 | 48.69% | 50.21% | +1.53 pp | 54.41% |
| `0.852-0.863` | 2,784 | 48.89% | 49.82% | +0.93 pp | 53.56% |
| `>= 0.863` | 2,876 | 48.61% | 49.61% | +1.00 pp | 53.41% |

### Nautilus UTILITY `mr_tank` with ally damage

Engage support with damage behind it.

**Gap MSE** 2.94 pp^2 | **Mean abs gap** 1.46 pp | **Accuracy** 53.42% | **Accuracy if calibrated** 52.67% | **Calibration lift** -0.75 pp | **Empirical effect** +7.33 pp | **HGNN effect** +10.22 pp | **Shrinkage** 1.39x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 684 | 44.74% | 44.72% | -0.01 pp | 55.56% |
| `0.739-0.764` | 679 | 48.90% | 47.51% | -1.39 pp | 55.08% |
| `0.764-0.785` | 722 | 50.69% | 49.29% | -1.40 pp | 49.58% |
| `0.785-0.813` | 682 | 53.37% | 51.77% | -1.60 pp | 53.37% |
| `>= 0.813` | 290 | 52.07% | 54.94% | +2.87 pp | 54.14% |

### Galio MIDDLE `mr_tank` vs enemy magic

Anti-magic tank itemization (kept off-list MR-tank).

**Gap MSE** 2.22 pp^2 | **Mean abs gap** 1.17 pp | **Accuracy** 57.83% | **Accuracy if calibrated** 56.80% | **Calibration lift** -1.03 pp | **Empirical effect** +7.58 pp | **HGNN effect** +9.00 pp | **Shrinkage** 1.19x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 158 | 38.61% | 38.56% | -0.05 pp | 61.39% |
| `0.373-0.423` | 213 | 43.19% | 40.42% | -2.78 pp | 56.81% |
| `0.423-0.486` | 235 | 42.98% | 41.88% | -1.09 pp | 57.02% |
| `0.486-0.549` | 322 | 43.48% | 44.06% | +0.58 pp | 58.07% |
| `>= 0.549` | 433 | 46.19% | 47.56% | +1.37 pp | 57.27% |

### Malphite TOP `ar_tank` vs enemy physical

Armor tank into AD-heavy enemies.

**Gap MSE** 0.91 pp^2 | **Mean abs gap** 0.63 pp | **Accuracy** 54.90% | **Accuracy if calibrated** 54.63% | **Calibration lift** -0.27 pp | **Empirical effect** +10.33 pp | **HGNN effect** +8.51 pp | **Shrinkage** 0.82x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 669 | 45.44% | 45.35% | -0.09 pp | 57.10% |
| `0.387-0.448` | 806 | 45.91% | 46.83% | +0.93 pp | 55.71% |
| `0.448-0.508` | 1,238 | 49.19% | 48.98% | -0.21 pp | 54.44% |
| `0.508-0.557` | 1,544 | 50.71% | 50.72% | +0.00 pp | 51.94% |
| `>= 0.557` | 1,741 | 55.77% | 53.86% | -1.91 pp | 56.63% |

### Sylas MIDDLE `ability_power` vs enemy range

Short-range AP battlemage into enemy range pressure.

**Gap MSE** 7.93 pp^2 | **Mean abs gap** 2.55 pp | **Accuracy** 54.11% | **Accuracy if calibrated** 54.09% | **Calibration lift** -0.02 pp | **Empirical effect** -7.26 pp | **HGNN effect** -0.05 pp | **Shrinkage** 0.01x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 529 | 56.71% | 53.71% | -3.00 pp | 57.47% |
| `2` | 2,307 | 54.92% | 52.92% | -2.00 pp | 55.83% |
| `3` | 2,805 | 51.98% | 52.96% | +0.98 pp | 52.55% |
| `>= 4` | 813 | 49.45% | 53.66% | +4.21 pp | 52.40% |

### Nilah BOTTOM any build vs enemy range

Melee bot lane into range-heavy teams (kept off-list melee-ADC).

**Gap MSE** 12.31 pp^2 | **Mean abs gap** 2.68 pp | **Accuracy** 54.80% | **Accuracy if calibrated** 55.68% | **Calibration lift** +0.88 pp | **Empirical effect** -6.51 pp | **HGNN effect** -7.94 pp | **Shrinkage** 1.22x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 168 | 53.57% | 59.16% | +5.59 pp | 54.17% |
| `2` | 644 | 56.21% | 55.36% | -0.85 pp | 57.14% |
| `3` | 807 | 52.66% | 52.79% | +0.12 pp | 53.16% |
| `>= 4` | 204 | 47.06% | 51.22% | +4.16 pp | 54.41% |

### Kaisa BOTTOM any build vs enemy range

High-sample marksman vs enemy range pressure; large n keeps bins low-noise.

**Gap MSE** 13.41 pp^2 | **Mean abs gap** 3.07 pp | **Accuracy** 54.05% | **Accuracy if calibrated** 55.32% | **Calibration lift** +1.27 pp | **Empirical effect** +0.74 pp | **HGNN effect** -3.44 pp | **Shrinkage** -4.65x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 1,793 | 45.73% | 52.06% | +6.33 pp | 51.14% |
| `2` | 7,744 | 47.87% | 50.70% | +2.83 pp | 54.38% |
| `3` | 9,357 | 48.54% | 49.52% | +0.98 pp | 54.23% |
| `>= 4` | 2,707 | 46.47% | 48.62% | +2.15 pp | 54.41% |


## Richer Composition Trajectory Tables

### Kaisa BOTTOM `on_hit` vs enemy frontline count

On-hit marksman shreds added enemy frontline.

**Gap MSE** 3.72 pp^2 | **Mean abs gap** 1.84 pp | **Accuracy** 54.27% | **Accuracy if calibrated** 55.17% | **Calibration lift** +0.90 pp | **Empirical effect** +6.59 pp | **HGNN effect** +6.34 pp | **Shrinkage** 0.96x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,710 | 45.80% | 47.21% | +1.41 pp | 54.76% |
| `1` | 7,899 | 46.42% | 48.87% | +2.45 pp | 54.34% |
| `2` | 3,915 | 48.51% | 50.87% | +2.36 pp | 53.28% |
| `>= 3` | 733 | 52.39% | 53.55% | +1.16 pp | 55.66% |

### Ahri MIDDLE `ability_power` vs enemy frontline count

AP mid improves as enemies stack durable targets.

**Gap MSE** 2.15 pp^2 | **Mean abs gap** 1.15 pp | **Accuracy** 53.92% | **Accuracy if calibrated** 54.22% | **Calibration lift** +0.30 pp | **Empirical effect** +7.64 pp | **HGNN effect** +4.88 pp | **Shrinkage** 0.64x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,287 | 48.89% | 49.08% | +0.19 pp | 55.54% |
| `1` | 6,903 | 49.76% | 50.31% | +0.54 pp | 52.98% |
| `2` | 3,277 | 50.56% | 51.85% | +1.29 pp | 53.25% |
| `>= 3` | 589 | 56.54% | 53.97% | -2.57 pp | 56.88% |

### Sylas JUNGLE `ability_power` vs enemy frontline count

Sustained AP skirmisher into beefy teams.

**Gap MSE** 5.01 pp^2 | **Mean abs gap** 1.35 pp | **Accuracy** 53.23% | **Accuracy if calibrated** 53.56% | **Calibration lift** +0.33 pp | **Empirical effect** +9.37 pp | **HGNN effect** +5.18 pp | **Shrinkage** 0.55x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,480 | 49.86% | 49.62% | -0.25 pp | 51.08% |
| `1` | 2,359 | 51.17% | 50.78% | -0.39 pp | 54.05% |
| `2` | 1,123 | 52.89% | 52.59% | -0.31 pp | 52.98% |
| `>= 3` | 184 | 59.24% | 54.80% | -4.44 pp | 61.41% |

### Sylas MIDDLE `ability_power` vs enemy frontline count

Same AP anti-frontline pattern from lane.

**Gap MSE** 13.53 pp^2 | **Mean abs gap** 2.48 pp | **Accuracy** 54.11% | **Accuracy if calibrated** 54.26% | **Calibration lift** +0.15 pp | **Empirical effect** +11.72 pp | **HGNN effect** +2.59 pp | **Shrinkage** 0.22x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,781 | 50.42% | 52.53% | +2.11 pp | 52.16% |
| `1` | 2,941 | 53.55% | 52.92% | -0.63 pp | 54.54% |
| `2` | 1,489 | 53.93% | 53.79% | -0.14 pp | 54.00% |
| `>= 3` | 243 | 62.14% | 55.12% | -7.02 pp | 63.79% |

### Karma UTILITY any build vs enemy frontline count

Utility support gains value as enemies stack frontline to zone.

**Gap MSE** 1.32 pp^2 | **Mean abs gap** 0.87 pp | **Accuracy** 53.56% | **Accuracy if calibrated** 54.21% | **Calibration lift** +0.64 pp | **Empirical effect** +1.56 pp | **HGNN effect** +3.37 pp | **Shrinkage** 2.16x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 5,678 | 49.10% | 48.98% | -0.12 pp | 53.79% |
| `1` | 8,586 | 48.26% | 49.81% | +1.54 pp | 53.31% |
| `2` | 3,819 | 50.62% | 50.73% | +0.11 pp | 54.02% |
| `>= 3` | 679 | 50.66% | 52.35% | +1.69 pp | 52.28% |

### Vayne BOTTOM `on_hit` vs enemy frontline count

Classic anti-tank marksman pattern.

**Gap MSE** 7.26 pp^2 | **Mean abs gap** 2.51 pp | **Accuracy** 54.15% | **Accuracy if calibrated** 55.42% | **Calibration lift** +1.27 pp | **Empirical effect** +15.19 pp | **HGNN effect** +10.58 pp | **Shrinkage** 0.70x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 896 | 46.43% | 48.42% | +2.00 pp | 53.68% |
| `1` | 1,770 | 49.60% | 50.98% | +1.38 pp | 54.75% |
| `2` | 918 | 50.33% | 54.37% | +4.04 pp | 51.96% |
| `>= 3` | 198 | 61.62% | 59.01% | -2.61 pp | 61.11% |

### Thresh UTILITY `ar_tank` vs enemy burst count

Durable engage support punished by multiple burst threats.

**Gap MSE** 1.43 pp^2 | **Mean abs gap** 1.12 pp | **Accuracy** 53.14% | **Accuracy if calibrated** 54.48% | **Calibration lift** +1.34 pp | **Empirical effect** -4.39 pp | **HGNN effect** -6.87 pp | **Shrinkage** 1.56x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 995 | 49.35% | 50.77% | +1.42 pp | 52.66% |
| `1` | 1,769 | 48.95% | 49.41% | +0.45 pp | 52.35% |
| `2` | 778 | 45.76% | 47.30% | +1.54 pp | 55.01% |
| `>= 3` | 109 | 44.95% | 43.90% | -1.06 pp | 56.88% |

### Nautilus UTILITY `mr_tank` vs enemy burst count

High-HP engage tank loses into concentrated burst.

**Gap MSE** 16.14 pp^2 | **Mean abs gap** 2.69 pp | **Accuracy** 53.42% | **Accuracy if calibrated** 52.70% | **Calibration lift** -0.72 pp | **Empirical effect** +3.26 pp | **HGNN effect** -3.56 pp | **Shrinkage** -1.09x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 852 | 50.59% | 49.62% | -0.97 pp | 53.52% |
| `1` | 1,472 | 48.91% | 49.20% | +0.28 pp | 53.87% |
| `2` | 642 | 49.69% | 47.96% | -1.73 pp | 52.18% |
| `>= 3` | 91 | 53.85% | 46.06% | -7.78 pp | 53.85% |

### Zed MIDDLE `lethality` vs enemy burst count

Assassin into enemy burst stacking.

**Gap MSE** 34.29 pp^2 | **Mean abs gap** 3.42 pp | **Accuracy** 54.07% | **Accuracy if calibrated** 54.43% | **Calibration lift** +0.36 pp | **Empirical effect** -17.04 pp | **HGNN effect** -5.16 pp | **Shrinkage** 0.30x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,816 | 53.69% | 53.44% | -0.25 pp | 54.19% |
| `1` | 3,080 | 51.46% | 52.54% | +1.07 pp | 53.51% |
| `2` | 1,131 | 51.72% | 51.01% | -0.71 pp | 54.64% |
| `>= 3` | 161 | 36.65% | 48.28% | +11.64 pp | 59.63% |

### Nami UTILITY `utility_protection` vs enemy burst count

Protective enchanter punished by burst-heavy enemies.

**Gap MSE** 2.92 pp^2 | **Mean abs gap** 1.51 pp | **Accuracy** 53.57% | **Accuracy if calibrated** 54.13% | **Calibration lift** +0.56 pp | **Empirical effect** -1.85 pp | **HGNN effect** -6.02 pp | **Shrinkage** 3.26x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,342 | 51.27% | 52.62% | +1.35 pp | 53.71% |
| `1` | 7,172 | 49.85% | 51.06% | +1.21 pp | 53.72% |
| `2` | 3,063 | 49.76% | 49.11% | -0.65 pp | 53.25% |
| `>= 3` | 429 | 49.42% | 46.60% | -2.82 pp | 51.75% |

### Jinx BOTTOM `crit` vs enemy burst count

Fragile crit carry into burst-heavy enemies.

**Gap MSE** 3.64 pp^2 | **Mean abs gap** 1.86 pp | **Accuracy** 53.36% | **Accuracy if calibrated** 54.04% | **Calibration lift** +0.68 pp | **Empirical effect** -4.86 pp | **HGNN effect** -3.71 pp | **Shrinkage** 0.76x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,968 | 53.45% | 54.90% | +1.44 pp | 54.51% |
| `1` | 7,062 | 52.36% | 54.00% | +1.64 pp | 53.17% |
| `2` | 3,015 | 51.18% | 52.93% | +1.75 pp | 52.90% |
| `>= 3` | 426 | 48.59% | 51.18% | +2.59 pp | 49.06% |

### Malphite TOP `ar_tank` vs heavy damage-taken count

Armor tank loses into teams with multiple high-soak targets.

**Gap MSE** 2.41 pp^2 | **Mean abs gap** 1.27 pp | **Accuracy** 54.90% | **Accuracy if calibrated** 55.14% | **Calibration lift** +0.23 pp | **Empirical effect** -12.88 pp | **HGNN effect** -11.07 pp | **Shrinkage** 0.86x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,616 | 53.71% | 54.09% | +0.37 pp | 55.20% |
| `1` | 2,903 | 50.40% | 50.03% | -0.36 pp | 55.12% |
| `2` | 1,310 | 48.63% | 46.48% | -2.15 pp | 53.44% |
| `>= 3` | 169 | 40.83% | 43.02% | +2.19 pp | 59.76% |

### Viego JUNGLE any build vs enemy high-HP count

On-hit bruiser jungler into high-HP enemy teams.

**Gap MSE** 1.16 pp^2 | **Mean abs gap** 0.95 pp | **Accuracy** 54.59% | **Accuracy if calibrated** 54.84% | **Calibration lift** +0.25 pp | **Empirical effect** +1.45 pp | **HGNN effect** +2.93 pp | **Shrinkage** 2.03x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,051 | 48.48% | 47.11% | -1.37 pp | 54.61% |
| `1` | 5,723 | 48.87% | 47.89% | -0.99 pp | 54.66% |
| `2` | 3,742 | 47.35% | 48.69% | +1.33 pp | 54.97% |
| `>= 3` | 1,290 | 49.92% | 50.04% | +0.12 pp | 53.18% |


## Retained Prior And User-Requested Trajectory Tables

### Malphite all roles `ar_tank` vs enemy physical

Original armor-stack audit, retained beyond TOP-only.

**Gap MSE** 0.52 pp^2 | **Mean abs gap** 0.54 pp | **Accuracy** 54.49% | **Accuracy if calibrated** 54.46% | **Calibration lift** -0.03 pp | **Empirical effect** +10.24 pp | **HGNN effect** +8.86 pp | **Shrinkage** 0.86x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 728 | 45.47% | 45.51% | +0.04 pp | 56.87% |
| `0.387-0.448` | 920 | 47.17% | 47.07% | -0.10 pp | 54.46% |
| `0.448-0.508` | 1,376 | 48.69% | 49.20% | +0.51 pp | 53.85% |
| `0.508-0.557` | 1,720 | 50.23% | 50.94% | +0.71 pp | 51.51% |
| `>= 0.557` | 2,041 | 55.71% | 54.37% | -1.34 pp | 56.59% |

### Galio all roles `mr_tank` vs enemy magic

Original anti-magic tank family, broader than MIDDLE-only.

**Gap MSE** 5.55 pp^2 | **Mean abs gap** 1.96 pp | **Accuracy** 57.13% | **Accuracy if calibrated** 57.13% | **Calibration lift** +0.00 pp | **Empirical effect** +6.08 pp | **HGNN effect** +9.28 pp | **Shrinkage** 1.53x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 173 | 39.88% | 39.01% | -0.87 pp | 60.12% |
| `0.373-0.423` | 228 | 44.74% | 40.73% | -4.01 pp | 55.26% |
| `0.423-0.486` | 270 | 44.81% | 42.48% | -2.33 pp | 54.81% |
| `0.486-0.549` | 387 | 44.96% | 44.72% | -0.24 pp | 57.62% |
| `>= 0.549` | 570 | 45.96% | 48.28% | +2.32 pp | 57.72% |

### Nautilus all roles `mr_tank` vs enemy magic

Top-20 MR-tank anti-magic case alongside Galio.

**Gap MSE** 5.44 pp^2 | **Mean abs gap** 1.64 pp | **Accuracy** 53.51% | **Accuracy if calibrated** 53.38% | **Calibration lift** -0.13 pp | **Empirical effect** -1.91 pp | **HGNN effect** +1.52 pp | **Shrinkage** -0.79x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 218 | 51.38% | 48.33% | -3.05 pp | 54.59% |
| `0.373-0.423` | 399 | 52.13% | 47.95% | -4.18 pp | 50.88% |
| `0.423-0.486` | 507 | 48.52% | 48.44% | -0.08 pp | 53.45% |
| `0.486-0.549` | 757 | 49.01% | 48.51% | -0.50 pp | 53.63% |
| `>= 0.549` | 1,221 | 49.47% | 49.84% | +0.38 pp | 54.14% |

### Nautilus all roles `ar_tank` vs enemy physical

Physical-heavy enemy teams remain a support-tank check.

**Gap MSE** 1.16 pp^2 | **Mean abs gap** 0.83 pp | **Accuracy** 54.52% | **Accuracy if calibrated** 54.37% | **Calibration lift** -0.15 pp | **Empirical effect** +5.85 pp | **HGNN effect** +4.87 pp | **Shrinkage** 0.83x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 915 | 46.23% | 46.23% | -0.00 pp | 57.38% |
| `0.387-0.448` | 1,287 | 45.53% | 47.33% | +1.80 pp | 54.70% |
| `0.448-0.508` | 1,749 | 50.20% | 48.95% | -1.25 pp | 55.80% |
| `0.508-0.557` | 1,908 | 49.21% | 49.35% | +0.13 pp | 52.57% |
| `>= 0.557` | 2,210 | 52.08% | 51.10% | -0.98 pp | 53.89% |

### Darius TOP any build vs enemy range count

Static team range pressure, stronger than lane-only range.

**Gap MSE** 1.93 pp^2 | **Mean abs gap** 1.38 pp | **Accuracy** 54.84% | **Accuracy if calibrated** 55.37% | **Calibration lift** +0.53 pp | **Empirical effect** -4.82 pp | **HGNN effect** -5.06 pp | **Shrinkage** 1.05x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 662 | 50.91% | 52.25% | +1.34 pp | 55.44% |
| `2` | 2,985 | 48.84% | 50.27% | +1.43 pp | 54.34% |
| `3` | 3,478 | 50.60% | 48.97% | -1.63 pp | 54.63% |
| `>= 4` | 972 | 46.09% | 47.19% | +1.10 pp | 56.69% |

### Darius TOP any build vs same-role range

User-requested static melee/ranged lane audit.

**Gap MSE** 0.02 pp^2 | **Mean abs gap** 0.11 pp | **Accuracy** 54.84% | **Accuracy if calibrated** 54.90% | **Calibration lift** +0.06 pp | **Empirical effect** -3.15 pp | **HGNN effect** -3.01 pp | **Shrinkage** 0.96x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 250` | 6,747 | 49.96% | 50.01% | +0.04 pp | 54.69% |
| `> 250` | 1,350 | 46.81% | 46.99% | +0.18 pp | 55.56% |

### MasterYi JUNGLE any build vs enemy hard CC

User-requested low-CC audit; unique even though gap is modest.

**Gap MSE** 1.46 pp^2 | **Mean abs gap** 0.87 pp | **Accuracy** 53.24% | **Accuracy if calibrated** 53.38% | **Calibration lift** +0.14 pp | **Empirical effect** -0.26 pp | **HGNN effect** -2.73 pp | **Shrinkage** 10.51x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,050 | 53.17% | 53.37% | +0.20 pp | 55.17% |
| `1` | 3,261 | 51.64% | 52.40% | +0.76 pp | 52.99% |
| `2` | 1,636 | 51.22% | 51.48% | +0.26 pp | 50.92% |
| `>= 3` | 395 | 52.91% | 50.64% | -2.27 pp | 54.94% |

### Selected enchanters UTILITY with skirmish allies

Original enchanter-with-skirmishers synergy probe.

**Gap MSE** 0.16 pp^2 | **Mean abs gap** 0.39 pp | **Accuracy** 53.71% | **Accuracy if calibrated** 53.78% | **Calibration lift** +0.07 pp | **Empirical effect** +3.14 pp | **HGNN effect** +2.36 pp | **Shrinkage** 0.75x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 36,795 | 50.16% | 50.52% | +0.36 pp | 53.77% |
| `1` | 7,128 | 52.30% | 51.90% | -0.40 pp | 53.31% |
| `>= 2` | 319 | 53.29% | 52.87% | -0.42 pp | 55.17% |

### Low own-damage teams vs enemy heal/shield

Original low-damage into sustain audit.

**Gap MSE** 0.57 pp^2 | **Mean abs gap** 0.62 pp | **Accuracy** 55.06% | **Accuracy if calibrated** 55.05% | **Calibration lift** -0.01 pp | **Empirical effect** -1.63 pp | **HGNN effect** -1.63 pp | **Shrinkage** 1.00x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.028` | 11,038 | 49.66% | 49.11% | -0.55 pp | 54.95% |
| `0.028-0.077` | 10,736 | 48.64% | 48.53% | -0.11 pp | 55.57% |
| `0.077-0.200` | 10,473 | 46.57% | 48.00% | +1.43 pp | 55.04% |
| `0.200-0.202` | 11,951 | 47.11% | 47.58% | +0.47 pp | 55.18% |
| `>= 0.202` | 11,263 | 48.02% | 47.48% | -0.54 pp | 54.57% |

### Ambessa TOP `attack_damage` vs enemy damage

Durable bruiser into enemy damage pressure.

**Gap MSE** 1.32 pp^2 | **Mean abs gap** 0.96 pp | **Accuracy** 54.59% | **Accuracy if calibrated** 54.87% | **Calibration lift** +0.29 pp | **Empirical effect** -2.91 pp | **HGNN effect** -2.87 pp | **Shrinkage** 0.98x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 1,164 | 50.69% | 49.97% | -0.72 pp | 54.21% |
| `0.739-0.764` | 1,204 | 48.67% | 48.75% | +0.08 pp | 53.82% |
| `0.764-0.785` | 1,140 | 46.40% | 48.17% | +1.76 pp | 53.60% |
| `0.785-0.813` | 1,195 | 46.03% | 47.60% | +1.58 pp | 57.15% |
| `>= 0.813` | 1,260 | 47.78% | 47.10% | -0.68 pp | 54.13% |

### LeeSin JUNGLE `ad_off_tank` vs enemy magic

Bruiser jungler resisting magic-heavy enemies.

**Gap MSE** 5.97 pp^2 | **Mean abs gap** 1.96 pp | **Accuracy** 57.67% | **Accuracy if calibrated** 57.25% | **Calibration lift** -0.42 pp | **Empirical effect** +3.43 pp | **HGNN effect** -0.40 pp | **Shrinkage** -0.12x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 647 | 42.35% | 45.07% | +2.73 pp | 58.11% |
| `0.373-0.423` | 573 | 45.20% | 44.51% | -0.69 pp | 55.50% |
| `0.423-0.486` | 541 | 45.10% | 44.31% | -0.79 pp | 57.49% |
| `0.486-0.549` | 485 | 40.21% | 44.69% | +4.48 pp | 60.62% |
| `>= 0.549` | 367 | 45.78% | 44.67% | -1.11 pp | 56.68% |

### Thresh UTILITY `mr_tank` vs enemy magic

MR-tank support anti-magic case.

**Gap MSE** 21.41 pp^2 | **Mean abs gap** 3.12 pp | **Accuracy** 54.04% | **Accuracy if calibrated** 53.90% | **Calibration lift** -0.14 pp | **Empirical effect** +6.44 pp | **HGNN effect** +4.01 pp | **Shrinkage** 0.62x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 56 | 44.64% | 48.43% | +3.79 pp | 48.21% |
| `0.373-0.423` | 76 | 40.79% | 50.30% | +9.51 pp | 56.58% |
| `0.423-0.486` | 110 | 50.00% | 49.40% | -0.60 pp | 50.91% |
| `0.486-0.549` | 140 | 50.71% | 50.38% | -0.34 pp | 48.57% |
| `>= 0.549` | 323 | 51.08% | 52.44% | +1.36 pp | 57.89% |


## Inspected Lower-Signal Trajectory Tables

### Focus HP `<= 2309` vs enemy burst count

Broad HP-vs-burst check; useful but lower signal than champion-specific rows.

**Gap MSE** 0.26 pp^2 | **Mean abs gap** 0.45 pp | **Accuracy** 53.85% | **Accuracy if calibrated** 53.88% | **Calibration lift** +0.03 pp | **Empirical effect** -4.89 pp | **HGNN effect** -3.56 pp | **Shrinkage** 0.73x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 92,697 | 51.98% | 51.40% | -0.57 pp | 54.37% |
| `1` | 157,900 | 50.54% | 50.91% | +0.37 pp | 53.69% |
| `2` | 67,915 | 49.80% | 49.70% | -0.10 pp | 53.38% |
| `>= 3` | 9,477 | 47.08% | 47.84% | +0.76 pp | 54.80% |

### Focus HP `>= 2478` vs enemy burst count

High-HP slots also drop into burst stacks, so champion/build specificity matters.

**Gap MSE** 0.15 pp^2 | **Mean abs gap** 0.34 pp | **Accuracy** 54.07% | **Accuracy if calibrated** 54.09% | **Calibration lift** +0.02 pp | **Empirical effect** -3.84 pp | **HGNN effect** -4.07 pp | **Shrinkage** 1.06x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 103,541 | 50.77% | 50.39% | -0.38 pp | 54.14% |
| `1` | 174,445 | 49.39% | 49.64% | +0.24 pp | 54.19% |
| `2` | 74,569 | 48.20% | 48.32% | +0.12 pp | 53.59% |
| `>= 3` | 10,375 | 46.93% | 46.32% | -0.61 pp | 54.75% |

### Ahri MIDDLE `ability_power` vs heavy damage-taken count

AP mid vs multiple high-soak enemies; weaker axis than frontline count.

**Gap MSE** 1.56 pp^2 | **Mean abs gap** 1.08 pp | **Accuracy** 53.92% | **Accuracy if calibrated** 53.98% | **Calibration lift** +0.06 pp | **Empirical effect** +1.05 pp | **HGNN effect** -0.63 pp | **Shrinkage** -0.60x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,740 | 50.94% | 50.57% | -0.37 pp | 53.96% |
| `1` | 7,431 | 49.74% | 50.45% | +0.72 pp | 53.99% |
| `2` | 3,508 | 49.14% | 50.32% | +1.17 pp | 53.65% |
| `>= 3` | 377 | 51.99% | 49.94% | -2.05 pp | 54.64% |

### Kaisa BOTTOM `on_hit` vs heavy damage-taken count

On-hit marksman vs high-soak enemies; frontline count is the stronger cut.

**Gap MSE** 14.60 pp^2 | **Mean abs gap** 3.14 pp | **Accuracy** 54.27% | **Accuracy if calibrated** 54.96% | **Calibration lift** +0.69 pp | **Empirical effect** -3.34 pp | **HGNN effect** +2.47 pp | **Shrinkage** -0.74x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,269 | 47.72% | 48.39% | +0.68 pp | 54.09% |
| `1` | 8,375 | 47.03% | 48.89% | +1.86 pp | 54.47% |
| `2` | 4,151 | 46.40% | 49.92% | +3.52 pp | 53.72% |
| `>= 3` | 462 | 44.37% | 50.86% | +6.49 pp | 57.14% |


## Top-20 Matchup And Synergy Audits

### Yasuo MIDDLE `crit` with ally CC

Yasuo's ult chains off ally knock-ups; scales with team CC.

**Gap MSE** 14.31 pp^2 | **Mean abs gap** 3.49 pp | **Accuracy** 52.61% | **Accuracy if calibrated** 54.30% | **Calibration lift** +1.69 pp | **Empirical effect** +1.79 pp | **HGNN effect** +5.16 pp | **Shrinkage** 2.89x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.374` | 1,135 | 51.10% | 45.28% | -5.82 pp | 51.28% |
| `0.374-0.429` | 1,597 | 49.28% | 45.96% | -3.32 pp | 54.35% |
| `0.429-0.479` | 1,817 | 51.35% | 47.09% | -4.26 pp | 51.07% |
| `0.479-0.539` | 2,041 | 49.83% | 48.23% | -1.59 pp | 51.79% |
| `>= 0.539` | 2,093 | 52.89% | 50.44% | -2.45 pp | 54.13% |

### Jhin BOTTOM `crit` with ally CC

Immobile crit marksman; measured synergy with team CC is near flat.

**Gap MSE** 0.26 pp^2 | **Mean abs gap** 0.48 pp | **Accuracy** 54.56% | **Accuracy if calibrated** 54.59% | **Calibration lift** +0.03 pp | **Empirical effect** +0.04 pp | **HGNN effect** +0.08 pp | **Shrinkage** 2.20x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.374` | 889 | 48.93% | 48.52% | -0.41 pp | 52.87% |
| `0.374-0.429` | 2,204 | 48.28% | 48.66% | +0.39 pp | 56.17% |
| `0.429-0.479` | 3,424 | 48.19% | 48.59% | +0.40 pp | 53.53% |
| `0.479-0.539` | 4,940 | 47.89% | 48.71% | +0.81 pp | 55.18% |
| `>= 0.539` | 7,421 | 48.97% | 48.60% | -0.37 pp | 54.33% |

### Lulu UTILITY `utility_protection` with ally damage

Enchanter value rises with carry damage to amplify and peel for.

**Gap MSE** 0.33 pp^2 | **Mean abs gap** 0.52 pp | **Accuracy** 53.52% | **Accuracy if calibrated** 53.36% | **Calibration lift** -0.16 pp | **Empirical effect** +4.03 pp | **HGNN effect** +4.05 pp | **Shrinkage** 1.00x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 4,473 | 48.25% | 48.96% | +0.71 pp | 53.74% |
| `0.739-0.764` | 3,793 | 50.07% | 50.35% | +0.28 pp | 54.07% |
| `0.764-0.785` | 2,907 | 51.26% | 51.10% | -0.15 pp | 53.59% |
| `0.785-0.813` | 1,841 | 51.11% | 51.83% | +0.71 pp | 51.44% |
| `>= 0.813` | 329 | 52.28% | 53.01% | +0.73 pp | 55.02% |

### Ezreal BOTTOM `attack_damage` vs enemy hard CC

Skillshot poke marksman punished as enemy hard CC stacks.

**Gap MSE** 1.70 pp^2 | **Mean abs gap** 0.80 pp | **Accuracy** 55.25% | **Accuracy if calibrated** 55.08% | **Calibration lift** -0.17 pp | **Empirical effect** -0.62 pp | **HGNN effect** +1.35 pp | **Shrinkage** -2.17x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 8,028 | 47.58% | 48.15% | +0.57 pp | 55.49% |
| `1` | 12,577 | 48.25% | 48.25% | +0.00 pp | 55.08% |
| `2` | 6,129 | 48.51% | 48.59% | +0.08 pp | 54.66% |
| `>= 3` | 1,250 | 46.96% | 49.50% | +2.54 pp | 58.24% |

### Jayce TOP `attack_damage` vs enemy frontline count

Poke bruiser empirically holds up into frontline-heavy teams; model heavily shrinks the effect.

**Gap MSE** 8.84 pp^2 | **Mean abs gap** 2.72 pp | **Accuracy** 52.28% | **Accuracy if calibrated** 53.31% | **Calibration lift** +1.03 pp | **Empirical effect** +5.93 pp | **HGNN effect** +1.38 pp | **Shrinkage** 0.23x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,276 | 47.58% | 51.40% | +3.81 pp | 52.55% |
| `1` | 3,860 | 47.90% | 51.32% | +3.42 pp | 51.37% |
| `2` | 1,862 | 48.71% | 51.64% | +2.93 pp | 52.90% |
| `>= 3` | 342 | 53.51% | 52.78% | -0.73 pp | 57.31% |

### LeeSin JUNGLE `attack_damage` vs enemy scaling

Early-tempo bruiser jungler fades as enemy scaling rises.

**Gap MSE** 5.01 pp^2 | **Mean abs gap** 2.11 pp | **Accuracy** 53.70% | **Accuracy if calibrated** 53.97% | **Calibration lift** +0.28 pp | **Empirical effect** -4.30 pp | **HGNN effect** +0.71 pp | **Shrinkage** -0.16x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.829` | 2,907 | 52.98% | 51.30% | -1.67 pp | 54.63% |
| `0.829-0.841` | 2,882 | 49.31% | 50.96% | +1.65 pp | 54.55% |
| `0.841-0.852` | 3,073 | 49.66% | 50.99% | +1.33 pp | 54.57% |
| `0.852-0.863` | 2,808 | 48.72% | 51.30% | +2.58 pp | 52.96% |
| `>= 0.863` | 3,211 | 48.68% | 52.01% | +3.33 pp | 51.92% |

### Caitlyn BOTTOM `crit` vs enemy burst count

Immobile siege ADC punished by multiple burst and dive threats.

**Gap MSE** 4.28 pp^2 | **Mean abs gap** 2.05 pp | **Accuracy** 53.57% | **Accuracy if calibrated** 54.11% | **Calibration lift** +0.53 pp | **Empirical effect** -2.29 pp | **HGNN effect** -1.83 pp | **Shrinkage** 0.80x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,569 | 49.49% | 51.25% | +1.76 pp | 53.49% |
| `1` | 8,114 | 48.62% | 51.09% | +2.47 pp | 53.78% |
| `2` | 3,628 | 48.70% | 50.44% | +1.73 pp | 53.11% |
| `>= 3` | 517 | 47.20% | 49.42% | +2.22 pp | 54.16% |


## Overall Summary

Detailed audit tables above are rendered from the `val` split.

| Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 46 | 200 | 1.63 pp | 11.64 pp | 5.54 pp^2 | 54.06% | 54.20% | +0.15 pp |

| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,145,051 | 11,450,510 | 46 | 200 | 1.08 pp | 6.56 pp | 2.66 pp^2 | 54.67% | 54.75% | +0.08 pp |
| Validation | 143,131 | 1,431,310 | 46 | 200 | 1.63 pp | 11.64 pp | 5.54 pp^2 | 54.06% | 54.20% | +0.15 pp |
| Test | 143,131 | 1,431,310 | 46 | 200 | 1.99 pp | 11.97 pp | 6.77 pp^2 | 53.91% | 54.04% | +0.13 pp |

Gap MSE is `mean((HGNN_focus_WR - empirical_focus_WR)^2)` across populated threshold bins, rendered as percentage-points squared.

## Reproduction Commands

The checked-in report uses the focus-slot audit path. Checkpoints with semantic MoE slot deltas are scored with per-slot focus-side probabilities instead of one repeated match-level probability. The best audit-focused checkpoint was produced by continuing a grouped MoE checkpoint with the reference-target semantic context calibration objective.

```bash
uv run python -m app.ml.context_examples_audit \
  --context-cache-dir app/ml/data/cache \
  --model-cache-dir app/ml/data/cache \
  --model-path app/ml/data/experiments/semantic_focus_reference_w3000_cont6/model.pt \
  --encoder-sidecar-path app/ml/data/experiments/semantic_identity_sidecar_full.npz \
  --prediction-cache app/ml/data/experiments/semantic_focus_reference_w3000_cont6/audit_focus_side_probability.npy \
  --audit-split val \
  --output app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md \
  --refresh-predictions

uv run python -m app.ml.train \
  --cache-dir app/ml/data/cache \
  --encoder-sidecar-path app/ml/data/experiments/semantic_identity_sidecar_full.npz \
  --warm-start-model-path app/ml/data/experiments/semantic_focus_reference_w300_cont6/model.pt \
  --model-path app/ml/data/experiments/semantic_focus_reference_w3000_cont6/model.pt \
  --metrics-path app/ml/data/experiments/semantic_focus_reference_w3000_cont6/metrics.json \
  --use-learned-semantic-moe \
  --use-semantic-group-features \
  --semantic-context-calibration-loss-weight 3000 \
  --semantic-context-calibration-min-count 4 \
  --semantic-context-calibration-tail-weight 3 \
  --checkpoint-metric val_context_gap_mse \
  --checkpoint-min-delta -1000000 \
  --learning-rate 0.00001 \
  --patience 10 \
  --max-epochs 6 \
  --device cuda
```
