# HGNN Context Examples Audit

Updated: 2026-06-08.

Status note, 2026-06-10: keep this document as a qualitative evaluation fixture.
The specific group-context instances are critical for inspecting semantic
failures, but they should be paired with the lower-noise EB/group guardrail in
[HGNN_GROUP_CONTEXT_AUDIT.md](HGNN_GROUP_CONTEXT_AUDIT.md) and the experiment
rules in [EXPERIMENTS.md](EXPERIMENTS.md).

This audit joins the empirical focus-side context examples to the trained semantic HGNN predictions for the same cached games. Each audit is its own table: one row per threshold bin reporting `n / empirical WR / HGNN WR / gap / accuracy`, with a per-table Gap MSE, accuracy, and the accuracy headroom from perfect calibration (`Calibration lift`) above it. Gap is `HGNN WR - empirical WR`; zero gap is the target.

## Scope And Threshold Definitions

- Context source: `app/ml/data/cache` side-row arrays, `val` split only.
- HGNN model: `app/ml/data/experiments/semantic_context_compact_run/model.pt`.
- HGNN cache: `app/ml/data/experiments/semantic_context_compact_cache`.
- Encoder sidecar artifact: cache metadata or materialized cache arrays.
- HGNN WR uses focus-slot semantic MoE probabilities when a checkpoint exposes slot deltas; older checkpoints fall back to raw `final_logit` probabilities.
- Semantic group feature schema: v2, 25 compact per-slot features; used only by checkpoints trained with `--use-semantic-group-features`.
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
| Headline Trajectory Audit Tables | 10 | 47 | 1.69 pp | 6.22 pp | 4.64 pp^2 | 58.36% | 58.46% | +0.10 pp |
| Richer Composition Trajectory Tables | 13 | 52 | 1.48 pp | 10.46 pp | 5.49 pp^2 | 57.55% | 57.66% | +0.12 pp |
| Retained Prior And User-Requested Trajectory Tables | 12 | 53 | 1.62 pp | 5.86 pp | 4.28 pp^2 | 58.12% | 58.16% | +0.04 pp |
| Inspected Lower-Signal Trajectory Tables | 4 | 16 | 0.78 pp | 2.60 pp | 1.22 pp^2 | 57.83% | 57.84% | +0.01 pp |
| Top-20 Matchup And Synergy Audits | 7 | 32 | 0.98 pp | 3.12 pp | 1.60 pp^2 | 57.60% | 57.69% | +0.09 pp |

## Train, Validation, And Test Summary

These rows reuse the same audit specs and prediction cache, but evaluate the cached train, validation, and test ranges separately.

| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,145,051 | 11,450,510 | 46 | 200 | 0.75 pp | 5.31 pp | 1.33 pp^2 | 58.59% | 58.60% | +0.01 pp |
| Validation | 143,131 | 1,431,310 | 46 | 200 | 1.43 pp | 10.46 pp | 4.01 pp^2 | 57.85% | 57.88% | +0.04 pp |
| Test | 143,131 | 1,431,310 | 46 | 200 | 1.45 pp | 8.20 pp | 4.02 pp^2 | 57.26% | 57.30% | +0.04 pp |

## Enemy Count Tail Shrinkage

| Audit | Axis | Baseline bin | Tail bin | Empirical tail effect | HGNN tail effect | Shrinkage |
|---|---|---:|---:|---:|---:|---:|
| Sylas MIDDLE `ability_power` vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -7.26 pp | -2.38 pp | 0.33x |
| Nilah BOTTOM any build vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -6.51 pp | -5.17 pp | 0.79x |
| Kaisa BOTTOM any build vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | +0.74 pp | -1.66 pp | -2.25x |
| Kaisa BOTTOM `on_hit` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +6.59 pp | +5.54 pp | 0.84x |
| Ahri MIDDLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +7.64 pp | +5.44 pp | 0.71x |
| Sylas JUNGLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +9.37 pp | +5.56 pp | 0.59x |
| Sylas MIDDLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +11.72 pp | +3.37 pp | 0.29x |
| Karma UTILITY any build vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +1.56 pp | +7.11 pp | 4.56x |
| Vayne BOTTOM `on_hit` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +15.19 pp | +9.71 pp | 0.64x |
| Thresh UTILITY `ar_tank` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.39 pp | -4.22 pp | 0.96x |
| Nautilus UTILITY `mr_tank` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | +3.26 pp | +0.46 pp | 0.14x |
| Zed MIDDLE `lethality` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -17.04 pp | -6.05 pp | 0.35x |
| Nami UTILITY `utility_protection` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -1.85 pp | -3.58 pp | 1.93x |
| Jinx BOTTOM `crit` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.86 pp | -4.44 pp | 0.91x |
| Malphite TOP `ar_tank` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -12.88 pp | -9.05 pp | 0.70x |
| Viego JUNGLE any build vs enemy high-HP count | `enemy_high_hp_count` | `0` | `>= 3` | +1.45 pp | +2.65 pp | 1.83x |
| Darius TOP any build vs enemy range count | `enemy_ranged_count` | `<= 1` | `>= 4` | -4.82 pp | -3.93 pp | 0.82x |
| MasterYi JUNGLE any build vs enemy hard CC | `enemy_hard_cc_count` | `0` | `>= 3` | -0.26 pp | -2.09 pp | 8.04x |
| Focus HP `<= 2309` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.89 pp | -3.86 pp | 0.79x |
| Focus HP `>= 2478` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -3.84 pp | -3.90 pp | 1.01x |
| Ahri MIDDLE `ability_power` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | +1.05 pp | -2.24 pp | -2.13x |
| Kaisa BOTTOM `on_hit` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -3.34 pp | -1.29 pp | 0.39x |
| Ezreal BOTTOM `attack_damage` vs enemy hard CC | `enemy_hard_cc_count` | `0` | `>= 3` | -0.62 pp | +0.25 pp | -0.40x |
| Jayce TOP `attack_damage` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +5.93 pp | +5.89 pp | 0.99x |
| Caitlyn BOTTOM `crit` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -2.29 pp | -1.93 pp | 0.84x |

## Headline Trajectory Audit Tables

### Yasuo TOP `crit` vs enemy siege

Melee crit carry punished by poke and siege.

**Gap MSE** 4.90 pp^2 | **Mean abs gap** 1.84 pp | **Accuracy** 56.54% | **Accuracy if calibrated** 56.37% | **Calibration lift** -0.16 pp | **Empirical effect** -0.46 pp | **HGNN effect** -4.04 pp | **Shrinkage** 8.82x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.441` | 782 | 51.53% | 54.45% | +2.91 pp | 57.16% |
| `0.441-0.471` | 797 | 49.44% | 52.78% | +3.34 pp | 57.47% |
| `0.471-0.499` | 759 | 51.38% | 51.57% | +0.19 pp | 54.94% |
| `0.499-0.530` | 714 | 49.30% | 51.39% | +2.09 pp | 57.00% |
| `>= 0.530` | 650 | 51.08% | 50.41% | -0.67 pp | 56.00% |

### Graves JUNGLE `lethality` vs enemy damage

Burst jungler into high enemy damage.

**Gap MSE** 18.49 pp^2 | **Mean abs gap** 3.93 pp | **Accuracy** 68.94% | **Accuracy if calibrated** 69.16% | **Calibration lift** +0.21 pp | **Empirical effect** -11.09 pp | **HGNN effect** -10.20 pp | **Shrinkage** 0.92x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 483 | 41.20% | 34.98% | -6.22 pp | 64.60% |
| `0.739-0.764` | 408 | 33.09% | 29.31% | -3.78 pp | 68.14% |
| `0.764-0.785` | 340 | 30.29% | 27.24% | -3.05 pp | 70.00% |
| `0.785-0.813` | 351 | 27.92% | 26.64% | -1.28 pp | 72.65% |
| `>= 0.813` | 279 | 30.11% | 24.77% | -5.33 pp | 71.68% |

### Yasuo MIDDLE `crit` vs enemy siege

Same melee-carry-into-poke pattern across lane.

**Gap MSE** 0.78 pp^2 | **Mean abs gap** 0.80 pp | **Accuracy** 58.39% | **Accuracy if calibrated** 58.26% | **Calibration lift** -0.13 pp | **Empirical effect** -1.65 pp | **HGNN effect** -2.04 pp | **Shrinkage** 1.24x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.441` | 1,844 | 51.03% | 52.45% | +1.42 pp | 59.33% |
| `0.441-0.471` | 1,720 | 52.21% | 51.65% | -0.56 pp | 59.30% |
| `0.471-0.499` | 1,758 | 50.40% | 51.02% | +0.63 pp | 57.11% |
| `0.499-0.530` | 1,674 | 51.73% | 51.37% | -0.36 pp | 58.12% |
| `>= 0.530` | 1,687 | 49.38% | 50.40% | +1.03 pp | 58.03% |

### Ahri MIDDLE `ability_power` vs enemy scaling

AP mid into scaling enemy compositions.

**Gap MSE** 2.17 pp^2 | **Mean abs gap** 1.20 pp | **Accuracy** 57.38% | **Accuracy if calibrated** 57.33% | **Calibration lift** -0.05 pp | **Empirical effect** -4.27 pp | **HGNN effect** -1.93 pp | **Shrinkage** 0.45x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.829` | 3,162 | 52.88% | 52.82% | -0.06 pp | 57.91% |
| `0.829-0.841` | 3,036 | 50.49% | 50.85% | +0.35 pp | 59.22% |
| `0.841-0.852` | 3,198 | 48.69% | 50.57% | +1.88 pp | 56.35% |
| `0.852-0.863` | 2,784 | 48.89% | 50.29% | +1.40 pp | 56.57% |
| `>= 0.863` | 2,876 | 48.61% | 50.89% | +2.28 pp | 56.78% |

### Nautilus UTILITY `mr_tank` with ally damage

Engage support with damage behind it.

**Gap MSE** 2.17 pp^2 | **Mean abs gap** 1.26 pp | **Accuracy** 58.06% | **Accuracy if calibrated** 58.13% | **Calibration lift** +0.07 pp | **Empirical effect** +7.33 pp | **HGNN effect** +8.80 pp | **Shrinkage** 1.20x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 684 | 44.74% | 44.78% | +0.05 pp | 59.80% |
| `0.739-0.764` | 679 | 48.90% | 47.65% | -1.25 pp | 58.47% |
| `0.764-0.785` | 722 | 50.69% | 49.58% | -1.11 pp | 55.54% |
| `0.785-0.813` | 682 | 53.37% | 50.97% | -2.40 pp | 58.50% |
| `>= 0.813` | 290 | 52.07% | 53.58% | +1.51 pp | 58.28% |

### Galio MIDDLE `mr_tank` vs enemy magic

Anti-magic tank itemization (kept off-list MR-tank).

**Gap MSE** 2.49 pp^2 | **Mean abs gap** 1.36 pp | **Accuracy** 62.01% | **Accuracy if calibrated** 62.01% | **Calibration lift** +0.00 pp | **Empirical effect** +7.58 pp | **HGNN effect** +5.51 pp | **Shrinkage** 0.73x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 158 | 38.61% | 40.27% | +1.66 pp | 63.29% |
| `0.373-0.423` | 213 | 43.19% | 40.49% | -2.71 pp | 65.26% |
| `0.423-0.486` | 235 | 42.98% | 42.28% | -0.70 pp | 62.55% |
| `0.486-0.549` | 322 | 43.48% | 42.17% | -1.30 pp | 59.32% |
| `>= 0.549` | 433 | 46.19% | 45.78% | -0.41 pp | 61.66% |

### Malphite TOP `ar_tank` vs enemy physical

Armor tank into AD-heavy enemies.

**Gap MSE** 0.89 pp^2 | **Mean abs gap** 0.89 pp | **Accuracy** 57.72% | **Accuracy if calibrated** 57.95% | **Calibration lift** +0.23 pp | **Empirical effect** +10.33 pp | **HGNN effect** +8.45 pp | **Shrinkage** 0.82x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 669 | 45.44% | 46.15% | +0.71 pp | 60.69% |
| `0.387-0.448` | 806 | 45.91% | 46.49% | +0.59 pp | 55.71% |
| `0.448-0.508` | 1,238 | 49.19% | 49.87% | +0.68 pp | 55.33% |
| `0.508-0.557` | 1,544 | 50.71% | 52.04% | +1.33 pp | 57.38% |
| `>= 0.557` | 1,741 | 55.77% | 54.60% | -1.17 pp | 59.51% |

### Sylas MIDDLE `ability_power` vs enemy range

Short-range AP battlemage into enemy range pressure.

**Gap MSE** 4.27 pp^2 | **Mean abs gap** 1.88 pp | **Accuracy** 56.77% | **Accuracy if calibrated** 56.93% | **Calibration lift** +0.15 pp | **Empirical effect** -7.26 pp | **HGNN effect** -2.38 pp | **Shrinkage** 0.33x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 529 | 56.71% | 54.99% | -1.72 pp | 58.22% |
| `2` | 2,307 | 54.92% | 53.05% | -1.87 pp | 57.52% |
| `3` | 2,805 | 51.98% | 52.74% | +0.76 pp | 56.19% |
| `>= 4` | 813 | 49.45% | 52.61% | +3.17 pp | 55.72% |

### Nilah BOTTOM any build vs enemy range

Melee bot lane into range-heavy teams (kept off-list melee-ADC).

**Gap MSE** 5.86 pp^2 | **Mean abs gap** 2.31 pp | **Accuracy** 57.54% | **Accuracy if calibrated** 58.75% | **Calibration lift** +1.21 pp | **Empirical effect** -6.51 pp | **HGNN effect** -5.17 pp | **Shrinkage** 0.79x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 168 | 53.57% | 54.89% | +1.32 pp | 55.95% |
| `2` | 644 | 56.21% | 52.99% | -3.23 pp | 58.23% |
| `3` | 807 | 52.66% | 50.61% | -2.06 pp | 58.49% |
| `>= 4` | 204 | 47.06% | 49.72% | +2.66 pp | 52.94% |

### Kaisa BOTTOM any build vs enemy range

High-sample marksman vs enemy range pressure; large n keeps bins low-noise.

**Gap MSE** 4.56 pp^2 | **Mean abs gap** 1.61 pp | **Accuracy** 58.96% | **Accuracy if calibrated** 59.15% | **Calibration lift** +0.19 pp | **Empirical effect** +0.74 pp | **HGNN effect** -1.66 pp | **Shrinkage** -2.25x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 1,793 | 45.73% | 49.59% | +3.86 pp | 60.35% |
| `2` | 7,744 | 47.87% | 48.97% | +1.10 pp | 59.67% |
| `3` | 9,357 | 48.54% | 48.57% | +0.02 pp | 57.83% |
| `>= 4` | 2,707 | 46.47% | 47.93% | +1.46 pp | 59.88% |


## Richer Composition Trajectory Tables

### Kaisa BOTTOM `on_hit` vs enemy frontline count

On-hit marksman shreds added enemy frontline.

**Gap MSE** 0.38 pp^2 | **Mean abs gap** 0.57 pp | **Accuracy** 59.30% | **Accuracy if calibrated** 59.37% | **Calibration lift** +0.07 pp | **Empirical effect** +6.59 pp | **HGNN effect** +5.54 pp | **Shrinkage** 0.84x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,710 | 45.80% | 46.56% | +0.77 pp | 59.34% |
| `1` | 7,899 | 46.42% | 47.25% | +0.82 pp | 58.93% |
| `2` | 3,915 | 48.51% | 48.92% | +0.41 pp | 59.64% |
| `>= 3` | 733 | 52.39% | 52.10% | -0.29 pp | 61.26% |

### Ahri MIDDLE `ability_power` vs enemy frontline count

AP mid improves as enemies stack durable targets.

**Gap MSE** 1.54 pp^2 | **Mean abs gap** 1.22 pp | **Accuracy** 57.38% | **Accuracy if calibrated** 57.24% | **Calibration lift** -0.14 pp | **Empirical effect** +7.64 pp | **HGNN effect** +5.44 pp | **Shrinkage** 0.71x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,287 | 48.89% | 50.08% | +1.19 pp | 56.54% |
| `1` | 6,903 | 49.76% | 50.88% | +1.12 pp | 57.50% |
| `2` | 3,277 | 50.56% | 52.14% | +1.57 pp | 57.95% |
| `>= 3` | 589 | 56.54% | 55.52% | -1.01 pp | 58.91% |

### Sylas JUNGLE `ability_power` vs enemy frontline count

Sustained AP skirmisher into beefy teams.

**Gap MSE** 2.55 pp^2 | **Mean abs gap** 1.55 pp | **Accuracy** 57.11% | **Accuracy if calibrated** 57.60% | **Calibration lift** +0.49 pp | **Empirical effect** +9.37 pp | **HGNN effect** +5.56 pp | **Shrinkage** 0.59x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,480 | 49.86% | 51.57% | +1.70 pp | 55.88% |
| `1` | 2,359 | 51.17% | 52.19% | +1.03 pp | 57.69% |
| `2` | 1,123 | 52.89% | 54.24% | +1.34 pp | 56.63% |
| `>= 3` | 184 | 59.24% | 57.13% | -2.11 pp | 62.50% |

### Sylas MIDDLE `ability_power` vs enemy frontline count

Same AP anti-frontline pattern from lane.

**Gap MSE** 11.25 pp^2 | **Mean abs gap** 2.32 pp | **Accuracy** 56.77% | **Accuracy if calibrated** 57.02% | **Calibration lift** +0.25 pp | **Empirical effect** +11.72 pp | **HGNN effect** +3.37 pp | **Shrinkage** 0.29x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,781 | 50.42% | 52.41% | +1.99 pp | 54.52% |
| `1` | 2,941 | 53.55% | 52.79% | -0.76 pp | 57.67% |
| `2` | 1,489 | 53.93% | 53.74% | -0.19 pp | 56.62% |
| `>= 3` | 243 | 62.14% | 55.78% | -6.36 pp | 63.37% |

### Karma UTILITY any build vs enemy frontline count

Utility support gains value as enemies stack frontline to zone.

**Gap MSE** 5.61 pp^2 | **Mean abs gap** 1.85 pp | **Accuracy** 57.25% | **Accuracy if calibrated** 57.32% | **Calibration lift** +0.06 pp | **Empirical effect** +1.56 pp | **HGNN effect** +7.11 pp | **Shrinkage** 4.56x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 5,678 | 49.10% | 47.94% | -1.16 pp | 56.38% |
| `1` | 8,586 | 48.26% | 49.45% | +1.19 pp | 57.83% |
| `2` | 3,819 | 50.62% | 51.26% | +0.65 pp | 57.32% |
| `>= 3` | 679 | 50.66% | 55.05% | +4.39 pp | 57.00% |

### Vayne BOTTOM `on_hit` vs enemy frontline count

Classic anti-tank marksman pattern.

**Gap MSE** 7.14 pp^2 | **Mean abs gap** 2.08 pp | **Accuracy** 57.99% | **Accuracy if calibrated** 58.41% | **Calibration lift** +0.42 pp | **Empirical effect** +15.19 pp | **HGNN effect** +9.71 pp | **Shrinkage** 0.64x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 896 | 46.43% | 47.45% | +1.02 pp | 55.13% |
| `1` | 1,770 | 49.60% | 49.55% | -0.06 pp | 58.53% |
| `2` | 918 | 50.33% | 53.09% | +2.76 pp | 58.93% |
| `>= 3` | 198 | 61.62% | 57.16% | -4.46 pp | 61.62% |

### Thresh UTILITY `ar_tank` vs enemy burst count

Durable engage support punished by multiple burst threats.

**Gap MSE** 0.65 pp^2 | **Mean abs gap** 0.50 pp | **Accuracy** 57.05% | **Accuracy if calibrated** 57.38% | **Calibration lift** +0.33 pp | **Empirical effect** -4.39 pp | **HGNN effect** -4.22 pp | **Shrinkage** 0.96x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 995 | 49.35% | 49.35% | -0.00 pp | 56.48% |
| `1` | 1,769 | 48.95% | 48.69% | -0.27 pp | 57.09% |
| `2` | 778 | 45.76% | 47.33% | +1.58 pp | 57.07% |
| `>= 3` | 109 | 44.95% | 45.13% | +0.17 pp | 61.47% |

### Nautilus UTILITY `mr_tank` vs enemy burst count

High-HP engage tank loses into concentrated burst.

**Gap MSE** 6.15 pp^2 | **Mean abs gap** 1.84 pp | **Accuracy** 58.06% | **Accuracy if calibrated** 58.06% | **Calibration lift** +0.00 pp | **Empirical effect** +3.26 pp | **HGNN effect** +0.46 pp | **Shrinkage** 0.14x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 852 | 50.59% | 48.82% | -1.76 pp | 59.74% |
| `1` | 1,472 | 48.91% | 48.63% | -0.29 pp | 58.56% |
| `2` | 642 | 49.69% | 48.94% | -0.75 pp | 55.45% |
| `>= 3` | 91 | 53.85% | 49.28% | -4.57 pp | 52.75% |

### Zed MIDDLE `lethality` vs enemy burst count

Assassin into enemy burst stacking.

**Gap MSE** 27.97 pp^2 | **Mean abs gap** 3.14 pp | **Accuracy** 57.29% | **Accuracy if calibrated** 57.48% | **Calibration lift** +0.19 pp | **Empirical effect** -17.04 pp | **HGNN effect** -6.05 pp | **Shrinkage** 0.35x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,816 | 53.69% | 53.15% | -0.54 pp | 57.82% |
| `1` | 3,080 | 51.46% | 51.53% | +0.07 pp | 56.53% |
| `2` | 1,131 | 51.72% | 50.24% | -1.48 pp | 58.36% |
| `>= 3` | 161 | 36.65% | 47.10% | +10.46 pp | 58.39% |

### Nami UTILITY `utility_protection` vs enemy burst count

Protective enchanter punished by burst-heavy enemies.

**Gap MSE** 0.83 pp^2 | **Mean abs gap** 0.83 pp | **Accuracy** 57.37% | **Accuracy if calibrated** 57.47% | **Calibration lift** +0.10 pp | **Empirical effect** -1.85 pp | **HGNN effect** -3.58 pp | **Shrinkage** 1.93x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,342 | 51.27% | 52.28% | +1.01 pp | 56.33% |
| `1` | 7,172 | 49.85% | 51.16% | +1.32 pp | 57.47% |
| `2` | 3,063 | 49.76% | 50.02% | +0.26 pp | 58.99% |
| `>= 3` | 429 | 49.42% | 48.70% | -0.72 pp | 54.55% |

### Jinx BOTTOM `crit` vs enemy burst count

Fragile crit carry into burst-heavy enemies.

**Gap MSE** 1.19 pp^2 | **Mean abs gap** 0.99 pp | **Accuracy** 57.31% | **Accuracy if calibrated** 57.20% | **Calibration lift** -0.12 pp | **Empirical effect** -4.86 pp | **HGNN effect** -4.44 pp | **Shrinkage** 0.91x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,968 | 53.45% | 54.66% | +1.21 pp | 57.84% |
| `1` | 7,062 | 52.36% | 53.04% | +0.67 pp | 57.56% |
| `2` | 3,015 | 51.18% | 51.63% | +0.45 pp | 56.48% |
| `>= 3` | 426 | 48.59% | 50.22% | +1.63 pp | 54.23% |

### Malphite TOP `ar_tank` vs heavy damage-taken count

Armor tank loses into teams with multiple high-soak targets.

**Gap MSE** 5.25 pp^2 | **Mean abs gap** 1.50 pp | **Accuracy** 57.72% | **Accuracy if calibrated** 58.00% | **Calibration lift** +0.28 pp | **Empirical effect** -12.88 pp | **HGNN effect** -9.05 pp | **Shrinkage** 0.70x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,616 | 53.71% | 54.36% | +0.65 pp | 58.23% |
| `1` | 2,903 | 50.40% | 50.66% | +0.27 pp | 57.25% |
| `2` | 1,310 | 48.63% | 48.02% | -0.61 pp | 57.25% |
| `>= 3` | 169 | 40.83% | 45.32% | +4.49 pp | 64.50% |

### Viego JUNGLE any build vs enemy high-HP count

On-hit bruiser jungler into high-HP enemy teams.

**Gap MSE** 0.89 pp^2 | **Mean abs gap** 0.81 pp | **Accuracy** 56.82% | **Accuracy if calibrated** 57.19% | **Calibration lift** +0.37 pp | **Empirical effect** +1.45 pp | **HGNN effect** +2.65 pp | **Shrinkage** 1.83x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,051 | 48.48% | 47.28% | -1.20 pp | 56.51% |
| `1` | 5,723 | 48.87% | 47.71% | -1.16 pp | 56.91% |
| `2` | 3,742 | 47.35% | 48.24% | +0.88 pp | 56.41% |
| `>= 3` | 1,290 | 49.92% | 49.93% | +0.01 pp | 58.29% |


## Retained Prior And User-Requested Trajectory Tables

### Malphite all roles `ar_tank` vs enemy physical

Original armor-stack audit, retained beyond TOP-only.

**Gap MSE** 1.21 pp^2 | **Mean abs gap** 1.03 pp | **Accuracy** 57.63% | **Accuracy if calibrated** 57.58% | **Calibration lift** -0.04 pp | **Empirical effect** +10.24 pp | **HGNN effect** +8.49 pp | **Shrinkage** 0.83x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 728 | 45.47% | 46.31% | +0.84 pp | 60.44% |
| `0.387-0.448` | 920 | 47.17% | 46.61% | -0.57 pp | 55.65% |
| `0.448-0.508` | 1,376 | 48.69% | 49.80% | +1.11 pp | 55.67% |
| `0.508-0.557` | 1,720 | 50.23% | 51.95% | +1.72 pp | 56.92% |
| `>= 0.557` | 2,041 | 55.71% | 54.80% | -0.91 pp | 59.43% |

### Galio all roles `mr_tank` vs enemy magic

Original anti-magic tank family, broader than MIDDLE-only.

**Gap MSE** 4.47 pp^2 | **Mean abs gap** 1.85 pp | **Accuracy** 61.00% | **Accuracy if calibrated** 61.18% | **Calibration lift** +0.18 pp | **Empirical effect** +6.08 pp | **HGNN effect** +6.60 pp | **Shrinkage** 1.08x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 173 | 39.88% | 41.33% | +1.45 pp | 63.01% |
| `0.373-0.423` | 228 | 44.74% | 40.97% | -3.77 pp | 63.60% |
| `0.423-0.486` | 270 | 44.81% | 43.85% | -0.97 pp | 62.59% |
| `0.486-0.549` | 387 | 44.96% | 43.85% | -1.11 pp | 59.43% |
| `>= 0.549` | 570 | 45.96% | 47.93% | +1.96 pp | 59.65% |

### Nautilus all roles `mr_tank` vs enemy magic

Top-20 MR-tank anti-magic case alongside Galio.

**Gap MSE** 3.37 pp^2 | **Mean abs gap** 1.28 pp | **Accuracy** 58.12% | **Accuracy if calibrated** 58.64% | **Calibration lift** +0.52 pp | **Empirical effect** -1.91 pp | **HGNN effect** -0.98 pp | **Shrinkage** 0.51x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 218 | 51.38% | 50.55% | -0.83 pp | 59.17% |
| `0.373-0.423` | 399 | 52.13% | 48.57% | -3.56 pp | 56.14% |
| `0.423-0.486` | 507 | 48.52% | 48.48% | -0.04 pp | 59.17% |
| `0.486-0.549` | 757 | 49.01% | 47.15% | -1.86 pp | 56.94% |
| `>= 0.549` | 1,221 | 49.47% | 49.57% | +0.10 pp | 58.89% |

### Nautilus all roles `ar_tank` vs enemy physical

Physical-heavy enemy teams remain a support-tank check.

**Gap MSE** 0.47 pp^2 | **Mean abs gap** 0.54 pp | **Accuracy** 57.91% | **Accuracy if calibrated** 57.97% | **Calibration lift** +0.06 pp | **Empirical effect** +5.85 pp | **HGNN effect** +6.20 pp | **Shrinkage** 1.06x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 915 | 46.23% | 46.06% | -0.17 pp | 60.33% |
| `0.387-0.448` | 1,287 | 45.53% | 46.72% | +1.19 pp | 56.10% |
| `0.448-0.508` | 1,749 | 50.20% | 49.28% | -0.92 pp | 58.20% |
| `0.508-0.557` | 1,908 | 49.21% | 49.46% | +0.25 pp | 56.97% |
| `>= 0.557` | 2,210 | 52.08% | 52.26% | +0.18 pp | 58.55% |

### Darius TOP any build vs enemy range count

Static team range pressure, stronger than lane-only range.

**Gap MSE** 3.40 pp^2 | **Mean abs gap** 1.75 pp | **Accuracy** 58.23% | **Accuracy if calibrated** 58.08% | **Calibration lift** -0.15 pp | **Empirical effect** -4.82 pp | **HGNN effect** -3.93 pp | **Shrinkage** 0.82x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 662 | 50.91% | 52.70% | +1.79 pp | 58.91% |
| `2` | 2,985 | 48.84% | 50.20% | +1.36 pp | 58.96% |
| `3` | 3,478 | 50.60% | 49.44% | -1.16 pp | 57.56% |
| `>= 4` | 972 | 46.09% | 48.77% | +2.68 pp | 57.92% |

### Darius TOP any build vs same-role range

User-requested static melee/ranged lane audit.

**Gap MSE** 3.46 pp^2 | **Mean abs gap** 1.33 pp | **Accuracy** 58.23% | **Accuracy if calibrated** 58.24% | **Calibration lift** +0.01 pp | **Empirical effect** -3.15 pp | **HGNN effect** -0.56 pp | **Shrinkage** 0.18x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 250` | 6,747 | 49.96% | 50.00% | +0.04 pp | 58.40% |
| `> 250` | 1,350 | 46.81% | 49.44% | +2.63 pp | 57.41% |

### MasterYi JUNGLE any build vs enemy hard CC

User-requested low-CC audit; unique even though gap is modest.

**Gap MSE** 1.53 pp^2 | **Mean abs gap** 1.00 pp | **Accuracy** 57.04% | **Accuracy if calibrated** 57.03% | **Calibration lift** -0.01 pp | **Empirical effect** -0.26 pp | **HGNN effect** -2.09 pp | **Shrinkage** 8.04x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,050 | 53.17% | 52.76% | -0.41 pp | 57.71% |
| `1` | 3,261 | 51.64% | 51.13% | -0.51 pp | 56.55% |
| `2` | 1,636 | 51.22% | 50.37% | -0.85 pp | 56.60% |
| `>= 3` | 395 | 52.91% | 50.68% | -2.23 pp | 59.49% |

### Selected enchanters UTILITY with skirmish allies

Original enchanter-with-skirmishers synergy probe.

**Gap MSE** 1.63 pp^2 | **Mean abs gap** 1.11 pp | **Accuracy** 57.36% | **Accuracy if calibrated** 57.46% | **Calibration lift** +0.09 pp | **Empirical effect** +3.14 pp | **HGNN effect** +0.70 pp | **Shrinkage** 0.22x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 36,795 | 50.16% | 50.61% | +0.46 pp | 57.56% |
| `1` | 7,128 | 52.30% | 51.42% | -0.88 pp | 56.44% |
| `>= 2` | 319 | 53.29% | 51.32% | -1.97 pp | 55.80% |

### Low own-damage teams vs enemy heal/shield

Original low-damage into sustain audit.

**Gap MSE** 0.63 pp^2 | **Mean abs gap** 0.71 pp | **Accuracy** 58.93% | **Accuracy if calibrated** 58.94% | **Calibration lift** +0.00 pp | **Empirical effect** -1.63 pp | **HGNN effect** -2.79 pp | **Shrinkage** 1.71x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.028` | 11,038 | 49.66% | 50.54% | +0.88 pp | 59.08% |
| `0.028-0.077` | 10,736 | 48.64% | 49.14% | +0.50 pp | 59.13% |
| `0.077-0.200` | 10,473 | 46.57% | 47.88% | +1.31 pp | 59.16% |
| `0.200-0.202` | 11,951 | 47.11% | 47.67% | +0.56 pp | 59.03% |
| `>= 0.202` | 11,263 | 48.02% | 47.75% | -0.27 pp | 58.29% |

### Ambessa TOP `attack_damage` vs enemy damage

Durable bruiser into enemy damage pressure.

**Gap MSE** 7.48 pp^2 | **Mean abs gap** 2.67 pp | **Accuracy** 57.04% | **Accuracy if calibrated** 56.95% | **Calibration lift** -0.08 pp | **Empirical effect** -2.91 pp | **HGNN effect** -3.60 pp | **Shrinkage** 1.24x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 1,164 | 50.69% | 53.17% | +2.49 pp | 58.42% |
| `0.739-0.764` | 1,204 | 48.67% | 51.11% | +2.44 pp | 56.73% |
| `0.764-0.785` | 1,140 | 46.40% | 49.98% | +3.57 pp | 57.19% |
| `0.785-0.813` | 1,195 | 46.03% | 49.07% | +3.05 pp | 56.40% |
| `>= 0.813` | 1,260 | 47.78% | 49.57% | +1.79 pp | 56.51% |

### LeeSin JUNGLE `ad_off_tank` vs enemy magic

Bruiser jungler resisting magic-heavy enemies.

**Gap MSE** 11.38 pp^2 | **Mean abs gap** 2.71 pp | **Accuracy** 58.44% | **Accuracy if calibrated** 58.97% | **Calibration lift** +0.54 pp | **Empirical effect** +3.43 pp | **HGNN effect** -3.02 pp | **Shrinkage** -0.88x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 647 | 42.35% | 48.21% | +5.86 pp | 57.81% |
| `0.373-0.423` | 573 | 45.20% | 46.34% | +1.14 pp | 58.12% |
| `0.423-0.486` | 541 | 45.10% | 46.80% | +1.70 pp | 58.04% |
| `0.486-0.549` | 485 | 40.21% | 44.45% | +4.25 pp | 60.00% |
| `>= 0.549` | 367 | 45.78% | 45.19% | -0.58 pp | 58.58% |

### Thresh UTILITY `mr_tank` vs enemy magic

MR-tank support anti-magic case.

**Gap MSE** 10.04 pp^2 | **Mean abs gap** 2.95 pp | **Accuracy** 57.73% | **Accuracy if calibrated** 57.87% | **Calibration lift** +0.14 pp | **Empirical effect** +6.44 pp | **HGNN effect** +6.71 pp | **Shrinkage** 1.04x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 56 | 44.64% | 42.94% | -1.70 pp | 50.00% |
| `0.373-0.423` | 76 | 40.79% | 44.94% | +4.15 pp | 68.42% |
| `0.423-0.486` | 110 | 50.00% | 46.46% | -3.54 pp | 54.55% |
| `0.486-0.549` | 140 | 50.71% | 46.78% | -3.94 pp | 50.00% |
| `>= 0.549` | 323 | 51.08% | 49.65% | -1.43 pp | 60.99% |


## Inspected Lower-Signal Trajectory Tables

### Focus HP `<= 2309` vs enemy burst count

Broad HP-vs-burst check; useful but lower signal than champion-specific rows.

**Gap MSE** 0.18 pp^2 | **Mean abs gap** 0.36 pp | **Accuracy** 57.73% | **Accuracy if calibrated** 57.73% | **Calibration lift** +0.00 pp | **Empirical effect** -4.89 pp | **HGNN effect** -3.86 pp | **Shrinkage** 0.79x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 92,697 | 51.98% | 51.60% | -0.38 pp | 58.03% |
| `1` | 157,900 | 50.54% | 50.55% | +0.02 pp | 57.89% |
| `2` | 67,915 | 49.80% | 49.42% | -0.37 pp | 57.06% |
| `>= 3` | 9,477 | 47.08% | 47.73% | +0.65 pp | 56.94% |

### Focus HP `>= 2478` vs enemy burst count

High-HP slots also drop into burst stacks, so champion/build specificity matters.

**Gap MSE** 0.03 pp^2 | **Mean abs gap** 0.13 pp | **Accuracy** 57.87% | **Accuracy if calibrated** 57.88% | **Calibration lift** +0.01 pp | **Empirical effect** -3.84 pp | **HGNN effect** -3.90 pp | **Shrinkage** 1.01x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 103,541 | 50.77% | 50.83% | +0.06 pp | 58.10% |
| `1` | 174,445 | 49.39% | 49.54% | +0.15 pp | 57.98% |
| `2` | 74,569 | 48.20% | 48.49% | +0.29 pp | 57.22% |
| `>= 3` | 10,375 | 46.93% | 46.94% | +0.01 pp | 58.43% |

### Ahri MIDDLE `ability_power` vs heavy damage-taken count

AP mid vs multiple high-soak enemies; weaker axis than frontline count.

**Gap MSE** 2.97 pp^2 | **Mean abs gap** 1.58 pp | **Accuracy** 57.38% | **Accuracy if calibrated** 57.37% | **Calibration lift** -0.01 pp | **Empirical effect** +1.05 pp | **HGNN effect** -2.24 pp | **Shrinkage** -2.13x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,740 | 50.94% | 51.63% | +0.69 pp | 56.87% |
| `1` | 7,431 | 49.74% | 51.05% | +1.31 pp | 57.62% |
| `2` | 3,508 | 49.14% | 50.85% | +1.71 pp | 57.24% |
| `>= 3` | 377 | 51.99% | 49.39% | -2.60 pp | 58.89% |

### Kaisa BOTTOM `on_hit` vs heavy damage-taken count

On-hit marksman vs high-soak enemies; frontline count is the stronger cut.

**Gap MSE** 1.71 pp^2 | **Mean abs gap** 1.04 pp | **Accuracy** 59.30% | **Accuracy if calibrated** 59.40% | **Calibration lift** +0.10 pp | **Empirical effect** -3.34 pp | **HGNN effect** -1.29 pp | **Shrinkage** 0.39x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,269 | 47.72% | 48.05% | +0.33 pp | 58.68% |
| `1` | 8,375 | 47.03% | 47.75% | +0.71 pp | 59.39% |
| `2` | 4,151 | 46.40% | 47.12% | +0.72 pp | 59.87% |
| `>= 3` | 462 | 44.37% | 46.76% | +2.39 pp | 58.44% |


## Top-20 Matchup And Synergy Audits

### Yasuo MIDDLE `crit` with ally CC

Yasuo's ult chains off ally knock-ups; scales with team CC.

**Gap MSE** 2.38 pp^2 | **Mean abs gap** 1.30 pp | **Accuracy** 58.39% | **Accuracy if calibrated** 58.29% | **Calibration lift** -0.10 pp | **Empirical effect** +1.79 pp | **HGNN effect** +5.02 pp | **Shrinkage** 2.80x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.374` | 1,135 | 51.10% | 48.80% | -2.30 pp | 58.06% |
| `0.374-0.429` | 1,597 | 49.28% | 49.75% | +0.47 pp | 55.98% |
| `0.429-0.479` | 1,817 | 51.35% | 50.85% | -0.50 pp | 58.45% |
| `0.479-0.539` | 2,041 | 49.83% | 52.13% | +2.30 pp | 58.50% |
| `>= 0.539` | 2,093 | 52.89% | 53.82% | +0.93 pp | 60.25% |

### Jhin BOTTOM `crit` with ally CC

Immobile crit marksman; measured synergy with team CC is near flat.

**Gap MSE** 2.55 pp^2 | **Mean abs gap** 1.23 pp | **Accuracy** 57.33% | **Accuracy if calibrated** 57.41% | **Calibration lift** +0.08 pp | **Empirical effect** +0.04 pp | **HGNN effect** +2.72 pp | **Shrinkage** 72.04x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.374` | 889 | 48.93% | 45.81% | -3.12 pp | 58.27% |
| `0.374-0.429` | 2,204 | 48.28% | 46.91% | -1.37 pp | 56.85% |
| `0.429-0.479` | 3,424 | 48.19% | 47.23% | -0.96 pp | 57.89% |
| `0.479-0.539` | 4,940 | 47.89% | 48.16% | +0.27 pp | 57.45% |
| `>= 0.539` | 7,421 | 48.97% | 48.53% | -0.43 pp | 57.03% |

### Lulu UTILITY `utility_protection` with ally damage

Enchanter value rises with carry damage to amplify and peel for.

**Gap MSE** 1.67 pp^2 | **Mean abs gap** 1.13 pp | **Accuracy** 57.33% | **Accuracy if calibrated** 57.30% | **Calibration lift** -0.03 pp | **Empirical effect** +4.03 pp | **HGNN effect** +5.49 pp | **Shrinkage** 1.36x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 4,473 | 48.25% | 49.09% | +0.85 pp | 57.61% |
| `0.739-0.764` | 3,793 | 50.07% | 50.95% | +0.88 pp | 57.29% |
| `0.764-0.785` | 2,907 | 51.26% | 51.73% | +0.47 pp | 58.27% |
| `0.785-0.813` | 1,841 | 51.11% | 52.25% | +1.14 pp | 54.86% |
| `>= 0.813` | 329 | 52.28% | 54.59% | +2.31 pp | 59.57% |

### Ezreal BOTTOM `attack_damage` vs enemy hard CC

Skillshot poke marksman punished as enemy hard CC stacks.

**Gap MSE** 2.59 pp^2 | **Mean abs gap** 1.20 pp | **Accuracy** 57.88% | **Accuracy if calibrated** 57.98% | **Calibration lift** +0.09 pp | **Empirical effect** -0.62 pp | **HGNN effect** +0.25 pp | **Shrinkage** -0.40x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 8,028 | 47.58% | 49.37% | +1.79 pp | 57.40% |
| `1` | 12,577 | 48.25% | 48.48% | +0.24 pp | 58.59% |
| `2` | 6,129 | 48.51% | 48.38% | -0.12 pp | 56.94% |
| `>= 3` | 1,250 | 46.96% | 49.62% | +2.66 pp | 58.48% |

### Jayce TOP `attack_damage` vs enemy frontline count

Poke bruiser empirically holds up into frontline-heavy teams; model heavily shrinks the effect.

**Gap MSE** 0.65 pp^2 | **Mean abs gap** 0.58 pp | **Accuracy** 58.26% | **Accuracy if calibrated** 58.82% | **Calibration lift** +0.56 pp | **Empirical effect** +5.93 pp | **HGNN effect** +5.89 pp | **Shrinkage** 0.99x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,276 | 47.58% | 47.71% | +0.12 pp | 58.52% |
| `1` | 3,860 | 47.90% | 48.53% | +0.62 pp | 57.67% |
| `2` | 1,862 | 48.71% | 50.19% | +1.48 pp | 59.72% |
| `>= 3` | 342 | 53.51% | 53.60% | +0.09 pp | 55.26% |

### LeeSin JUNGLE `attack_damage` vs enemy scaling

Early-tempo bruiser jungler fades as enemy scaling rises.

**Gap MSE** 0.50 pp^2 | **Mean abs gap** 0.56 pp | **Accuracy** 57.70% | **Accuracy if calibrated** 57.68% | **Calibration lift** -0.03 pp | **Empirical effect** -4.30 pp | **HGNN effect** -2.75 pp | **Shrinkage** 0.64x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.829` | 2,907 | 52.98% | 51.77% | -1.20 pp | 57.48% |
| `0.829-0.841` | 2,882 | 49.31% | 49.42% | +0.11 pp | 59.72% |
| `0.841-0.852` | 3,073 | 49.66% | 48.71% | -0.94 pp | 57.44% |
| `0.852-0.863` | 2,808 | 48.72% | 48.55% | -0.17 pp | 57.80% |
| `>= 0.863` | 3,211 | 48.68% | 49.03% | +0.35 pp | 56.28% |

### Caitlyn BOTTOM `crit` vs enemy burst count

Immobile siege ADC punished by multiple burst and dive threats.

**Gap MSE** 0.67 pp^2 | **Mean abs gap** 0.78 pp | **Accuracy** 56.80% | **Accuracy if calibrated** 56.98% | **Calibration lift** +0.17 pp | **Empirical effect** -2.29 pp | **HGNN effect** -1.93 pp | **Shrinkage** 0.84x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,569 | 49.49% | 50.04% | +0.55 pp | 56.53% |
| `1` | 8,114 | 48.62% | 49.71% | +1.09 pp | 56.90% |
| `2` | 3,628 | 48.70% | 49.29% | +0.59 pp | 56.86% |
| `>= 3` | 517 | 47.20% | 48.10% | +0.91 pp | 57.25% |


## Overall Summary

Detailed audit tables above are rendered from the `val` split.

| Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 46 | 200 | 1.43 pp | 10.46 pp | 4.01 pp^2 | 57.85% | 57.88% | +0.04 pp |

| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,145,051 | 11,450,510 | 46 | 200 | 0.75 pp | 5.31 pp | 1.33 pp^2 | 58.59% | 58.60% | +0.01 pp |
| Validation | 143,131 | 1,431,310 | 46 | 200 | 1.43 pp | 10.46 pp | 4.01 pp^2 | 57.85% | 57.88% | +0.04 pp |
| Test | 143,131 | 1,431,310 | 46 | 200 | 1.45 pp | 8.20 pp | 4.02 pp^2 | 57.26% | 57.30% | +0.04 pp |

Gap MSE is `mean((HGNN_focus_WR - empirical_focus_WR)^2)` across populated threshold bins, rendered as percentage-points squared.

## Reproduction Commands

The checked-in report uses the focus-slot audit path. Checkpoints with semantic MoE slot deltas are scored with per-slot focus-side probabilities instead of one repeated match-level probability. Regenerate predictions from the selected checkpoint with `--refresh-predictions`; omit it to reuse the prediction cache for report-only updates.

```bash
uv run python -m app.ml.context_examples_audit \
  --context-cache-dir app/ml/data/cache \
  --model-cache-dir app/ml/data/experiments/semantic_context_compact_cache \
  --model-path app/ml/data/experiments/semantic_context_compact_run/model.pt \
  --prediction-cache app/ml/data/experiments/overfit_mitigation/residual_seed4_uncertainty/residual_train_core_support_w3_reg_uncert_huber/audit_focus.npy \
  --audit-split val \
  --output app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md \
  --refresh-predictions
```
