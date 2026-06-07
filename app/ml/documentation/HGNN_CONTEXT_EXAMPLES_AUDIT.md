# HGNN Context Examples Audit

Updated: 2026-06-07.

This audit joins the empirical focus-side context examples to the trained semantic HGNN predictions for the same cached games. Each audit is its own table: one row per threshold bin reporting `n / empirical WR / HGNN WR / gap / accuracy`, with a per-table Gap MSE, accuracy, and the accuracy headroom from perfect calibration (`Calibration lift`) above it. Gap is `HGNN WR - empirical WR`; zero gap is the target.

## Scope And Threshold Definitions

- Context source: `app/ml/data/cache` side-row arrays, all splits combined.
- HGNN model: `app/ml/data/hgnn_production_model.pt`.
- HGNN cache: `app/ml/data/cache`.
- Encoder sidecar artifact: `app/ml/data/experiments/semantic_identity_sidecar_compact.npz`.
- HGNN WR uses focus-slot semantic MoE probabilities when a checkpoint exposes slot deltas; older checkpoints fall back to raw `final_logit` probabilities.
- Semantic group feature schema: v2, 25 compact per-slot features; used only by checkpoints trained with `--use-semantic-group-features`.
- Games audited: 1,431,313.
- Focus-slot rows audited: 14,313,130.
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
| Headline Trajectory Audit Tables | 10 | 47 | 1.52 pp | 6.40 pp | 4.53 pp^2 | 58.45% | 58.47% | +0.03 pp |
| Richer Composition Trajectory Tables | 13 | 52 | 1.16 pp | 5.31 pp | 2.47 pp^2 | 58.02% | 58.06% | +0.04 pp |
| Retained Prior And User-Requested Trajectory Tables | 12 | 53 | 1.61 pp | 4.56 pp | 4.21 pp^2 | 58.38% | 58.47% | +0.09 pp |
| Inspected Lower-Signal Trajectory Tables | 4 | 16 | 0.73 pp | 1.69 pp | 0.65 pp^2 | 58.11% | 58.12% | +0.02 pp |
| Top-20 Matchup And Synergy Audits | 7 | 32 | 1.18 pp | 2.68 pp | 1.78 pp^2 | 57.94% | 58.00% | +0.05 pp |

## Train, Validation, And Test Summary

These rows reuse the same audit specs and prediction cache, but evaluate the cached train, validation, and test ranges separately.

| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,145,051 | 11,450,510 | 46 | 200 | 1.36 pp | 6.47 pp | 3.34 pp^2 | 58.31% | 58.34% | +0.03 pp |
| Validation | 143,131 | 1,431,310 | 46 | 200 | 1.78 pp | 10.73 pp | 5.97 pp^2 | 57.70% | 57.76% | +0.05 pp |
| Test | 143,131 | 1,431,310 | 46 | 200 | 1.72 pp | 8.76 pp | 5.40 pp^2 | 57.17% | 57.22% | +0.04 pp |

## Enemy Count Tail Shrinkage

| Audit | Axis | Baseline bin | Tail bin | Empirical tail effect | HGNN tail effect | Shrinkage |
|---|---|---:|---:|---:|---:|---:|
| Sylas MIDDLE `ability_power` vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -3.98 pp | -1.64 pp | 0.41x |
| Nilah BOTTOM any build vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -8.47 pp | -3.77 pp | 0.45x |
| Kaisa BOTTOM any build vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -1.12 pp | -1.46 pp | 1.31x |
| Kaisa BOTTOM `on_hit` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +4.46 pp | +4.74 pp | 1.06x |
| Ahri MIDDLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +6.04 pp | +5.41 pp | 0.89x |
| Sylas JUNGLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +5.78 pp | +4.71 pp | 0.82x |
| Sylas MIDDLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +5.22 pp | +2.81 pp | 0.54x |
| Karma UTILITY any build vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +5.47 pp | +6.18 pp | 1.13x |
| Vayne BOTTOM `on_hit` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +10.84 pp | +9.40 pp | 0.87x |
| Thresh UTILITY `ar_tank` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -2.82 pp | -3.92 pp | 1.39x |
| Nautilus UTILITY `mr_tank` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -2.55 pp | -0.98 pp | 0.38x |
| Zed MIDDLE `lethality` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -7.61 pp | -4.77 pp | 0.63x |
| Nami UTILITY `utility_protection` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.32 pp | -4.70 pp | 1.09x |
| Jinx BOTTOM `crit` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -5.80 pp | -5.56 pp | 0.96x |
| Malphite TOP `ar_tank` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -10.76 pp | -11.10 pp | 1.03x |
| Viego JUNGLE any build vs enemy high-HP count | `enemy_high_hp_count` | `0` | `>= 3` | +2.93 pp | +2.41 pp | 0.82x |
| Darius TOP any build vs enemy range count | `enemy_ranged_count` | `<= 1` | `>= 4` | -4.67 pp | -3.93 pp | 0.84x |
| MasterYi JUNGLE any build vs enemy hard CC | `enemy_hard_cc_count` | `0` | `>= 3` | -2.61 pp | -2.97 pp | 1.14x |
| Focus HP `<= 2309` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.21 pp | -4.62 pp | 1.10x |
| Focus HP `>= 2478` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.62 pp | -4.70 pp | 1.02x |
| Ahri MIDDLE `ability_power` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | +0.66 pp | -1.45 pp | -2.20x |
| Kaisa BOTTOM `on_hit` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -1.92 pp | -1.07 pp | 0.55x |
| Ezreal BOTTOM `attack_damage` vs enemy hard CC | `enemy_hard_cc_count` | `0` | `>= 3` | -0.74 pp | +0.79 pp | -1.06x |
| Jayce TOP `attack_damage` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +5.83 pp | +5.19 pp | 0.89x |
| Caitlyn BOTTOM `crit` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -3.79 pp | -2.33 pp | 0.61x |

## Headline Trajectory Audit Tables

### Yasuo TOP `crit` vs enemy siege

Melee crit carry punished by poke and siege.

**Gap MSE** 3.40 pp^2 | **Mean abs gap** 1.60 pp | **Accuracy** 57.51% | **Accuracy if calibrated** 57.57% | **Calibration lift** +0.06 pp | **Empirical effect** -2.12 pp | **HGNN effect** -4.33 pp | **Shrinkage** 2.04x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.441` | 7,898 | 53.14% | 55.62% | +2.48 pp | 58.58% |
| `0.441-0.471` | 7,966 | 51.19% | 53.90% | +2.71 pp | 58.02% |
| `0.471-0.499` | 8,418 | 51.60% | 53.17% | +1.57 pp | 57.16% |
| `0.499-0.530` | 8,025 | 51.40% | 52.37% | +0.97 pp | 57.16% |
| `>= 0.530` | 7,901 | 51.02% | 51.29% | +0.27 pp | 56.66% |

### Graves JUNGLE `lethality` vs enemy damage

Burst jungler into high enemy damage.

**Gap MSE** 19.82 pp^2 | **Mean abs gap** 4.24 pp | **Accuracy** 69.74% | **Accuracy if calibrated** 69.97% | **Calibration lift** +0.23 pp | **Empirical effect** -14.62 pp | **HGNN effect** -10.57 pp | **Shrinkage** 0.72x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 4,165 | 40.67% | 34.27% | -6.40 pp | 65.31% |
| `0.739-0.764` | 3,431 | 32.96% | 28.01% | -4.95 pp | 68.64% |
| `0.764-0.785` | 2,953 | 29.90% | 26.14% | -3.76 pp | 71.01% |
| `0.785-0.813` | 2,842 | 28.68% | 24.94% | -3.74 pp | 72.38% |
| `>= 0.813` | 2,284 | 26.05% | 23.70% | -2.35 pp | 74.56% |

### Yasuo MIDDLE `crit` vs enemy siege

Same melee-carry-into-poke pattern across lane.

**Gap MSE** 0.61 pp^2 | **Mean abs gap** 0.72 pp | **Accuracy** 57.94% | **Accuracy if calibrated** 57.92% | **Calibration lift** -0.03 pp | **Empirical effect** -2.73 pp | **HGNN effect** -2.30 pp | **Shrinkage** 0.84x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.441` | 18,220 | 53.16% | 51.99% | -1.17 pp | 58.69% |
| `0.441-0.471` | 17,804 | 51.51% | 50.86% | -0.65 pp | 58.26% |
| `0.471-0.499` | 18,580 | 50.71% | 50.49% | -0.22 pp | 57.48% |
| `0.499-0.530` | 18,437 | 50.83% | 50.03% | -0.80 pp | 57.82% |
| `>= 0.530` | 18,885 | 50.43% | 49.69% | -0.74 pp | 57.50% |

### Ahri MIDDLE `ability_power` vs enemy scaling

AP mid into scaling enemy compositions.

**Gap MSE** 0.88 pp^2 | **Mean abs gap** 0.79 pp | **Accuracy** 57.72% | **Accuracy if calibrated** 57.70% | **Calibration lift** -0.01 pp | **Empirical effect** -2.98 pp | **HGNN effect** -1.86 pp | **Shrinkage** 0.62x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.829` | 31,972 | 52.74% | 52.81% | +0.07 pp | 58.60% |
| `0.829-0.841` | 30,274 | 50.55% | 50.91% | +0.36 pp | 57.97% |
| `0.841-0.852` | 32,419 | 49.01% | 50.45% | +1.44 pp | 57.48% |
| `0.852-0.863` | 28,389 | 49.48% | 50.38% | +0.90 pp | 57.29% |
| `>= 0.863` | 29,735 | 49.77% | 50.95% | +1.19 pp | 57.18% |

### Nautilus UTILITY `mr_tank` with ally damage

Engage support with damage behind it.

**Gap MSE** 0.36 pp^2 | **Mean abs gap** 0.42 pp | **Accuracy** 58.19% | **Accuracy if calibrated** 58.23% | **Calibration lift** +0.03 pp | **Empirical effect** +10.70 pp | **HGNN effect** +9.28 pp | **Shrinkage** 0.87x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 7,027 | 44.46% | 44.66% | +0.21 pp | 60.37% |
| `0.739-0.764` | 7,616 | 47.65% | 47.21% | -0.44 pp | 57.80% |
| `0.764-0.785` | 7,512 | 49.52% | 49.28% | -0.24 pp | 56.50% |
| `0.785-0.813` | 7,393 | 50.90% | 50.89% | -0.01 pp | 58.16% |
| `>= 0.813` | 2,888 | 55.16% | 53.94% | -1.22 pp | 58.41% |

### Galio MIDDLE `mr_tank` vs enemy magic

Anti-magic tank itemization (kept off-list MR-tank).

**Gap MSE** 4.88 pp^2 | **Mean abs gap** 1.90 pp | **Accuracy** 60.48% | **Accuracy if calibrated** 60.27% | **Calibration lift** -0.21 pp | **Empirical effect** +9.11 pp | **HGNN effect** +6.07 pp | **Shrinkage** 0.67x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 2,045 | 38.63% | 42.43% | +3.80 pp | 64.30% |
| `0.373-0.423` | 2,734 | 41.51% | 43.16% | +1.64 pp | 61.96% |
| `0.423-0.486` | 3,300 | 41.52% | 43.94% | +2.43 pp | 61.30% |
| `0.486-0.549` | 4,798 | 43.41% | 44.28% | +0.87 pp | 59.55% |
| `>= 0.549` | 6,179 | 47.74% | 48.51% | +0.77 pp | 58.86% |

### Malphite TOP `ar_tank` vs enemy physical

Armor tank into AD-heavy enemies.

**Gap MSE** 0.39 pp^2 | **Mean abs gap** 0.51 pp | **Accuracy** 58.03% | **Accuracy if calibrated** 58.04% | **Calibration lift** +0.01 pp | **Empirical effect** +9.32 pp | **HGNN effect** +10.32 pp | **Shrinkage** 1.11x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 7,668 | 44.98% | 44.92% | -0.06 pp | 59.23% |
| `0.387-0.448` | 9,238 | 46.83% | 46.12% | -0.71 pp | 57.06% |
| `0.448-0.508` | 12,809 | 49.44% | 49.55% | +0.10 pp | 57.75% |
| `0.508-0.557` | 15,162 | 51.35% | 52.08% | +0.73 pp | 57.11% |
| `>= 0.557` | 16,773 | 54.30% | 55.23% | +0.94 pp | 59.07% |

### Sylas MIDDLE `ability_power` vs enemy range

Short-range AP battlemage into enemy range pressure.

**Gap MSE** 4.26 pp^2 | **Mean abs gap** 1.87 pp | **Accuracy** 57.26% | **Accuracy if calibrated** 57.41% | **Calibration lift** +0.14 pp | **Empirical effect** -3.98 pp | **HGNN effect** -1.64 pp | **Shrinkage** 0.41x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 6,684 | 53.86% | 54.81% | +0.95 pp | 59.41% |
| `2` | 26,282 | 52.29% | 53.79% | +1.50 pp | 57.48% |
| `3` | 29,776 | 51.87% | 53.60% | +1.73 pp | 56.82% |
| `>= 4` | 7,476 | 49.88% | 53.18% | +3.30 pp | 56.35% |

### Nilah BOTTOM any build vs enemy range

Melee bot lane into range-heavy teams (kept off-list melee-ADC).

**Gap MSE** 10.88 pp^2 | **Mean abs gap** 2.83 pp | **Accuracy** 58.00% | **Accuracy if calibrated** 58.46% | **Calibration lift** +0.45 pp | **Empirical effect** -8.47 pp | **HGNN effect** -3.77 pp | **Shrinkage** 0.45x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 2,368 | 58.99% | 53.53% | -5.47 pp | 58.95% |
| `2` | 9,098 | 54.21% | 51.47% | -2.74 pp | 58.88% |
| `3` | 10,090 | 52.77% | 50.41% | -2.35 pp | 57.14% |
| `>= 4` | 2,476 | 50.53% | 49.76% | -0.77 pp | 57.39% |

### Kaisa BOTTOM any build vs enemy range

High-sample marksman vs enemy range pressure; large n keeps bins low-noise.

**Gap MSE** 0.23 pp^2 | **Mean abs gap** 0.40 pp | **Accuracy** 58.88% | **Accuracy if calibrated** 58.88% | **Calibration lift** -0.00 pp | **Empirical effect** -1.12 pp | **HGNN effect** -1.46 pp | **Shrinkage** 1.31x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 23,417 | 49.16% | 49.95% | +0.78 pp | 59.55% |
| `2` | 93,644 | 48.95% | 49.27% | +0.33 pp | 58.88% |
| `3` | 105,204 | 48.89% | 48.95% | +0.06 pp | 58.75% |
| `>= 4` | 26,296 | 48.05% | 48.49% | +0.44 pp | 58.78% |


## Richer Composition Trajectory Tables

### Kaisa BOTTOM `on_hit` vs enemy frontline count

On-hit marksman shreds added enemy frontline.

**Gap MSE** 0.67 pp^2 | **Mean abs gap** 0.80 pp | **Accuracy** 59.15% | **Accuracy if calibrated** 59.13% | **Calibration lift** -0.01 pp | **Empirical effect** +4.46 pp | **HGNN effect** +4.74 pp | **Shrinkage** 1.06x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 53,694 | 46.59% | 47.39% | +0.80 pp | 59.30% |
| `1` | 93,519 | 47.26% | 47.91% | +0.64 pp | 58.95% |
| `2` | 47,323 | 48.66% | 49.35% | +0.69 pp | 59.12% |
| `>= 3` | 9,099 | 51.05% | 52.13% | +1.08 pp | 60.34% |

### Ahri MIDDLE `ability_power` vs enemy frontline count

AP mid improves as enemies stack durable targets.

**Gap MSE** 0.55 pp^2 | **Mean abs gap** 0.65 pp | **Accuracy** 57.72% | **Accuracy if calibrated** 57.67% | **Calibration lift** -0.04 pp | **Empirical effect** +6.04 pp | **HGNN effect** +5.41 pp | **Shrinkage** 0.89x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 41,371 | 49.24% | 49.95% | +0.71 pp | 56.94% |
| `1` | 70,211 | 50.04% | 50.83% | +0.78 pp | 58.03% |
| `2` | 34,661 | 51.28% | 52.31% | +1.03 pp | 57.77% |
| `>= 3` | 6,546 | 55.29% | 55.36% | +0.07 pp | 58.97% |

### Sylas JUNGLE `ability_power` vs enemy frontline count

Sustained AP skirmisher into beefy teams.

**Gap MSE** 3.90 pp^2 | **Mean abs gap** 1.94 pp | **Accuracy** 57.16% | **Accuracy if calibrated** 57.37% | **Calibration lift** +0.21 pp | **Empirical effect** +5.78 pp | **HGNN effect** +4.71 pp | **Shrinkage** 0.82x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 13,443 | 49.74% | 52.17% | +2.43 pp | 56.52% |
| `1` | 22,739 | 51.26% | 53.06% | +1.80 pp | 57.25% |
| `2` | 11,247 | 52.56% | 54.71% | +2.15 pp | 57.16% |
| `>= 3` | 1,994 | 55.52% | 56.88% | +1.36 pp | 60.48% |

### Sylas MIDDLE `ability_power` vs enemy frontline count

Same AP anti-frontline pattern from lane.

**Gap MSE** 2.71 pp^2 | **Mean abs gap** 1.35 pp | **Accuracy** 57.26% | **Accuracy if calibrated** 57.46% | **Calibration lift** +0.20 pp | **Empirical effect** +5.22 pp | **HGNN effect** +2.81 pp | **Shrinkage** 0.54x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 18,611 | 50.59% | 53.16% | +2.57 pp | 56.71% |
| `1` | 32,065 | 51.68% | 53.58% | +1.90 pp | 57.26% |
| `2` | 16,435 | 53.51% | 54.29% | +0.78 pp | 57.72% |
| `>= 3` | 3,107 | 55.81% | 55.97% | +0.16 pp | 58.22% |

### Karma UTILITY any build vs enemy frontline count

Utility support gains value as enemies stack frontline to zone.

**Gap MSE** 2.52 pp^2 | **Mean abs gap** 1.57 pp | **Accuracy** 57.97% | **Accuracy if calibrated** 58.12% | **Calibration lift** +0.15 pp | **Empirical effect** +5.47 pp | **HGNN effect** +6.18 pp | **Shrinkage** 1.13x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 54,752 | 47.58% | 48.82% | +1.24 pp | 57.63% |
| `1` | 87,323 | 48.63% | 50.25% | +1.61 pp | 58.01% |
| `2` | 40,207 | 50.70% | 52.15% | +1.45 pp | 58.01% |
| `>= 3` | 6,998 | 53.04% | 55.00% | +1.95 pp | 59.82% |

### Vayne BOTTOM `on_hit` vs enemy frontline count

Classic anti-tank marksman pattern.

**Gap MSE** 14.04 pp^2 | **Mean abs gap** 3.57 pp | **Accuracy** 57.16% | **Accuracy if calibrated** 57.73% | **Calibration lift** +0.58 pp | **Empirical effect** +10.84 pp | **HGNN effect** +9.40 pp | **Shrinkage** 0.87x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 6,867 | 48.45% | 44.58% | -3.87 pp | 56.08% |
| `1` | 13,525 | 49.47% | 46.94% | -2.53 pp | 57.55% |
| `2` | 7,822 | 52.26% | 49.70% | -2.56 pp | 57.03% |
| `>= 3` | 1,685 | 59.29% | 53.98% | -5.31 pp | 58.99% |

### Thresh UTILITY `ar_tank` vs enemy burst count

Durable engage support punished by multiple burst threats.

**Gap MSE** 1.33 pp^2 | **Mean abs gap** 0.99 pp | **Accuracy** 58.25% | **Accuracy if calibrated** 58.22% | **Calibration lift** -0.03 pp | **Empirical effect** -2.82 pp | **HGNN effect** -3.92 pp | **Shrinkage** 1.39x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 10,691 | 50.01% | 49.31% | -0.70 pp | 58.73% |
| `1` | 20,042 | 49.28% | 48.05% | -1.23 pp | 57.97% |
| `2` | 8,459 | 47.12% | 46.90% | -0.23 pp | 58.36% |
| `>= 3` | 1,123 | 47.20% | 45.39% | -1.80 pp | 57.88% |

### Nautilus UTILITY `mr_tank` vs enemy burst count

High-HP engage tank loses into concentrated burst.

**Gap MSE** 0.54 pp^2 | **Mean abs gap** 0.57 pp | **Accuracy** 58.19% | **Accuracy if calibrated** 58.16% | **Calibration lift** -0.03 pp | **Empirical effect** -2.55 pp | **HGNN effect** -0.98 pp | **Shrinkage** 0.38x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 8,856 | 49.15% | 48.93% | -0.22 pp | 58.94% |
| `1` | 16,138 | 48.61% | 48.41% | -0.21 pp | 58.29% |
| `2` | 6,558 | 49.09% | 48.59% | -0.49 pp | 56.95% |
| `>= 3` | 884 | 46.61% | 47.96% | +1.35 pp | 58.03% |

### Zed MIDDLE `lethality` vs enemy burst count

Assassin into enemy burst stacking.

**Gap MSE** 3.57 pp^2 | **Mean abs gap** 1.49 pp | **Accuracy** 58.35% | **Accuracy if calibrated** 58.46% | **Calibration lift** +0.11 pp | **Empirical effect** -7.61 pp | **HGNN effect** -4.77 pp | **Shrinkage** 0.63x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 13,791 | 52.56% | 53.22% | +0.65 pp | 59.21% |
| `1` | 23,959 | 50.79% | 51.70% | +0.91 pp | 58.11% |
| `2` | 9,118 | 49.79% | 50.70% | +0.90 pp | 57.95% |
| `>= 3` | 1,170 | 44.96% | 48.45% | +3.49 pp | 56.41% |

### Nami UTILITY `utility_protection` vs enemy burst count

Protective enchanter punished by burst-heavy enemies.

**Gap MSE** 0.04 pp^2 | **Mean abs gap** 0.18 pp | **Accuracy** 57.73% | **Accuracy if calibrated** 57.73% | **Calibration lift** +0.00 pp | **Empirical effect** -4.32 pp | **HGNN effect** -4.70 pp | **Shrinkage** 1.09x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 43,445 | 51.59% | 51.70% | +0.10 pp | 57.79% |
| `1` | 75,187 | 50.36% | 50.54% | +0.17 pp | 57.84% |
| `2` | 30,847 | 49.34% | 49.18% | -0.16 pp | 57.33% |
| `>= 3` | 4,292 | 47.27% | 46.99% | -0.28 pp | 58.13% |

### Jinx BOTTOM `crit` vs enemy burst count

Fragile crit carry into burst-heavy enemies.

**Gap MSE** 0.06 pp^2 | **Mean abs gap** 0.23 pp | **Accuracy** 57.58% | **Accuracy if calibrated** 57.58% | **Calibration lift** -0.00 pp | **Empirical effect** -5.80 pp | **HGNN effect** -5.56 pp | **Shrinkage** 0.96x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 45,003 | 54.67% | 54.30% | -0.37 pp | 58.40% |
| `1` | 89,365 | 52.75% | 52.50% | -0.25 pp | 57.43% |
| `2` | 36,863 | 51.10% | 50.95% | -0.15 pp | 57.06% |
| `>= 3` | 4,933 | 48.87% | 48.74% | -0.14 pp | 56.90% |

### Malphite TOP `ar_tank` vs heavy damage-taken count

Armor tank loses into teams with multiple high-soak targets.

**Gap MSE** 1.98 pp^2 | **Mean abs gap** 1.31 pp | **Accuracy** 58.03% | **Accuracy if calibrated** 57.97% | **Calibration lift** -0.06 pp | **Empirical effect** -10.76 pp | **HGNN effect** -11.10 pp | **Shrinkage** 1.03x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 15,788 | 53.19% | 54.97% | +1.78 pp | 57.96% |
| `1` | 29,695 | 50.09% | 50.54% | +0.45 pp | 58.09% |
| `2` | 14,524 | 48.42% | 46.86% | -1.56 pp | 57.79% |
| `>= 3` | 1,643 | 42.42% | 43.87% | +1.45 pp | 59.89% |

### Viego JUNGLE any build vs enemy high-HP count

On-hit bruiser jungler into high-HP enemy teams.

**Gap MSE** 0.21 pp^2 | **Mean abs gap** 0.39 pp | **Accuracy** 58.24% | **Accuracy if calibrated** 58.24% | **Calibration lift** -0.00 pp | **Empirical effect** +2.93 pp | **HGNN effect** +2.41 pp | **Shrinkage** 0.82x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 35,197 | 46.86% | 46.80% | -0.05 pp | 58.99% |
| `1` | 67,762 | 47.85% | 47.21% | -0.64 pp | 58.08% |
| `2` | 47,858 | 47.93% | 48.24% | +0.30 pp | 57.97% |
| `>= 3` | 16,771 | 49.78% | 49.21% | -0.57 pp | 58.12% |


## Retained Prior And User-Requested Trajectory Tables

### Malphite all roles `ar_tank` vs enemy physical

Original armor-stack audit, retained beyond TOP-only.

**Gap MSE** 0.80 pp^2 | **Mean abs gap** 0.72 pp | **Accuracy** 57.88% | **Accuracy if calibrated** 57.91% | **Calibration lift** +0.03 pp | **Empirical effect** +8.94 pp | **HGNN effect** +10.38 pp | **Shrinkage** 1.16x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 10,529 | 45.48% | 44.29% | -1.19 pp | 58.97% |
| `0.387-0.448` | 13,017 | 46.88% | 45.38% | -1.49 pp | 56.96% |
| `0.448-0.508` | 16,829 | 49.06% | 48.89% | -0.18 pp | 57.54% |
| `0.508-0.557` | 19,526 | 50.86% | 51.34% | +0.49 pp | 57.06% |
| `>= 0.557` | 22,143 | 54.42% | 54.68% | +0.26 pp | 58.86% |

### Galio all roles `mr_tank` vs enemy magic

Original anti-magic tank family, broader than MIDDLE-only.

**Gap MSE** 5.85 pp^2 | **Mean abs gap** 2.23 pp | **Accuracy** 60.17% | **Accuracy if calibrated** 59.97% | **Calibration lift** -0.20 pp | **Empirical effect** +9.54 pp | **HGNN effect** +7.04 pp | **Shrinkage** 0.74x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 2,182 | 39.05% | 42.92% | +3.87 pp | 64.25% |
| `0.373-0.423` | 2,969 | 41.93% | 43.76% | +1.83 pp | 62.14% |
| `0.423-0.486` | 3,637 | 42.10% | 44.74% | +2.65 pp | 60.52% |
| `0.486-0.549` | 5,566 | 43.96% | 45.37% | +1.41 pp | 59.32% |
| `>= 0.549` | 7,642 | 48.59% | 49.96% | +1.38 pp | 58.68% |

### Nautilus all roles `mr_tank` vs enemy magic

Top-20 MR-tank anti-magic case alongside Galio.

**Gap MSE** 0.24 pp^2 | **Mean abs gap** 0.44 pp | **Accuracy** 58.18% | **Accuracy if calibrated** 58.29% | **Calibration lift** +0.10 pp | **Empirical effect** +1.45 pp | **HGNN effect** +0.26 pp | **Shrinkage** 0.18x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 2,319 | 48.25% | 49.06% | +0.80 pp | 58.47% |
| `0.373-0.423` | 4,007 | 48.49% | 48.74% | +0.25 pp | 58.22% |
| `0.423-0.486` | 5,380 | 48.55% | 48.36% | -0.19 pp | 58.74% |
| `0.486-0.549` | 8,657 | 47.98% | 47.42% | -0.56 pp | 57.95% |
| `>= 0.549` | 12,612 | 49.71% | 49.32% | -0.39 pp | 58.04% |

### Nautilus all roles `ar_tank` vs enemy physical

Physical-heavy enemy teams remain a support-tank check.

**Gap MSE** 0.59 pp^2 | **Mean abs gap** 0.70 pp | **Accuracy** 58.56% | **Accuracy if calibrated** 58.59% | **Calibration lift** +0.03 pp | **Empirical effect** +4.74 pp | **HGNN effect** +6.79 pp | **Shrinkage** 1.43x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 10,135 | 46.82% | 45.98% | -0.84 pp | 58.90% |
| `0.387-0.448` | 14,076 | 46.57% | 46.32% | -0.25 pp | 58.32% |
| `0.448-0.508` | 18,450 | 48.50% | 49.01% | +0.50 pp | 58.74% |
| `0.508-0.557` | 19,707 | 49.60% | 50.29% | +0.69 pp | 58.10% |
| `>= 0.557` | 21,520 | 51.56% | 52.77% | +1.21 pp | 58.82% |

### Darius TOP any build vs enemy range count

Static team range pressure, stronger than lane-only range.

**Gap MSE** 0.51 pp^2 | **Mean abs gap** 0.54 pp | **Accuracy** 58.04% | **Accuracy if calibrated** 58.05% | **Calibration lift** +0.01 pp | **Empirical effect** -4.67 pp | **HGNN effect** -3.93 pp | **Shrinkage** 0.84x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 7,151 | 52.38% | 51.50% | -0.88 pp | 59.22% |
| `2` | 28,731 | 49.50% | 49.53% | +0.03 pp | 58.20% |
| `3` | 30,943 | 49.41% | 48.31% | -1.11 pp | 57.72% |
| `>= 4` | 7,690 | 47.71% | 47.57% | -0.14 pp | 57.66% |

### Darius TOP any build vs same-role range

User-requested static melee/ranged lane audit.

**Gap MSE** 0.89 pp^2 | **Mean abs gap** 0.94 pp | **Accuracy** 58.04% | **Accuracy if calibrated** 58.00% | **Calibration lift** -0.05 pp | **Empirical effect** -2.89 pp | **HGNN effect** -1.02 pp | **Shrinkage** 0.35x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 250` | 63,492 | 49.98% | 49.16% | -0.83 pp | 57.97% |
| `> 250` | 11,023 | 47.09% | 48.14% | +1.05 pp | 58.49% |

### MasterYi JUNGLE any build vs enemy hard CC

User-requested low-CC audit; unique even though gap is modest.

**Gap MSE** 14.38 pp^2 | **Mean abs gap** 3.78 pp | **Accuracy** 57.06% | **Accuracy if calibrated** 57.93% | **Calibration lift** +0.87 pp | **Empirical effect** -2.61 pp | **HGNN effect** -2.97 pp | **Shrinkage** 1.14x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 17,755 | 53.24% | 49.90% | -3.34 pp | 57.17% |
| `1` | 27,265 | 52.22% | 48.37% | -3.85 pp | 56.82% |
| `2` | 12,667 | 51.29% | 47.07% | -4.22 pp | 57.42% |
| `>= 3` | 2,544 | 50.63% | 46.93% | -3.70 pp | 57.00% |

### Selected enchanters UTILITY with skirmish allies

Original enchanter-with-skirmishers synergy probe.

**Gap MSE** 0.40 pp^2 | **Mean abs gap** 0.57 pp | **Accuracy** 57.94% | **Accuracy if calibrated** 57.99% | **Calibration lift** +0.05 pp | **Empirical effect** +2.66 pp | **HGNN effect** +1.17 pp | **Shrinkage** 0.44x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 382,693 | 50.30% | 51.17% | +0.87 pp | 57.91% |
| `1` | 73,328 | 52.24% | 52.46% | +0.22 pp | 58.09% |
| `>= 2` | 3,338 | 52.97% | 52.34% | -0.62 pp | 57.97% |

### Low own-damage teams vs enemy heal/shield

Original low-damage into sustain audit.

**Gap MSE** 1.26 pp^2 | **Mean abs gap** 1.07 pp | **Accuracy** 59.07% | **Accuracy if calibrated** 59.11% | **Calibration lift** +0.04 pp | **Empirical effect** -2.19 pp | **HGNN effect** -2.89 pp | **Shrinkage** 1.32x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.028` | 114,704 | 49.67% | 51.01% | +1.33 pp | 59.21% |
| `0.028-0.077` | 116,203 | 48.25% | 49.80% | +1.55 pp | 59.32% |
| `0.077-0.200` | 111,021 | 47.38% | 48.44% | +1.07 pp | 59.36% |
| `0.200-0.202` | 120,704 | 47.52% | 48.28% | +0.76 pp | 58.63% |
| `>= 0.202` | 117,404 | 47.48% | 48.12% | +0.63 pp | 58.85% |

### Ambessa TOP `attack_damage` vs enemy damage

Durable bruiser into enemy damage pressure.

**Gap MSE** 12.84 pp^2 | **Mean abs gap** 3.53 pp | **Accuracy** 57.44% | **Accuracy if calibrated** 57.91% | **Calibration lift** +0.47 pp | **Empirical effect** -2.55 pp | **HGNN effect** -4.14 pp | **Shrinkage** 1.63x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 16,336 | 51.19% | 55.74% | +4.56 pp | 57.62% |
| `0.739-0.764` | 16,356 | 49.13% | 52.91% | +3.79 pp | 57.38% |
| `0.764-0.785` | 15,526 | 48.56% | 51.62% | +3.06 pp | 57.27% |
| `0.785-0.813` | 16,093 | 47.82% | 51.13% | +3.31 pp | 57.04% |
| `>= 0.813` | 15,631 | 48.64% | 51.60% | +2.96 pp | 57.91% |

### LeeSin JUNGLE `ad_off_tank` vs enemy magic

Bruiser jungler resisting magic-heavy enemies.

**Gap MSE** 5.00 pp^2 | **Mean abs gap** 2.03 pp | **Accuracy** 58.55% | **Accuracy if calibrated** 58.73% | **Calibration lift** +0.18 pp | **Empirical effect** -0.61 pp | **HGNN effect** -3.43 pp | **Shrinkage** 5.64x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 6,818 | 46.74% | 50.28% | +3.53 pp | 57.57% |
| `0.373-0.423` | 6,646 | 45.70% | 48.13% | +2.43 pp | 59.04% |
| `0.423-0.486` | 6,308 | 45.94% | 47.85% | +1.91 pp | 58.61% |
| `0.486-0.549` | 5,551 | 44.51% | 46.08% | +1.56 pp | 59.07% |
| `>= 0.549` | 4,645 | 46.14% | 46.85% | +0.71 pp | 58.60% |

### Thresh UTILITY `mr_tank` vs enemy magic

MR-tank support anti-magic case.

**Gap MSE** 5.58 pp^2 | **Mean abs gap** 2.15 pp | **Accuracy** 58.75% | **Accuracy if calibrated** 58.71% | **Calibration lift** -0.03 pp | **Empirical effect** -0.62 pp | **HGNN effect** +2.49 pp | **Shrinkage** -3.99x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 717 | 49.37% | 47.37% | -2.01 pp | 58.58% |
| `0.373-0.423` | 1,123 | 43.28% | 46.90% | +3.62 pp | 59.93% |
| `0.423-0.486` | 1,558 | 44.09% | 46.98% | +2.88 pp | 61.10% |
| `0.486-0.549` | 2,280 | 47.32% | 46.22% | -1.11 pp | 57.06% |
| `>= 0.549` | 3,916 | 48.75% | 49.86% | +1.11 pp | 58.48% |


## Inspected Lower-Signal Trajectory Tables

### Focus HP `<= 2309` vs enemy burst count

Broad HP-vs-burst check; useful but lower signal than champion-specific rows.

**Gap MSE** 0.30 pp^2 | **Mean abs gap** 0.52 pp | **Accuracy** 58.03% | **Accuracy if calibrated** 58.04% | **Calibration lift** +0.01 pp | **Empirical effect** -4.21 pp | **HGNN effect** -4.62 pp | **Shrinkage** 1.10x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 880,508 | 51.83% | 51.46% | -0.37 pp | 58.42% |
| `1` | 1,544,006 | 50.68% | 50.31% | -0.37 pp | 57.99% |
| `2` | 634,311 | 49.54% | 48.97% | -0.57 pp | 57.64% |
| `>= 3` | 85,831 | 47.62% | 46.84% | -0.78 pp | 57.84% |

### Focus HP `>= 2478` vs enemy burst count

High-HP slots also drop into burst stacks, so champion/build specificity matters.

**Gap MSE** 0.41 pp^2 | **Mean abs gap** 0.64 pp | **Accuracy** 58.13% | **Accuracy if calibrated** 58.15% | **Calibration lift** +0.03 pp | **Empirical effect** -4.62 pp | **HGNN effect** -4.70 pp | **Shrinkage** 1.02x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,042,274 | 50.70% | 51.38% | +0.69 pp | 58.51% |
| `1` | 1,844,824 | 49.44% | 50.02% | +0.58 pp | 57.98% |
| `2` | 755,241 | 48.04% | 48.72% | +0.68 pp | 57.95% |
| `>= 3` | 102,599 | 46.08% | 46.68% | +0.60 pp | 58.17% |

### Ahri MIDDLE `ability_power` vs heavy damage-taken count

AP mid vs multiple high-soak enemies; weaker axis than frontline count.

**Gap MSE** 0.80 pp^2 | **Mean abs gap** 0.84 pp | **Accuracy** 57.72% | **Accuracy if calibrated** 57.66% | **Calibration lift** -0.06 pp | **Empirical effect** +0.66 pp | **HGNN effect** -1.45 pp | **Shrinkage** -2.20x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 37,448 | 50.56% | 51.88% | +1.31 pp | 57.34% |
| `1` | 73,904 | 50.28% | 51.07% | +0.78 pp | 57.85% |
| `2` | 37,062 | 50.09% | 50.55% | +0.45 pp | 57.72% |
| `>= 3` | 4,375 | 51.22% | 50.42% | -0.80 pp | 58.74% |

### Kaisa BOTTOM `on_hit` vs heavy damage-taken count

On-hit marksman vs high-soak enemies; frontline count is the stronger cut.

**Gap MSE** 1.08 pp^2 | **Mean abs gap** 0.94 pp | **Accuracy** 59.15% | **Accuracy if calibrated** 59.12% | **Calibration lift** -0.03 pp | **Empirical effect** -1.92 pp | **HGNN effect** -1.07 pp | **Shrinkage** 0.55x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 49,452 | 48.06% | 48.89% | +0.84 pp | 59.12% |
| `1` | 99,219 | 47.59% | 48.28% | +0.70 pp | 59.25% |
| `2` | 49,599 | 47.25% | 47.77% | +0.52 pp | 58.97% |
| `>= 3` | 5,365 | 46.13% | 47.83% | +1.69 pp | 59.20% |


## Top-20 Matchup And Synergy Audits

### Yasuo MIDDLE `crit` with ally CC

Yasuo's ult chains off ally knock-ups; scales with team CC.

**Gap MSE** 1.21 pp^2 | **Mean abs gap** 0.91 pp | **Accuracy** 57.94% | **Accuracy if calibrated** 57.93% | **Calibration lift** -0.02 pp | **Empirical effect** +3.31 pp | **HGNN effect** +5.60 pp | **Shrinkage** 1.69x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.374` | 12,637 | 49.88% | 47.82% | -2.05 pp | 57.77% |
| `0.374-0.429` | 18,417 | 49.93% | 49.04% | -0.89 pp | 57.69% |
| `0.429-0.479` | 20,098 | 51.03% | 50.17% | -0.87 pp | 58.16% |
| `0.479-0.539` | 21,116 | 51.90% | 51.40% | -0.50 pp | 58.01% |
| `>= 0.539` | 19,658 | 53.19% | 53.42% | +0.23 pp | 58.00% |

### Jhin BOTTOM `crit` with ally CC

Immobile crit marksman; measured synergy with team CC is near flat.

**Gap MSE** 0.54 pp^2 | **Mean abs gap** 0.63 pp | **Accuracy** 58.17% | **Accuracy if calibrated** 58.22% | **Calibration lift** +0.04 pp | **Empirical effect** +0.41 pp | **HGNN effect** +2.57 pp | **Shrinkage** 6.30x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.374` | 8,279 | 47.49% | 46.46% | -1.03 pp | 57.76% |
| `0.374-0.429` | 20,108 | 47.41% | 47.05% | -0.35 pp | 58.59% |
| `0.429-0.479` | 32,253 | 47.53% | 47.67% | +0.15 pp | 58.78% |
| `0.479-0.539` | 45,878 | 47.78% | 48.25% | +0.47 pp | 57.94% |
| `>= 0.539` | 66,205 | 47.90% | 49.03% | +1.13 pp | 57.97% |

### Lulu UTILITY `utility_protection` with ally damage

Enchanter value rises with carry damage to amplify and peel for.

**Gap MSE** 1.74 pp^2 | **Mean abs gap** 1.17 pp | **Accuracy** 57.59% | **Accuracy if calibrated** 57.73% | **Calibration lift** +0.14 pp | **Empirical effect** +4.04 pp | **HGNN effect** +5.23 pp | **Shrinkage** 1.29x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 47,255 | 48.62% | 49.75% | +1.13 pp | 58.32% |
| `0.739-0.764` | 39,565 | 51.10% | 51.71% | +0.61 pp | 57.39% |
| `0.764-0.785` | 29,105 | 51.77% | 52.45% | +0.69 pp | 57.31% |
| `0.785-0.813` | 18,598 | 52.24% | 53.33% | +1.10 pp | 56.47% |
| `>= 0.813` | 3,340 | 52.66% | 54.98% | +2.32 pp | 58.23% |

### Ezreal BOTTOM `attack_damage` vs enemy hard CC

Skillshot poke marksman punished as enemy hard CC stacks.

**Gap MSE** 3.39 pp^2 | **Mean abs gap** 1.75 pp | **Accuracy** 58.32% | **Accuracy if calibrated** 58.42% | **Calibration lift** +0.10 pp | **Empirical effect** -0.74 pp | **HGNN effect** +0.79 pp | **Shrinkage** -1.06x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 76,154 | 48.67% | 49.83% | +1.16 pp | 58.01% |
| `1` | 118,462 | 48.03% | 49.50% | +1.48 pp | 58.53% |
| `2` | 54,154 | 47.72% | 49.40% | +1.68 pp | 58.26% |
| `>= 3` | 9,969 | 47.93% | 50.61% | +2.68 pp | 58.60% |

### Jayce TOP `attack_damage` vs enemy frontline count

Poke bruiser empirically holds up into frontline-heavy teams; model heavily shrinks the effect.

**Gap MSE** 2.10 pp^2 | **Mean abs gap** 1.36 pp | **Accuracy** 57.80% | **Accuracy if calibrated** 57.72% | **Calibration lift** -0.08 pp | **Empirical effect** +5.83 pp | **HGNN effect** +5.19 pp | **Shrinkage** 0.89x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 18,731 | 47.40% | 46.13% | -1.28 pp | 57.75% |
| `1` | 32,478 | 48.68% | 47.01% | -1.67 pp | 57.52% |
| `2` | 16,468 | 49.20% | 48.63% | -0.57 pp | 58.23% |
| `>= 3` | 3,066 | 53.23% | 51.32% | -1.91 pp | 58.68% |

### LeeSin JUNGLE `attack_damage` vs enemy scaling

Early-tempo bruiser jungler fades as enemy scaling rises.

**Gap MSE** 1.45 pp^2 | **Mean abs gap** 1.19 pp | **Accuracy** 57.86% | **Accuracy if calibrated** 57.94% | **Calibration lift** +0.09 pp | **Empirical effect** -2.72 pp | **HGNN effect** -2.56 pp | **Shrinkage** 0.94x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.829` | 25,720 | 52.11% | 53.00% | +0.90 pp | 58.76% |
| `0.829-0.841` | 26,041 | 49.34% | 50.85% | +1.51 pp | 58.07% |
| `0.841-0.852` | 29,735 | 48.83% | 50.07% | +1.23 pp | 57.73% |
| `0.852-0.863` | 27,510 | 48.66% | 49.89% | +1.24 pp | 57.46% |
| `>= 0.863` | 30,969 | 49.39% | 50.44% | +1.05 pp | 57.41% |

### Caitlyn BOTTOM `crit` vs enemy burst count

Immobile siege ADC punished by multiple burst and dive threats.

**Gap MSE** 2.56 pp^2 | **Mean abs gap** 1.50 pp | **Accuracy** 57.56% | **Accuracy if calibrated** 57.56% | **Calibration lift** -0.01 pp | **Empirical effect** -3.79 pp | **HGNN effect** -2.33 pp | **Shrinkage** 0.61x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 45,426 | 50.24% | 51.22% | +0.97 pp | 58.11% |
| `1` | 87,883 | 49.66% | 50.80% | +1.13 pp | 57.42% |
| `2` | 36,686 | 48.75% | 50.20% | +1.45 pp | 57.16% |
| `>= 3` | 4,970 | 46.46% | 48.89% | +2.43 pp | 58.01% |


## Overall Summary

Detailed audit tables above are rendered from the `all` split.

| Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 46 | 200 | 1.33 pp | 6.40 pp | 3.16 pp^2 | 58.14% | 58.17% | +0.03 pp |

| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,145,051 | 11,450,510 | 46 | 200 | 1.36 pp | 6.47 pp | 3.34 pp^2 | 58.31% | 58.34% | +0.03 pp |
| Validation | 143,131 | 1,431,310 | 46 | 200 | 1.78 pp | 10.73 pp | 5.97 pp^2 | 57.70% | 57.76% | +0.05 pp |
| Test | 143,131 | 1,431,310 | 46 | 200 | 1.72 pp | 8.76 pp | 5.40 pp^2 | 57.17% | 57.22% | +0.04 pp |

Gap MSE is `mean((HGNN_focus_WR - empirical_focus_WR)^2)` across populated threshold bins, rendered as percentage-points squared.

## Reproduction Commands

The checked-in report uses the focus-slot audit path. Checkpoints with semantic MoE slot deltas are scored with per-slot focus-side probabilities instead of one repeated match-level probability. Regenerate predictions from the selected checkpoint with `--refresh-predictions`; omit it to reuse the prediction cache for report-only updates.

```bash
uv run python -m app.ml.context_examples_audit \
  --context-cache-dir app/ml/data/cache \
  --model-cache-dir app/ml/data/cache \
  --model-path app/ml/data/hgnn_production_model.pt \
  --encoder-sidecar-path app/ml/data/experiments/semantic_identity_sidecar_compact.npz \
  --prediction-cache app/ml/data/audit_focus_side_probability.npy \
  --audit-split all \
  --output app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md \
  --refresh-predictions
```
