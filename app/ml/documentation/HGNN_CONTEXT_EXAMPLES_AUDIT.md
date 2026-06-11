# HGNN Context Examples Audit

Updated: 2026-06-11.

This audit joins the empirical focus-side context examples to the trained semantic HGNN predictions for the same cached games. Each fixed-fixture audit is its own table: one row per threshold bin reporting `n / empirical WR / HGNN WR / gap / accuracy`, with a per-table Gap MSE, accuracy, and the accuracy delta from perfect calibration (`Calibration lift`) above it. Gap is `HGNN WR - empirical WR`; zero gap is the target.

## Scope And Threshold Definitions

- Context source: historical `app/ml/data/cache` side-row arrays, `val` split
  only. Rerun with `--audit-split test` for the current v32 test-only audit
  protocol.
- HGNN model: `app/ml/data/hgnn_production_model.pt`.
- HGNN cache: `app/ml/data/cache`.
- Encoder sidecar artifact: `app/ml/data/semantic_identity_sidecar_compact.npz`.
- HGNN WR uses focus-slot semantic MoE probabilities when a checkpoint exposes slot deltas; older checkpoints fall back to raw `final_logit` probabilities.
- Semantic group feature schema: v2, 25 compact per-slot features; used only by checkpoints trained with `--use-semantic-group-features`.
- Games audited: 164,792.
- Focus-slot rows audited: 1,647,920.
- Model-alignment rows score each slot with its focus-side win probability; blue-side slots use the blue-team frame and red-side slots use the mirrored red-team frame.
- Continuous thresholds are global side-row team-average percentiles.
- Count thresholds use explicit enemy-team counts.
- WR, effects, and gaps are focus-side win-rate percentage points.
- Accuracy is focus-row classification accuracy at the 0.5 threshold (HGNN focus WR >= 0.5 predicts a focus-side win); the per-table value is bin-n weighted.
- `Acc if calibrated` shifts each bin's predictions so the bin mean equals the empirical WR (perfect calibration) while keeping the model's within-bin ranking, then re-thresholds at 0.5; `Calibration lift` is that minus current accuracy. It can be negative when the calibration shift crosses the 0.5 decision threshold in an unhelpful direction.
- Selected-enchanter probe uses Sona, Karma, Lulu, and Zilean in `UTILITY` with `utility_enchanter` or `utility_protection`.
- Low own-damage probe is anchored once per team side, then compared against the enemy heal/shield context.
- Effect shrinkage is `HGNN effect / empirical effect`; values between 0 and 1 under-express a same-direction effect, negative values are sign flips, and values above 1 over-express it.

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

## Fixed Fixture Gap Summary

| Section | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Headline Trajectory Audit Tables | 10 | 47 | 1.91 pp | 8.93 pp | 6.82 pp^2 | 57.86% | 58.06% | +0.19 pp |
| Richer Composition Trajectory Tables | 13 | 52 | 1.66 pp | 5.67 pp | 4.16 pp^2 | 57.39% | 57.50% | +0.11 pp |
| Retained Prior And User-Requested Trajectory Tables | 12 | 53 | 2.26 pp | 11.09 pp | 9.71 pp^2 | 57.84% | 57.99% | +0.15 pp |
| Inspected Lower-Signal Trajectory Tables | 4 | 16 | 1.58 pp | 5.91 pp | 4.63 pp^2 | 57.58% | 57.59% | +0.02 pp |
| Top-20 Matchup And Synergy Audits | 7 | 32 | 1.23 pp | 3.73 pp | 2.39 pp^2 | 57.35% | 57.40% | +0.06 pp |

## Train, Validation, And Test Summary

These rows reuse the same audit specs and prediction cache, but evaluate the cached train, validation, and test ranges separately.

| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,318,331 | 13,183,310 | 46 | 200 | 1.43 pp | 6.67 pp | 3.46 pp^2 | 58.22% | 58.25% | +0.03 pp |
| Validation | 164,792 | 1,647,920 | 46 | 200 | 1.80 pp | 11.09 pp | 6.01 pp^2 | 57.59% | 57.64% | +0.06 pp |
| Test | 164,792 | 1,647,920 | 46 | 200 | 1.87 pp | 11.85 pp | 6.77 pp^2 | 57.34% | 57.40% | +0.06 pp |

## Expanded Empirical Discovery

The generated audit tables below are the fixed, hand-authored fixture from `context_examples_audit.py`; they intentionally keep bin `n` visible because several legacy specs still have sparse tails. To restore broader empirical coverage with stricter support, this refresh also ran an exhaustive validation-split discovery pass over the current model-aligned cache.

- Source population was verified in ClickHouse: `game_data_filtered.participant_stats` has 16,479,150 participant rows over 1,647,915 valid games; `ml_game_player_pivot` has the same 1,647,915 games.
- Split sizes are train 1,318,331, validation 164,792, test 164,792 games. Cache metadata matches those counts exactly.
- Identity slices scanned: champion-position-build, champion-position, champion-build, champion all roles, position-build groups, and build groups.
- Axes scanned: enemy/ally physical, magic, true-damage, damage, damage-taken, heal/shield, CC, siege, scaling, ranged/melee counts, tank/frontline, burst, hard-CC, heavy-taken, high-HP, skirmish, and build-family counts.
- Support gates: champion-position-build bins needed at least 500 validation rows; broader champion slices 750-1,000; group slices 4,000-8,000. Continuous axes required all five quintile bins to clear the gate; count axes required at least four stable bins.
- Significance gate: stable-endpoint empirical effect at least 1.8 pp and z >= 3.0. The expanded pass found 973 stable trajectories and selected 160 after dedupe/axis caps; the tables below excerpt the top 70 identity rows and top 40 group rows by discovery score.
- Confirmation: the `Test effect` column is shown only when the same support and significance gate also cleared on the test split.
- Count-axis tables keep unstable tail bins visible when they exist, but `Stable emp effect`, `Stable HGNN effect`, and `Stable slope gap` use only the first and last stable bins.

The discovery pass deliberately excludes stale `classification_*` aggregate values because their `built_at`/support totals lag the current filtered population. It uses the current `identity_context_raw.npy` semantic surface, which was rebuilt from the filtered ClickHouse pipeline and is the surface the production model actually sees.

### High-Signal Identity Catalog

| Example | Axis | val n | min bin n | Stable emp effect | Stable HGNN effect | Stable slope gap | Mean abs gap | Test effect |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graves JUNGLE any build | ally damage | 20,494 | 1,889 | +13.5 pp | +9.9 pp | -3.7 pp | 1.9 pp | +11.5 pp |
| Graves all roles any build | ally damage | 20,883 | 1,926 | +13.3 pp | +9.8 pp | -3.5 pp | 1.9 pp | +11.5 pp |
| Vi JUNGLE `ad_off_tank` | enemy damage | 3,642 | 669 | -11.5 pp | -4.3 pp | +7.1 pp | 2.8 pp | N/A |
| Senna UTILITY `attack_damage` | enemy scaling | 3,773 | 666 | -10.8 pp | -3.6 pp | +7.2 pp | 4.8 pp | N/A |
| Rell UTILITY `utility_protection` | ally scaling | 4,234 | 651 | +11.0 pp | +4.1 pp | -6.9 pp | 2.9 pp | N/A |
| Irelia TOP `on_hit` | enemy true damage | 5,032 | 943 | -9.6 pp | -0.9 pp | +8.6 pp | 5.4 pp | N/A |
| Senna UTILITY `attack_damage` | enemy damage | 3,773 | 721 | -9.8 pp | -2.4 pp | +7.3 pp | 5.0 pp | N/A |
| Xerath all roles `ability_power` | ally marksman build count | 13,379 | 806 | +11.3 pp | +7.2 pp | -4.1 pp | 1.9 pp | +9.4 pp |
| Karma UTILITY `utility_protection` | enemy marksman build count | 19,215 | 791 | -10.9 pp | -5.7 pp | +5.2 pp | 1.9 pp | -10.0 pp |
| Karma all roles `utility_protection` | enemy marksman build count | 19,324 | 793 | -10.9 pp | -5.7 pp | +5.1 pp | 1.9 pp | N/A |
| Karma UTILITY any build | enemy marksman build count | 20,684 | 846 | -10.8 pp | -5.8 pp | +5.0 pp | 1.8 pp | N/A |
| Varus BOTTOM `on_hit` | enemy damage | 5,014 | 942 | -9.7 pp | -2.5 pp | +7.2 pp | 2.5 pp | -8.8 pp |
| Malphite TOP `ar_tank` | enemy physical | 7,692 | 862 | +11.7 pp | +10.2 pp | -1.5 pp | 1.6 pp | +8.0 pp |
| Sona UTILITY `utility_protection` | enemy high-HP count | 7,523 | 549 | +9.9 pp | +3.2 pp | -6.7 pp | 2.0 pp | N/A |
| Gragas TOP `ability_power` | ally damage | 3,592 | 595 | +9.8 pp | +3.7 pp | -6.1 pp | 2.4 pp | N/A |
| XinZhao all roles any build | ally ad build count | 16,503 | 1,125 | -10.7 pp | -7.1 pp | +3.6 pp | 1.3 pp | -9.6 pp |
| Graves JUNGLE any build | ally crit build count | 20,462 | 1,350 | +10.9 pp | +8.6 pp | -2.3 pp | 2.7 pp | +9.8 pp |
| Ezreal BOTTOM `attack_damage` | enemy burst count | 30,739 | 851 | -9.9 pp | -4.9 pp | +4.9 pp | 2.3 pp | N/A |
| Malphite TOP `ar_tank` | enemy magic | 7,692 | 811 | -11.4 pp | -10.4 pp | +1.0 pp | 1.0 pp | -8.1 pp |
| Ahri MIDDLE `ability_power` | enemy frontline count | 16,995 | 693 | +9.6 pp | +3.8 pp | -5.7 pp | 2.2 pp | +8.6 pp |
| Ahri MIDDLE `ability_power` | enemy tank build count | 16,995 | 693 | +9.6 pp | +3.8 pp | -5.7 pp | 2.2 pp | +8.6 pp |
| Ezreal BOTTOM any build | ally damage | 32,055 | 3,483 | +9.9 pp | +5.4 pp | -4.5 pp | 1.8 pp | +8.0 pp |
| Smolder all roles any build | enemy marksman build count | 35,869 | 1,200 | -9.9 pp | -5.0 pp | +4.8 pp | 1.4 pp | N/A |
| Jhin all roles `crit` | enemy marksman build count | 20,287 | 762 | -9.1 pp | -2.3 pp | +6.8 pp | 1.8 pp | N/A |
| Jhin BOTTOM any build | enemy marksman build count | 21,094 | 811 | -9.1 pp | -2.6 pp | +6.5 pp | 1.8 pp | N/A |
| Ezreal all roles `attack_damage` | enemy burst count | 31,534 | 872 | -9.6 pp | -4.9 pp | +4.7 pp | 2.2 pp | N/A |
| Senna all roles `attack_damage` | ally scaling | 6,542 | 922 | +8.7 pp | +1.6 pp | -7.1 pp | 3.1 pp | +4.3 pp |
| Graves JUNGLE `crit` | ally ad build count | 16,639 | 511 | -9.6 pp | -14.3 pp | -4.7 pp | 2.8 pp | N/A |
| Shyvana JUNGLE any build | enemy scaling | 6,333 | 1,157 | -9.2 pp | -3.6 pp | +5.6 pp | 3.3 pp | N/A |
| Graves JUNGLE any build | ally scaling | 20,494 | 1,850 | +10.2 pp | +7.8 pp | -2.4 pp | 2.0 pp | +9.6 pp |
| Graves all roles any build | ally scaling | 20,883 | 1,882 | +10.2 pp | +7.7 pp | -2.5 pp | 2.0 pp | +9.5 pp |
| Jhin BOTTOM `crit` | enemy marksman build count | 20,006 | 753 | -8.9 pp | -2.3 pp | +6.6 pp | 1.7 pp | N/A |
| Qiyana JUNGLE `lethality` | enemy damage | 3,513 | 664 | -8.8 pp | -3.1 pp | +5.7 pp | 4.8 pp | N/A |
| Graves all roles any build | ally crit build count | 20,850 | 1,367 | +10.3 pp | +8.3 pp | -1.9 pp | 2.7 pp | +9.8 pp |
| Nami UTILITY `utility_enchanter` | ally damage | 10,746 | 1,045 | +9.4 pp | +4.9 pp | -4.6 pp | 2.5 pp | +9.8 pp |
| Nami all roles `utility_enchanter` | ally damage | 10,746 | 1,045 | +9.4 pp | +4.9 pp | -4.6 pp | 2.5 pp | N/A |
| Malphite all roles `ar_tank` | enemy physical | 8,661 | 948 | +10.9 pp | +10.2 pp | -0.7 pp | 1.3 pp | +8.6 pp |
| Ezreal all roles any build | ally damage | 33,515 | 3,567 | +9.4 pp | +5.2 pp | -4.2 pp | 1.7 pp | +7.8 pp |
| Nidalee JUNGLE any build | ally siege | 6,467 | 950 | +9.2 pp | +4.8 pp | -4.5 pp | 2.4 pp | N/A |
| Ornn TOP `ar_tank` | enemy ranged count | 6,668 | 518 | -8.7 pp | -3.0 pp | +5.7 pp | 2.9 pp | N/A |
| Ornn TOP `ar_tank` | enemy melee count | 6,668 | 518 | +8.7 pp | +3.0 pp | -5.7 pp | 2.9 pp | N/A |
| Varus BOTTOM `on_hit` | enemy scaling | 5,014 | 935 | -8.7 pp | -2.8 pp | +5.9 pp | 1.9 pp | -9.2 pp |
| XinZhao JUNGLE any build | ally ad build count | 11,927 | 813 | -9.4 pp | -5.5 pp | +3.9 pp | 2.0 pp | N/A |
| Nocturne JUNGLE any build | ally scaling | 9,451 | 852 | +8.1 pp | +0.8 pp | -7.3 pp | 2.0 pp | N/A |
| Soraka UTILITY `utility_protection` | ally scaling | 8,977 | 671 | +8.8 pp | +3.6 pp | -5.1 pp | 2.3 pp | +7.4 pp |
| Nidalee JUNGLE `ability_power` | ally siege | 6,403 | 938 | +9.0 pp | +4.8 pp | -4.2 pp | 2.3 pp | N/A |
| Senna all roles `attack_damage` | ally siege | 6,542 | 1,014 | +8.0 pp | +1.3 pp | -6.7 pp | 3.1 pp | +5.0 pp |
| Anivia MIDDLE `ability_power` | enemy true damage | 5,235 | 1,011 | -8.0 pp | -1.5 pp | +6.6 pp | 2.8 pp | N/A |
| Renekton TOP `attack_damage` | enemy scaling | 5,278 | 989 | -7.9 pp | -1.0 pp | +6.9 pp | 3.3 pp | -7.2 pp |
| Sylas JUNGLE any build | enemy damage | 8,544 | 1,607 | -8.2 pp | -2.2 pp | +6.1 pp | 2.1 pp | -6.2 pp |
| Fizz MIDDLE `ability_power` | ally scaling | 5,689 | 752 | +8.7 pp | +3.7 pp | -5.0 pp | 1.4 pp | +6.8 pp |
| JarvanIV JUNGLE `attack_damage` | enemy scaling | 5,788 | 1,076 | -7.7 pp | -0.8 pp | +6.9 pp | 4.0 pp | N/A |
| Ezreal BOTTOM any build | enemy burst count | 32,031 | 875 | -8.7 pp | -4.4 pp | +4.3 pp | 2.1 pp | N/A |
| Smolder all roles `ad_off_tank` | enemy scaling | 11,203 | 1,974 | -8.6 pp | -4.2 pp | +4.4 pp | 2.0 pp | N/A |
| Smolder all roles any build | enemy burst count | 36,283 | 1,045 | -8.7 pp | -4.3 pp | +4.4 pp | 1.4 pp | N/A |
| Malphite all roles `ar_tank` | enemy magic | 8,661 | 899 | -10.1 pp | -10.3 pp | -0.2 pp | 1.0 pp | -8.9 pp |
| Shaco JUNGLE any build | ally damage | 6,496 | 1,215 | +8.0 pp | +2.9 pp | -5.1 pp | 3.8 pp | N/A |
| Ekko MIDDLE `ability_power` | enemy damage | 5,069 | 937 | -8.1 pp | -2.6 pp | +5.5 pp | 1.9 pp | N/A |
| Senna BOTTOM any build | ally scaling | 7,515 | 931 | +7.8 pp | +2.2 pp | -5.6 pp | 3.1 pp | +6.6 pp |
| Nami UTILITY `utility_protection` | ally marksman build count | 16,849 | 848 | +7.6 pp | +0.9 pp | -6.7 pp | 1.7 pp | N/A |
| Nami all roles `utility_protection` | ally marksman build count | 16,849 | 848 | +7.6 pp | +0.9 pp | -6.7 pp | 1.7 pp | N/A |
| Irelia all roles `on_hit` | enemy true damage | 8,086 | 1,569 | -7.3 pp | -1.1 pp | +6.2 pp | 4.8 pp | N/A |
| Smolder BOTTOM any build | ally marksman build count | 33,108 | 3,155 | +9.3 pp | +7.8 pp | -1.6 pp | 0.8 pp | +9.6 pp |
| Smolder BOTTOM any build | enemy burst count | 33,374 | 972 | -8.5 pp | -4.3 pp | +4.1 pp | 1.3 pp | N/A |
| Nidalee all roles `ability_power` | ally siege | 6,693 | 963 | +8.4 pp | +4.4 pp | -4.0 pp | 2.4 pp | N/A |
| Sylas all roles `ap_off_tank` | enemy siege | 6,643 | 1,111 | -9.3 pp | -8.0 pp | +1.3 pp | 1.9 pp | -6.7 pp |
| Sylas all roles `ap_off_tank` | enemy scaling | 6,643 | 1,012 | -9.4 pp | -8.7 pp | +0.7 pp | 2.0 pp | -6.0 pp |
| Hwei MIDDLE `ability_power` | enemy damage | 5,874 | 1,065 | -8.3 pp | -4.2 pp | +4.2 pp | 1.3 pp | N/A |
| Smolder BOTTOM `ad_off_tank` | enemy siege | 10,235 | 1,979 | -8.3 pp | -4.6 pp | +3.7 pp | 1.7 pp | N/A |
| Sylas JUNGLE `ability_power` | enemy damage | 6,588 | 1,214 | -7.5 pp | -1.7 pp | +5.9 pp | 1.9 pp | -7.1 pp |

### High-Signal Group Catalog

| Example | Axis | val n | min bin n | Stable emp effect | Stable HGNN effect | Stable slope gap | Mean abs gap | Test effect |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_damage all roles group | enemy marksman build count | 212,774 | 8,519 | -7.3 pp | -4.3 pp | +2.9 pp | 1.0 pp | N/A |
| lethality JUNGLE group | ally scaling | 62,511 | 7,119 | +7.3 pp | +6.2 pp | -1.1 pp | 0.7 pp | +6.4 pp |
| ability_power MIDDLE group | enemy burst count | 224,155 | 6,706 | -6.6 pp | -3.8 pp | +2.8 pp | 1.1 pp | -5.2 pp |
| ability_power MIDDLE group | ally marksman build count | 223,536 | 9,268 | +6.7 pp | +5.6 pp | -1.2 pp | 0.9 pp | +7.8 pp |
| ability_power all roles group | enemy burst count | 424,236 | 12,854 | -6.2 pp | -3.9 pp | +2.3 pp | 1.1 pp | -5.1 pp |
| lethality all roles group | ally scaling | 110,456 | 12,869 | +6.6 pp | +5.4 pp | -1.2 pp | 0.5 pp | +5.7 pp |
| utility_protection UTILITY group | ally marksman build count | 161,611 | 6,053 | +5.8 pp | +3.1 pp | -2.6 pp | 0.9 pp | +7.6 pp |
| attack_damage TOP group | ally scaling | 63,087 | 7,286 | +4.9 pp | +0.3 pp | -4.6 pp | 1.3 pp | +3.0 pp |
| ability_power all roles group | enemy marksman build count | 419,945 | 16,326 | -5.3 pp | -2.9 pp | +2.4 pp | 0.9 pp | -6.3 pp |
| ap_off_tank all roles group | enemy damage | 51,029 | 9,410 | -5.9 pp | -4.9 pp | +0.9 pp | 0.7 pp | -5.7 pp |
| ap_off_tank all roles group | enemy scaling | 51,029 | 8,793 | -6.0 pp | -6.3 pp | -0.3 pp | 0.7 pp | -5.2 pp |
| ad_off_tank TOP group | ally siege | 54,887 | 7,828 | +5.0 pp | +2.1 pp | -3.0 pp | 0.9 pp | +4.7 pp |
| utility_protection UTILITY group | ally damage | 162,573 | 11,251 | +5.3 pp | +3.8 pp | -1.5 pp | 0.4 pp | +5.7 pp |
| ability_power MIDDLE group | ally tank build count | 224,183 | 6,588 | -5.3 pp | -3.7 pp | +1.6 pp | 0.8 pp | -3.7 pp |
| ability_power MIDDLE group | enemy marksman build count | 221,962 | 8,526 | -4.9 pp | -2.5 pp | +2.4 pp | 0.8 pp | -6.6 pp |
| attack_damage BOTTOM group | ally damage | 53,886 | 7,591 | +4.9 pp | +2.5 pp | -2.4 pp | 0.6 pp | +3.3 pp |
| ad_off_tank JUNGLE group | enemy damage | 49,793 | 9,199 | -4.9 pp | -2.7 pp | +2.1 pp | 0.9 pp | -5.2 pp |
| lethality all roles group | ally damage | 110,456 | 14,986 | +5.5 pp | +5.2 pp | -0.3 pp | 0.3 pp | +5.2 pp |
| ability_power all roles group | ally marksman build count | 423,058 | 23,347 | +5.5 pp | +5.6 pp | +0.1 pp | 0.8 pp | +6.8 pp |
| utility_protection all roles group | ally damage | 165,859 | 11,304 | +5.1 pp | +3.7 pp | -1.3 pp | 0.4 pp | +5.5 pp |
| utility_protection UTILITY group | enemy marksman build count | 160,972 | 6,729 | -4.9 pp | -3.1 pp | +1.8 pp | 0.6 pp | -6.1 pp |
| attack_damage all roles group | enemy frontline count | 214,608 | 8,634 | +5.5 pp | +5.5 pp | +0.0 pp | 0.4 pp | +5.1 pp |
| attack_damage all roles group | enemy tank build count | 214,608 | 8,634 | +5.5 pp | +5.5 pp | +0.0 pp | 0.4 pp | +5.1 pp |
| utility_protection UTILITY group | enemy burst count | 162,436 | 4,532 | -5.0 pp | -3.3 pp | +1.6 pp | 0.7 pp | N/A |
| utility_protection UTILITY group | ally scaling | 162,573 | 20,288 | +4.8 pp | +3.2 pp | -1.7 pp | 0.5 pp | +4.7 pp |
| lethality JUNGLE group | ally damage | 62,511 | 8,559 | +5.4 pp | +5.4 pp | -0.0 pp | 0.2 pp | +5.0 pp |
| ability_power all roles group | ally tank build count | 424,357 | 9,658 | -5.0 pp | -4.0 pp | +1.0 pp | 0.8 pp | -4.4 pp |
| on_hit BOTTOM group | enemy damage | 30,999 | 5,799 | -4.6 pp | -2.4 pp | +2.2 pp | 0.5 pp | -3.8 pp |
| attack_damage BOTTOM group | ally true damage | 53,886 | 9,622 | +4.5 pp | +2.2 pp | -2.3 pp | 1.0 pp | N/A |
| crit all roles group | ally ad build count | 295,904 | 12,799 | -3.4 pp | -8.2 pp | -4.8 pp | 1.6 pp | -3.8 pp |
| utility_protection all roles group | ally scaling | 165,859 | 20,314 | +4.6 pp | +3.1 pp | -1.5 pp | 0.5 pp | +4.5 pp |
| crit all roles group | enemy burst count | 296,407 | 8,869 | -4.9 pp | -4.3 pp | +0.6 pp | 0.4 pp | -4.3 pp |
| ability_power BOTTOM group | enemy scaling | 41,354 | 7,648 | -4.8 pp | -3.7 pp | +1.1 pp | 0.4 pp | -3.7 pp |
| ar_tank UTILITY group | enemy cc | 47,837 | 9,116 | -4.2 pp | -6.6 pp | -2.4 pp | 1.0 pp | -2.9 pp |
| on_hit all roles group | ally scaling | 71,136 | 12,081 | +5.0 pp | +5.0 pp | -0.0 pp | 0.4 pp | +5.4 pp |
| ability_power all roles group | enemy frontline count | 423,602 | 16,160 | +4.8 pp | +4.6 pp | -0.3 pp | 0.7 pp | +5.7 pp |
| ability_power all roles group | enemy tank build count | 423,602 | 16,160 | +4.8 pp | +4.6 pp | -0.3 pp | 0.7 pp | +5.7 pp |
| utility_protection UTILITY group | enemy frontline count | 162,144 | 6,521 | +4.7 pp | +3.7 pp | -1.0 pp | 0.4 pp | +4.7 pp |
| utility_protection UTILITY group | enemy tank build count | 162,144 | 6,521 | +4.7 pp | +3.7 pp | -1.0 pp | 0.4 pp | +4.7 pp |
| utility_enchanter UTILITY group | enemy siege | 27,748 | 5,069 | -4.3 pp | -2.4 pp | +1.9 pp | 1.0 pp | -3.3 pp |

## Detailed Discovery Tables

### Malphite TOP `ar_tank` vs enemy physical

Validation support: n=7,692, min stable bin n=862. Stable empirical effect +11.7 pp; stable HGNN effect +10.2 pp; stable slope gap -1.5 pp. Test confirmation effect +8.0 pp with min bin n=883.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.387` | 862 | 43.6% | 45.4% | +1.8 pp | yes |
| `20-40% 0.387-0.448` | 1,008 | 50.3% | 46.3% | -4.0 pp | yes |
| `40-60% 0.448-0.508` | 1,551 | 49.1% | 49.9% | +0.8 pp | yes |
| `60-80% 0.508-0.557` | 1,952 | 52.6% | 51.6% | -1.0 pp | yes |
| `80-100% >= 0.557` | 2,319 | 55.4% | 55.6% | +0.3 pp | yes |

### Vi JUNGLE `ad_off_tank` vs enemy damage

Validation support: n=3,642, min stable bin n=669. Stable empirical effect -11.5 pp; stable HGNN effect -4.3 pp; stable slope gap +7.1 pp. Test confirmation did not clear the same support/significance gate.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.739` | 740 | 54.9% | 49.7% | -5.1 pp | yes |
| `20-40% 0.739-0.764` | 778 | 49.1% | 46.9% | -2.2 pp | yes |
| `40-60% 0.764-0.785` | 669 | 44.7% | 46.3% | +1.6 pp | yes |
| `60-80% 0.785-0.813` | 734 | 48.8% | 45.9% | -2.9 pp | yes |
| `80-100% >= 0.813` | 721 | 43.4% | 45.4% | +2.0 pp | yes |

### Senna UTILITY `attack_damage` vs enemy scaling

Validation support: n=3,773, min stable bin n=666. Stable empirical effect -10.8 pp; stable HGNN effect -3.6 pp; stable slope gap +7.2 pp. Test confirmation did not clear the same support/significance gate.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.829` | 810 | 61.9% | 53.3% | -8.5 pp | yes |
| `20-40% 0.829-0.841` | 751 | 57.0% | 50.8% | -6.2 pp | yes |
| `40-60% 0.841-0.852` | 814 | 54.3% | 49.2% | -5.1 pp | yes |
| `60-80% 0.852-0.863` | 666 | 52.0% | 49.0% | -3.0 pp | yes |
| `80-100% >= 0.863` | 732 | 51.1% | 49.7% | -1.4 pp | yes |

### Irelia TOP `on_hit` vs enemy true damage

Validation support: n=5,032, min stable bin n=943. Stable empirical effect -9.6 pp; stable HGNN effect -0.9 pp; stable slope gap +8.6 pp. Test confirmation did not clear the same support/significance gate.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.045` | 1,072 | 54.2% | 55.1% | +0.9 pp | yes |
| `20-40% 0.045-0.056` | 1,045 | 50.2% | 55.1% | +4.8 pp | yes |
| `40-60% 0.056-0.072` | 1,015 | 50.8% | 55.5% | +4.6 pp | yes |
| `60-80% 0.072-0.092` | 957 | 47.2% | 54.6% | +7.3 pp | yes |
| `80-100% >= 0.092` | 943 | 44.6% | 54.2% | +9.5 pp | yes |

### Karma UTILITY `utility_protection` vs enemy marksman build count

Validation support: n=19,215, min stable bin n=791. Stable empirical effect -10.9 pp; stable HGNN effect -5.7 pp; stable slope gap +5.2 pp. Test confirmation effect -10.0 pp with min bin n=548.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 791 | 57.3% | 54.6% | -2.6 pp | yes |
| `1` | 6,820 | 51.4% | 52.0% | +0.6 pp | yes |
| `2` | 8,642 | 47.6% | 49.4% | +1.8 pp | yes |
| `3` | 2,962 | 46.4% | 48.9% | +2.5 pp | yes |
| `4` | 178 | 50.0% | 52.5% | +2.5 pp | no |
| `5` | 3 | 66.7% | 58.9% | -7.8 pp | no |

### Varus BOTTOM `on_hit` vs enemy damage

Validation support: n=5,014, min stable bin n=942. Stable empirical effect -9.7 pp; stable HGNN effect -2.5 pp; stable slope gap +7.2 pp. Test confirmation effect -8.8 pp with min bin n=1,591.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.739` | 1,038 | 55.1% | 50.2% | -4.9 pp | yes |
| `20-40% 0.739-0.764` | 1,011 | 50.5% | 48.3% | -2.2 pp | yes |
| `40-60% 0.764-0.785` | 942 | 46.6% | 47.9% | +1.3 pp | yes |
| `60-80% 0.785-0.813` | 1,003 | 48.5% | 46.8% | -1.7 pp | yes |
| `80-100% >= 0.813` | 1,020 | 45.4% | 47.7% | +2.3 pp | yes |

### Ezreal BOTTOM `attack_damage` vs enemy burst count

Validation support: n=30,739, min stable bin n=851. Stable empirical effect -9.9 pp; stable HGNN effect -4.9 pp; stable slope gap +4.9 pp. Test confirmation did not clear the same support/significance gate.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 9,895 | 49.6% | 50.4% | +0.8 pp | yes |
| `1` | 14,112 | 47.6% | 48.4% | +0.9 pp | yes |
| `2` | 5,881 | 45.7% | 47.3% | +1.7 pp | yes |
| `3` | 851 | 39.7% | 45.5% | +5.7 pp | yes |
| `4` | 22 | 18.2% | 38.2% | +20.0 pp | no |

### Ahri MIDDLE `ability_power` vs enemy frontline count

Validation support: n=16,995, min stable bin n=693. Stable empirical effect +9.6 pp; stable HGNN effect +3.8 pp; stable slope gap -5.7 pp. Test confirmation effect +8.6 pp with min bin n=584.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 4,940 | 47.7% | 50.7% | +3.0 pp | yes |
| `1` | 7,700 | 50.2% | 51.5% | +1.3 pp | yes |
| `2` | 3,662 | 50.8% | 52.6% | +1.8 pp | yes |
| `3` | 693 | 57.3% | 54.5% | -2.8 pp | yes |
| `4` | 35 | 60.0% | 55.8% | -4.2 pp | no |
| `5` | 3 | 33.3% | 58.6% | +25.2 pp | no |

### Ornn TOP `ar_tank` vs enemy ranged count

Validation support: n=6,668, min stable bin n=518. Stable empirical effect -8.7 pp; stable HGNN effect -3.0 pp; stable slope gap +5.7 pp. Test confirmation did not clear the same support/significance gate.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 12 | 50.0% | 47.0% | -3.0 pp | no |
| `1` | 518 | 59.5% | 57.0% | -2.4 pp | yes |
| `2` | 2,454 | 51.4% | 54.6% | +3.2 pp | yes |
| `3` | 2,912 | 51.0% | 53.7% | +2.7 pp | yes |
| `4` | 784 | 50.8% | 54.0% | +3.3 pp | yes |
| `5` | 46 | 54.3% | 54.7% | +0.4 pp | no |

### Nami UTILITY `utility_enchanter` vs ally damage

Validation support: n=10,746, min stable bin n=1,045. Stable empirical effect +9.4 pp; stable HGNN effect +4.9 pp; stable slope gap -4.6 pp. Test confirmation effect +9.8 pp with min bin n=516.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.739` | 2,020 | 47.3% | 46.6% | -0.8 pp | yes |
| `20-40% 0.739-0.764` | 2,527 | 48.0% | 48.9% | +0.8 pp | yes |
| `40-60% 0.764-0.785` | 2,594 | 52.6% | 49.6% | -2.9 pp | yes |
| `60-80% 0.785-0.813` | 2,560 | 52.7% | 50.2% | -2.5 pp | yes |
| `80-100% >= 0.813` | 1,045 | 56.7% | 51.4% | -5.3 pp | yes |

### Renekton TOP `attack_damage` vs enemy scaling

Validation support: n=5,278, min stable bin n=989. Stable empirical effect -7.9 pp; stable HGNN effect -1.0 pp; stable slope gap +6.9 pp. Test confirmation effect -7.2 pp with min bin n=891.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.829` | 990 | 51.6% | 48.6% | -3.0 pp | yes |
| `20-40% 0.829-0.841` | 989 | 49.8% | 46.3% | -3.5 pp | yes |
| `40-60% 0.841-0.852` | 1,091 | 50.0% | 46.9% | -3.0 pp | yes |
| `60-80% 0.852-0.863` | 1,038 | 44.4% | 47.4% | +3.0 pp | yes |
| `80-100% >= 0.863` | 1,170 | 43.8% | 47.6% | +3.8 pp | yes |

### Sylas JUNGLE any build vs enemy damage

Validation support: n=8,544, min stable bin n=1,607. Stable empirical effect -8.2 pp; stable HGNN effect -2.2 pp; stable slope gap +6.1 pp. Test confirmation effect -6.2 pp with min bin n=1,537.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.739` | 1,694 | 53.8% | 53.3% | -0.5 pp | yes |
| `20-40% 0.739-0.764` | 1,664 | 49.7% | 51.1% | +1.4 pp | yes |
| `40-60% 0.764-0.785` | 1,607 | 50.5% | 51.3% | +0.8 pp | yes |
| `60-80% 0.785-0.813` | 1,774 | 48.1% | 50.5% | +2.3 pp | yes |
| `80-100% >= 0.813` | 1,805 | 45.6% | 51.2% | +5.6 pp | yes |

### Graves JUNGLE any build vs ally damage

Validation support: n=20,494, min stable bin n=1,889. Stable empirical effect +13.5 pp; stable HGNN effect +9.9 pp; stable slope gap -3.7 pp. Test confirmation effect +11.5 pp with min bin n=1,948.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.739` | 1,889 | 38.4% | 38.4% | -0.1 pp | yes |
| `20-40% 0.739-0.764` | 2,391 | 49.1% | 47.8% | -1.4 pp | yes |
| `40-60% 0.764-0.785` | 3,532 | 51.9% | 49.8% | -2.1 pp | yes |
| `60-80% 0.785-0.813` | 5,390 | 51.7% | 49.6% | -2.1 pp | yes |
| `80-100% >= 0.813` | 7,292 | 52.0% | 48.3% | -3.7 pp | yes |

### Jhin BOTTOM `crit` vs enemy marksman build count

Validation support: n=20,006, min stable bin n=753. Stable empirical effect -8.9 pp; stable HGNN effect -2.3 pp; stable slope gap +6.6 pp. Test confirmation did not clear the same support/significance gate.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 753 | 53.4% | 49.9% | -3.4 pp | yes |
| `1` | 6,665 | 48.5% | 48.4% | -0.1 pp | yes |
| `2` | 9,109 | 47.2% | 47.0% | -0.2 pp | yes |
| `3` | 3,479 | 44.4% | 47.6% | +3.2 pp | yes |
| `4` | 236 | 45.8% | 51.5% | +5.7 pp | no |
| `5` | 3 | 66.7% | 54.8% | -11.9 pp | no |

### Graves all roles any build vs ally damage

Validation support: n=20,883, min stable bin n=1,926. Stable empirical effect +13.3 pp; stable HGNN effect +9.8 pp; stable slope gap -3.5 pp. Test confirmation effect +11.5 pp with min bin n=1,999.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.739` | 1,926 | 38.5% | 38.4% | -0.1 pp | yes |
| `20-40% 0.739-0.764` | 2,444 | 49.0% | 47.7% | -1.3 pp | yes |
| `40-60% 0.764-0.785` | 3,616 | 51.8% | 49.8% | -2.0 pp | yes |
| `60-80% 0.785-0.813` | 5,485 | 51.8% | 49.5% | -2.2 pp | yes |
| `80-100% >= 0.813` | 7,412 | 51.8% | 48.2% | -3.7 pp | yes |

### Rell UTILITY `utility_protection` vs ally scaling

Validation support: n=4,234, min stable bin n=651. Stable empirical effect +11.0 pp; stable HGNN effect +4.1 pp; stable slope gap -6.9 pp. Test confirmation did not clear the same support/significance gate.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.829` | 914 | 46.6% | 48.4% | +1.8 pp | yes |
| `20-40% 0.829-0.841` | 894 | 51.8% | 50.0% | -1.8 pp | yes |
| `40-60% 0.841-0.852` | 964 | 55.3% | 51.4% | -3.9 pp | yes |
| `60-80% 0.852-0.863` | 811 | 53.9% | 51.9% | -1.9 pp | yes |
| `80-100% >= 0.863` | 651 | 57.6% | 52.5% | -5.1 pp | yes |

### Senna UTILITY `attack_damage` vs enemy damage

Validation support: n=3,773, min stable bin n=721. Stable empirical effect -9.8 pp; stable HGNN effect -2.4 pp; stable slope gap +7.3 pp. Test confirmation did not clear the same support/significance gate.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.739` | 758 | 61.5% | 52.9% | -8.6 pp | yes |
| `20-40% 0.739-0.764` | 732 | 56.8% | 49.9% | -7.0 pp | yes |
| `40-60% 0.764-0.785` | 721 | 55.1% | 49.5% | -5.5 pp | yes |
| `60-80% 0.785-0.813` | 790 | 52.3% | 49.6% | -2.7 pp | yes |
| `80-100% >= 0.813` | 772 | 51.7% | 50.5% | -1.2 pp | yes |

### Xerath all roles `ability_power` vs ally marksman build count

Validation support: n=13,379, min stable bin n=806. Stable empirical effect +11.3 pp; stable HGNN effect +7.2 pp; stable slope gap -4.1 pp. Test confirmation effect +9.4 pp with min bin n=789.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 806 | 40.8% | 45.6% | +4.8 pp | yes |
| `1` | 5,084 | 47.4% | 47.9% | +0.5 pp | yes |
| `2` | 5,674 | 49.3% | 50.8% | +1.5 pp | yes |
| `3` | 1,815 | 52.1% | 52.8% | +0.7 pp | yes |
| `4` | 75 | 45.3% | 51.5% | +6.2 pp | no |

## Enemy Count Tail Shrinkage

| Audit | Axis | Baseline bin | Tail bin | Empirical tail effect | HGNN tail effect | Shrinkage |
|---|---|---:|---:|---:|---:|---:|
| Sylas MIDDLE `ability_power` vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -5.65 pp | -3.37 pp | 0.60x |
| Nilah BOTTOM any build vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | -13.91 pp | -6.23 pp | 0.45x |
| Kaisa BOTTOM any build vs enemy range | `enemy_ranged_count` | `<= 1` | `>= 4` | +1.82 pp | -2.63 pp | -1.44x |
| Kaisa BOTTOM `on_hit` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +3.02 pp | +4.37 pp | 1.44x |
| Ahri MIDDLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +9.61 pp | +3.91 pp | 0.41x |
| Sylas JUNGLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +4.71 pp | +4.69 pp | 1.00x |
| Sylas MIDDLE `ability_power` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +4.52 pp | +3.73 pp | 0.83x |
| Karma UTILITY any build vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +7.52 pp | +5.75 pp | 0.77x |
| Vayne BOTTOM `on_hit` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +10.75 pp | +9.10 pp | 0.85x |
| Thresh UTILITY `ar_tank` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -2.72 pp | -3.50 pp | 1.29x |
| Nautilus UTILITY `mr_tank` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -2.79 pp | -0.83 pp | 0.30x |
| Zed MIDDLE `lethality` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -0.86 pp | -2.30 pp | 2.69x |
| Nami UTILITY `utility_protection` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -3.36 pp | -4.38 pp | 1.30x |
| Jinx BOTTOM `crit` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -5.94 pp | -5.70 pp | 0.96x |
| Malphite TOP `ar_tank` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -2.97 pp | -10.45 pp | 3.52x |
| Viego JUNGLE any build vs enemy high-HP count | `enemy_high_hp_count` | `0` | `>= 3` | +2.84 pp | +3.03 pp | 1.06x |
| Darius TOP any build vs enemy range count | `enemy_ranged_count` | `<= 1` | `>= 4` | -2.91 pp | -5.20 pp | 1.79x |
| MasterYi JUNGLE any build vs enemy hard CC | `enemy_hard_cc_count` | `0` | `>= 3` | -0.33 pp | -3.75 pp | 11.22x |
| Focus HP `<= 2309` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -5.67 pp | -4.31 pp | 0.76x |
| Focus HP `>= 2478` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -5.96 pp | -4.37 pp | 0.73x |
| Ahri MIDDLE `ability_power` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -2.79 pp | -1.83 pp | 0.66x |
| Kaisa BOTTOM `on_hit` vs heavy damage-taken count | `enemy_heavy_taken_count` | `0` | `>= 3` | -5.50 pp | -1.54 pp | 0.28x |
| Ezreal BOTTOM `attack_damage` vs enemy hard CC | `enemy_hard_cc_count` | `0` | `>= 3` | -3.60 pp | -1.23 pp | 0.34x |
| Jayce TOP `attack_damage` vs enemy frontline count | `enemy_frontline_count` | `0` | `>= 3` | +7.46 pp | +4.82 pp | 0.65x |
| Caitlyn BOTTOM `crit` vs enemy burst count | `enemy_burst_count` | `0` | `>= 3` | -4.99 pp | -2.12 pp | 0.42x |

## Headline Trajectory Audit Tables

### Yasuo TOP `crit` vs enemy siege

Melee crit carry punished by poke and siege.

**Gap MSE** 0.51 pp^2 | **Mean abs gap** 0.63 pp | **Accuracy** 57.20% | **Accuracy if calibrated** 56.66% | **Calibration lift** -0.54 pp | **Empirical effect** -0.93 pp | **HGNN effect** -1.64 pp | **Shrinkage** 1.77x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.441` | 1,011 | 52.42% | 53.63% | +1.21 pp | 59.15% |
| `0.441-0.471` | 872 | 53.44% | 52.76% | -0.68 pp | 56.65% |
| `0.471-0.499` | 855 | 52.51% | 53.10% | +0.58 pp | 57.08% |
| `0.499-0.530` | 785 | 52.74% | 52.55% | -0.19 pp | 57.96% |
| `>= 0.530` | 736 | 51.49% | 51.99% | +0.50 pp | 54.48% |

### Graves JUNGLE `lethality` vs enemy damage

Burst jungler into high enemy damage.

**Gap MSE** 5.73 pp^2 | **Mean abs gap** 1.67 pp | **Accuracy** 71.94% | **Accuracy if calibrated** 71.79% | **Calibration lift** -0.15 pp | **Empirical effect** -13.78 pp | **HGNN effect** -9.97 pp | **Shrinkage** 0.72x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 486 | 40.12% | 35.07% | -5.05 pp | 67.90% |
| `0.739-0.764` | 428 | 28.50% | 28.86% | +0.36 pp | 72.20% |
| `0.764-0.785` | 400 | 28.00% | 26.97% | -1.03 pp | 72.00% |
| `0.785-0.813` | 387 | 25.84% | 25.16% | -0.68 pp | 75.19% |
| `>= 0.813` | 334 | 26.35% | 25.11% | -1.24 pp | 73.65% |

### Yasuo MIDDLE `crit` vs enemy siege

Same melee-carry-into-poke pattern across lane.

**Gap MSE** 1.60 pp^2 | **Mean abs gap** 1.05 pp | **Accuracy** 57.71% | **Accuracy if calibrated** 57.69% | **Calibration lift** -0.02 pp | **Empirical effect** -1.22 pp | **HGNN effect** +0.01 pp | **Shrinkage** -0.01x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.441` | 2,230 | 52.06% | 50.40% | -1.66 pp | 60.72% |
| `0.441-0.471` | 1,957 | 48.90% | 49.65% | +0.75 pp | 57.38% |
| `0.471-0.499` | 1,871 | 48.53% | 50.63% | +2.10 pp | 57.14% |
| `0.499-0.530` | 1,817 | 50.52% | 50.23% | -0.29 pp | 57.02% |
| `>= 0.530` | 1,719 | 50.84% | 50.41% | -0.43 pp | 55.56% |

### Ahri MIDDLE `ability_power` vs enemy scaling

AP mid into scaling enemy compositions.

**Gap MSE** 6.16 pp^2 | **Mean abs gap** 2.04 pp | **Accuracy** 57.22% | **Accuracy if calibrated** 57.81% | **Calibration lift** +0.58 pp | **Empirical effect** -5.04 pp | **HGNN effect** -0.68 pp | **Shrinkage** 0.14x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.829` | 3,587 | 53.30% | 52.46% | -0.85 pp | 58.38% |
| `0.829-0.841` | 3,300 | 50.88% | 51.35% | +0.47 pp | 58.82% |
| `0.841-0.852` | 3,539 | 47.33% | 51.27% | +3.94 pp | 55.72% |
| `0.852-0.863` | 3,122 | 49.74% | 51.16% | +1.41 pp | 56.66% |
| `>= 0.863` | 3,485 | 48.26% | 51.78% | +3.51 pp | 56.56% |

### Nautilus UTILITY `mr_tank` with ally damage

Engage support with damage behind it.

**Gap MSE** 3.18 pp^2 | **Mean abs gap** 1.43 pp | **Accuracy** 58.24% | **Accuracy if calibrated** 58.42% | **Calibration lift** +0.19 pp | **Empirical effect** +12.62 pp | **HGNN effect** +9.87 pp | **Shrinkage** 0.78x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 625 | 42.24% | 44.41% | +2.17 pp | 63.20% |
| `0.739-0.764` | 758 | 47.89% | 47.28% | -0.61 pp | 58.44% |
| `0.764-0.785` | 789 | 51.20% | 48.03% | -3.18 pp | 56.65% |
| `0.785-0.813` | 763 | 50.72% | 50.09% | -0.63 pp | 55.57% |
| `>= 0.813` | 288 | 54.86% | 54.27% | -0.59 pp | 58.33% |

### Galio MIDDLE `mr_tank` vs enemy magic

Anti-magic tank itemization (kept off-list MR-tank).

**Gap MSE** 25.75 pp^2 | **Mean abs gap** 4.36 pp | **Accuracy** 60.53% | **Accuracy if calibrated** 60.40% | **Calibration lift** -0.13 pp | **Empirical effect** +13.02 pp | **HGNN effect** +5.26 pp | **Shrinkage** 0.40x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 172 | 33.72% | 42.65% | +8.93 pp | 67.44% |
| `0.373-0.423` | 210 | 40.95% | 43.73% | +2.78 pp | 61.90% |
| `0.423-0.486` | 266 | 39.85% | 44.48% | +4.63 pp | 57.89% |
| `0.486-0.549` | 353 | 48.73% | 44.42% | -4.30 pp | 58.07% |
| `>= 0.549` | 537 | 46.74% | 47.90% | +1.16 pp | 60.71% |

### Malphite TOP `ar_tank` vs enemy physical

Armor tank into AD-heavy enemies.

**Gap MSE** 4.21 pp^2 | **Mean abs gap** 1.56 pp | **Accuracy** 57.12% | **Accuracy if calibrated** 57.71% | **Calibration lift** +0.59 pp | **Empirical effect** +11.75 pp | **HGNN effect** +10.24 pp | **Shrinkage** 0.87x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 862 | 43.62% | 45.39% | +1.77 pp | 60.09% |
| `0.387-0.448` | 1,008 | 50.30% | 46.26% | -4.04 pp | 56.55% |
| `0.448-0.508` | 1,551 | 49.07% | 49.85% | +0.79 pp | 56.48% |
| `0.508-0.557` | 1,952 | 52.56% | 51.60% | -0.97 pp | 56.61% |
| `>= 0.557` | 2,319 | 55.37% | 55.63% | +0.26 pp | 57.14% |

### Sylas MIDDLE `ability_power` vs enemy range

Short-range AP battlemage into enemy range pressure.

**Gap MSE** 7.42 pp^2 | **Mean abs gap** 2.58 pp | **Accuracy** 56.55% | **Accuracy if calibrated** 56.36% | **Calibration lift** -0.18 pp | **Empirical effect** -5.65 pp | **HGNN effect** -3.37 pp | **Shrinkage** 0.60x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 624 | 54.81% | 56.26% | +1.46 pp | 58.17% |
| `2` | 2,798 | 52.14% | 54.25% | +2.10 pp | 57.79% |
| `3` | 3,271 | 50.44% | 53.47% | +3.03 pp | 55.33% |
| `>= 4` | 1,007 | 49.16% | 52.89% | +3.74 pp | 56.01% |

### Nilah BOTTOM any build vs enemy range

Melee bot lane into range-heavy teams (kept off-list melee-ADC).

**Gap MSE** 7.50 pp^2 | **Mean abs gap** 2.16 pp | **Accuracy** 55.29% | **Accuracy if calibrated** 55.40% | **Calibration lift** +0.11 pp | **Empirical effect** -13.91 pp | **HGNN effect** -6.23 pp | **Shrinkage** 0.45x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 160 | 58.75% | 55.08% | -3.67 pp | 60.62% |
| `2` | 684 | 52.05% | 51.68% | -0.37 pp | 53.07% |
| `3` | 814 | 50.12% | 49.54% | -0.58 pp | 56.27% |
| `>= 4` | 223 | 44.84% | 48.85% | +4.01 pp | 54.71% |

### Kaisa BOTTOM any build vs enemy range

High-sample marksman vs enemy range pressure; large n keeps bins low-noise.

**Gap MSE** 6.26 pp^2 | **Mean abs gap** 1.77 pp | **Accuracy** 57.96% | **Accuracy if calibrated** 58.14% | **Calibration lift** +0.18 pp | **Empirical effect** +1.82 pp | **HGNN effect** -2.63 pp | **Shrinkage** -1.44x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 1,666 | 45.44% | 50.21% | +4.77 pp | 58.52% |
| `2` | 6,788 | 47.63% | 48.95% | +1.32 pp | 57.98% |
| `3` | 8,168 | 47.62% | 48.30% | +0.68 pp | 57.57% |
| `>= 4` | 2,410 | 47.26% | 47.57% | +0.31 pp | 58.84% |


## Richer Composition Trajectory Tables

### Kaisa BOTTOM `on_hit` vs enemy frontline count

On-hit marksman shreds added enemy frontline.

**Gap MSE** 4.02 pp^2 | **Mean abs gap** 1.88 pp | **Accuracy** 58.37% | **Accuracy if calibrated** 58.62% | **Calibration lift** +0.25 pp | **Empirical effect** +3.02 pp | **HGNN effect** +4.37 pp | **Shrinkage** 1.44x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,152 | 46.27% | 46.98% | +0.71 pp | 57.42% |
| `1` | 6,811 | 44.80% | 47.31% | +2.52 pp | 58.58% |
| `2` | 3,391 | 46.62% | 48.86% | +2.24 pp | 58.98% |
| `>= 3` | 706 | 49.29% | 51.35% | +2.05 pp | 58.92% |

### Ahri MIDDLE `ability_power` vs enemy frontline count

AP mid improves as enemies stack durable targets.

**Gap MSE** 5.27 pp^2 | **Mean abs gap** 2.19 pp | **Accuracy** 57.22% | **Accuracy if calibrated** 57.41% | **Calibration lift** +0.18 pp | **Empirical effect** +9.61 pp | **HGNN effect** +3.91 pp | **Shrinkage** 0.41x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,940 | 47.71% | 50.69% | +2.98 pp | 56.13% |
| `1` | 7,700 | 50.19% | 51.47% | +1.28 pp | 57.60% |
| `2` | 3,662 | 50.79% | 52.57% | +1.78 pp | 57.62% |
| `>= 3` | 731 | 57.32% | 54.60% | -2.72 pp | 58.69% |

### Sylas JUNGLE `ability_power` vs enemy frontline count

Sustained AP skirmisher into beefy teams.

**Gap MSE** 5.19 pp^2 | **Mean abs gap** 1.75 pp | **Accuracy** 56.50% | **Accuracy if calibrated** 56.45% | **Calibration lift** -0.05 pp | **Empirical effect** +4.71 pp | **HGNN effect** +4.69 pp | **Shrinkage** 1.00x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 1,913 | 52.54% | 53.03% | +0.49 pp | 56.51% |
| `1` | 3,001 | 51.48% | 53.44% | +1.96 pp | 55.98% |
| `2` | 1,398 | 51.57% | 55.63% | +4.06 pp | 57.30% |
| `>= 3` | 276 | 57.25% | 57.72% | +0.47 pp | 57.97% |

### Sylas MIDDLE `ability_power` vs enemy frontline count

Same AP anti-frontline pattern from lane.

**Gap MSE** 5.43 pp^2 | **Mean abs gap** 2.11 pp | **Accuracy** 56.55% | **Accuracy if calibrated** 56.65% | **Calibration lift** +0.10 pp | **Empirical effect** +4.52 pp | **HGNN effect** +3.73 pp | **Shrinkage** 0.83x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,238 | 51.52% | 53.30% | +1.79 pp | 56.66% |
| `1` | 3,483 | 49.96% | 53.68% | +3.72 pp | 55.84% |
| `2` | 1,656 | 52.66% | 54.58% | +1.92 pp | 57.67% |
| `>= 3` | 323 | 56.04% | 57.04% | +1.00 pp | 57.59% |

### Karma UTILITY any build vs enemy frontline count

Utility support gains value as enemies stack frontline to zone.

**Gap MSE** 2.36 pp^2 | **Mean abs gap** 1.31 pp | **Accuracy** 57.84% | **Accuracy if calibrated** 58.02% | **Calibration lift** +0.18 pp | **Empirical effect** +7.52 pp | **HGNN effect** +5.75 pp | **Shrinkage** 0.77x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 6,294 | 46.50% | 48.98% | +2.48 pp | 58.40% |
| `1` | 9,507 | 49.75% | 50.17% | +0.41 pp | 58.14% |
| `2` | 4,288 | 50.26% | 51.88% | +1.62 pp | 56.13% |
| `>= 3` | 796 | 54.02% | 54.74% | +0.72 pp | 59.05% |

### Vayne BOTTOM `on_hit` vs enemy frontline count

Classic anti-tank marksman pattern.

**Gap MSE** 8.58 pp^2 | **Mean abs gap** 2.81 pp | **Accuracy** 57.50% | **Accuracy if calibrated** 56.75% | **Calibration lift** -0.75 pp | **Empirical effect** +10.75 pp | **HGNN effect** +9.10 pp | **Shrinkage** 0.85x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 835 | 46.47% | 44.26% | -2.21 pp | 55.93% |
| `1` | 1,629 | 49.42% | 46.06% | -3.36 pp | 57.03% |
| `2` | 808 | 50.25% | 48.42% | -1.83 pp | 60.40% |
| `>= 3` | 201 | 57.21% | 53.36% | -3.85 pp | 56.22% |

### Thresh UTILITY `ar_tank` vs enemy burst count

Durable engage support punished by multiple burst threats.

**Gap MSE** 2.12 pp^2 | **Mean abs gap** 1.30 pp | **Accuracy** 57.49% | **Accuracy if calibrated** 57.41% | **Calibration lift** -0.08 pp | **Empirical effect** -2.72 pp | **HGNN effect** -3.50 pp | **Shrinkage** 1.29x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 959 | 49.74% | 49.11% | -0.63 pp | 57.25% |
| `1` | 1,690 | 50.53% | 48.21% | -2.32 pp | 58.05% |
| `2` | 854 | 47.78% | 46.94% | -0.84 pp | 56.21% |
| `>= 3` | 134 | 47.01% | 45.61% | -1.41 pp | 60.45% |

### Nautilus UTILITY `mr_tank` vs enemy burst count

High-HP engage tank loses into concentrated burst.

**Gap MSE** 6.49 pp^2 | **Mean abs gap** 2.41 pp | **Accuracy** 58.24% | **Accuracy if calibrated** 59.17% | **Calibration lift** +0.93 pp | **Empirical effect** -2.79 pp | **HGNN effect** -0.83 pp | **Shrinkage** 0.30x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 899 | 46.61% | 48.48% | +1.88 pp | 60.40% |
| `1` | 1,595 | 49.78% | 47.91% | -1.87 pp | 56.87% |
| `2` | 640 | 50.62% | 48.57% | -2.06 pp | 57.81% |
| `>= 3` | 89 | 43.82% | 47.66% | +3.84 pp | 64.04% |

### Zed MIDDLE `lethality` vs enemy burst count

Assassin into enemy burst stacking.

**Gap MSE** 0.89 pp^2 | **Mean abs gap** 0.84 pp | **Accuracy** 58.68% | **Accuracy if calibrated** 58.87% | **Calibration lift** +0.19 pp | **Empirical effect** -0.86 pp | **HGNN effect** -2.30 pp | **Shrinkage** 2.69x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,723 | 52.15% | 52.80% | +0.65 pp | 59.71% |
| `1` | 4,477 | 51.62% | 51.26% | -0.36 pp | 58.77% |
| `2` | 1,695 | 48.67% | 50.22% | +1.55 pp | 57.17% |
| `>= 3` | 232 | 51.29% | 50.50% | -0.80 pp | 56.03% |

### Nami UTILITY `utility_protection` vs enemy burst count

Protective enchanter punished by burst-heavy enemies.

**Gap MSE** 0.31 pp^2 | **Mean abs gap** 0.46 pp | **Accuracy** 56.80% | **Accuracy if calibrated** 56.75% | **Calibration lift** -0.06 pp | **Empirical effect** -3.36 pp | **HGNN effect** -4.38 pp | **Shrinkage** 1.30x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,870 | 50.78% | 51.56% | +0.78 pp | 57.10% |
| `1` | 8,090 | 50.22% | 50.30% | +0.08 pp | 56.76% |
| `2` | 3,473 | 48.17% | 48.92% | +0.75 pp | 56.67% |
| `>= 3` | 504 | 47.42% | 47.19% | -0.23 pp | 55.56% |

### Jinx BOTTOM `crit` vs enemy burst count

Fragile crit carry into burst-heavy enemies.

**Gap MSE** 0.30 pp^2 | **Mean abs gap** 0.49 pp | **Accuracy** 56.19% | **Accuracy if calibrated** 56.20% | **Calibration lift** +0.01 pp | **Empirical effect** -5.94 pp | **HGNN effect** -5.70 pp | **Shrinkage** 0.96x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,115 | 52.88% | 53.08% | +0.20 pp | 56.21% |
| `1` | 7,325 | 51.89% | 51.48% | -0.41 pp | 55.60% |
| `2` | 3,310 | 49.34% | 50.23% | +0.90 pp | 57.19% |
| `>= 3` | 458 | 46.94% | 47.39% | +0.44 pp | 58.30% |

### Malphite TOP `ar_tank` vs heavy damage-taken count

Armor tank loses into teams with multiple high-soak targets.

**Gap MSE** 10.98 pp^2 | **Mean abs gap** 2.61 pp | **Accuracy** 57.12% | **Accuracy if calibrated** 57.58% | **Calibration lift** +0.46 pp | **Empirical effect** -2.97 pp | **HGNN effect** -10.45 pp | **Shrinkage** 3.52x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,082 | 53.22% | 55.03% | +1.81 pp | 56.72% |
| `1` | 3,667 | 50.91% | 50.87% | -0.05 pp | 58.09% |
| `2` | 1,742 | 50.40% | 47.50% | -2.91 pp | 55.97% |
| `>= 3` | 201 | 50.25% | 44.58% | -5.67 pp | 53.73% |

### Viego JUNGLE any build vs enemy high-HP count

On-hit bruiser jungler into high-HP enemy teams.

**Gap MSE** 2.13 pp^2 | **Mean abs gap** 1.45 pp | **Accuracy** 57.80% | **Accuracy if calibrated** 57.83% | **Calibration lift** +0.03 pp | **Empirical effect** +2.84 pp | **HGNN effect** +3.03 pp | **Shrinkage** 1.06x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,323 | 48.00% | 46.52% | -1.48 pp | 57.42% |
| `1` | 6,171 | 49.08% | 47.40% | -1.68 pp | 58.37% |
| `2` | 4,035 | 47.09% | 48.44% | +1.35 pp | 56.88% |
| `>= 3` | 1,304 | 50.84% | 49.55% | -1.29 pp | 58.97% |


## Retained Prior And User-Requested Trajectory Tables

### Malphite all roles `ar_tank` vs enemy physical

Original armor-stack audit, retained beyond TOP-only.

**Gap MSE** 3.96 pp^2 | **Mean abs gap** 1.29 pp | **Accuracy** 57.27% | **Accuracy if calibrated** 57.52% | **Calibration lift** +0.25 pp | **Empirical effect** +10.86 pp | **HGNN effect** +10.16 pp | **Shrinkage** 0.94x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 948 | 44.62% | 45.56% | +0.94 pp | 60.34% |
| `0.387-0.448` | 1,151 | 50.30% | 46.02% | -4.29 pp | 55.69% |
| `0.448-0.508` | 1,710 | 49.36% | 49.89% | +0.53 pp | 57.02% |
| `0.508-0.557` | 2,188 | 52.01% | 51.53% | -0.48 pp | 56.90% |
| `>= 0.557` | 2,664 | 55.48% | 55.72% | +0.24 pp | 57.32% |

### Galio all roles `mr_tank` vs enemy magic

Original anti-magic tank family, broader than MIDDLE-only.

**Gap MSE** 18.51 pp^2 | **Mean abs gap** 3.47 pp | **Accuracy** 61.06% | **Accuracy if calibrated** 60.85% | **Calibration lift** -0.22 pp | **Empirical effect** +14.59 pp | **HGNN effect** +6.47 pp | **Shrinkage** 0.44x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 186 | 35.48% | 43.35% | +7.87 pp | 68.82% |
| `0.373-0.423` | 235 | 42.13% | 44.60% | +2.47 pp | 61.28% |
| `0.423-0.486` | 312 | 42.95% | 45.43% | +2.48 pp | 60.90% |
| `0.486-0.549` | 416 | 50.24% | 45.95% | -4.29 pp | 58.17% |
| `>= 0.549` | 695 | 50.07% | 49.82% | -0.25 pp | 60.72% |

### Nautilus all roles `mr_tank` vs enemy magic

Top-20 MR-tank anti-magic case alongside Galio.

**Gap MSE** 4.52 pp^2 | **Mean abs gap** 1.68 pp | **Accuracy** 58.35% | **Accuracy if calibrated** 59.05% | **Calibration lift** +0.70 pp | **Empirical effect** +4.89 pp | **HGNN effect** -0.11 pp | **Shrinkage** -0.02x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 231 | 45.45% | 49.14% | +3.68 pp | 60.61% |
| `0.373-0.423` | 443 | 48.31% | 48.58% | +0.27 pp | 59.37% |
| `0.423-0.486` | 512 | 50.59% | 47.93% | -2.65 pp | 58.40% |
| `0.486-0.549` | 799 | 47.18% | 46.73% | -0.45 pp | 55.44% |
| `>= 0.549` | 1,297 | 50.35% | 49.03% | -1.32 pp | 59.37% |

### Nautilus all roles `ar_tank` vs enemy physical

Physical-heavy enemy teams remain a support-tank check.

**Gap MSE** 5.46 pp^2 | **Mean abs gap** 1.62 pp | **Accuracy** 57.70% | **Accuracy if calibrated** 57.45% | **Calibration lift** -0.25 pp | **Empirical effect** +3.73 pp | **HGNN effect** +7.87 pp | **Shrinkage** 2.11x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.387` | 935 | 49.09% | 45.04% | -4.05 pp | 57.01% |
| `0.387-0.448` | 1,426 | 48.60% | 45.37% | -3.23 pp | 55.89% |
| `0.448-0.508` | 1,844 | 47.56% | 48.24% | +0.68 pp | 57.70% |
| `0.508-0.557` | 2,129 | 49.46% | 49.39% | -0.07 pp | 58.06% |
| `>= 0.557` | 2,465 | 52.82% | 52.91% | +0.09 pp | 58.70% |

### Darius TOP any build vs enemy range count

Static team range pressure, stronger than lane-only range.

**Gap MSE** 0.95 pp^2 | **Mean abs gap** 0.67 pp | **Accuracy** 57.84% | **Accuracy if calibrated** 57.95% | **Calibration lift** +0.11 pp | **Empirical effect** -2.91 pp | **HGNN effect** -5.20 pp | **Shrinkage** 1.79x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 1` | 769 | 52.15% | 52.54% | +0.40 pp | 57.22% |
| `2` | 3,389 | 49.69% | 49.46% | -0.23 pp | 57.57% |
| `3` | 3,804 | 47.63% | 47.79% | +0.16 pp | 58.89% |
| `>= 4` | 1,048 | 49.24% | 47.35% | -1.89 pp | 55.34% |

### Darius TOP any build vs same-role range

User-requested static melee/ranged lane audit.

**Gap MSE** 0.06 pp^2 | **Mean abs gap** 0.25 pp | **Accuracy** 57.84% | **Accuracy if calibrated** 57.95% | **Calibration lift** +0.11 pp | **Empirical effect** -1.77 pp | **HGNN effect** -1.28 pp | **Shrinkage** 0.72x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 250` | 7,440 | 49.29% | 48.99% | -0.29 pp | 57.98% |
| `> 250` | 1,570 | 47.52% | 47.72% | +0.20 pp | 57.13% |

### MasterYi JUNGLE any build vs enemy hard CC

User-requested low-CC audit; unique even though gap is modest.

**Gap MSE** 13.01 pp^2 | **Mean abs gap** 3.38 pp | **Accuracy** 57.74% | **Accuracy if calibrated** 58.02% | **Calibration lift** +0.28 pp | **Empirical effect** -0.33 pp | **HGNN effect** -3.75 pp | **Shrinkage** 11.22x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,611 | 51.93% | 50.26% | -1.67 pp | 57.95% |
| `1` | 4,097 | 52.18% | 48.30% | -3.89 pp | 57.92% |
| `2` | 2,106 | 49.19% | 46.33% | -2.86 pp | 56.98% |
| `>= 3` | 500 | 51.60% | 46.51% | -5.09 pp | 58.40% |

### Selected enchanters UTILITY with skirmish allies

Original enchanter-with-skirmishers synergy probe.

**Gap MSE** 1.22 pp^2 | **Mean abs gap** 0.89 pp | **Accuracy** 57.46% | **Accuracy if calibrated** 57.58% | **Calibration lift** +0.11 pp | **Empirical effect** +3.78 pp | **HGNN effect** +1.29 pp | **Shrinkage** 0.34x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 40,862 | 50.23% | 50.97% | +0.74 pp | 57.41% |
| `1` | 7,855 | 51.74% | 51.90% | +0.16 pp | 57.77% |
| `>= 2` | 374 | 54.01% | 52.26% | -1.75 pp | 56.42% |

### Low own-damage teams vs enemy heal/shield

Original low-damage into sustain audit.

**Gap MSE** 2.26 pp^2 | **Mean abs gap** 1.36 pp | **Accuracy** 58.17% | **Accuracy if calibrated** 58.25% | **Calibration lift** +0.07 pp | **Empirical effect** -2.04 pp | **HGNN effect** -3.91 pp | **Shrinkage** 1.91x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.028` | 12,089 | 49.48% | 51.49% | +2.01 pp | 58.93% |
| `0.028-0.077` | 11,577 | 48.15% | 49.83% | +1.68 pp | 57.88% |
| `0.077-0.200` | 12,234 | 46.67% | 48.01% | +1.34 pp | 58.62% |
| `0.200-0.202` | 13,203 | 46.71% | 48.32% | +1.61 pp | 57.77% |
| `>= 0.202` | 12,768 | 47.44% | 47.58% | +0.14 pp | 57.71% |

### Ambessa TOP `attack_damage` vs enemy damage

Durable bruiser into enemy damage pressure.

**Gap MSE** 9.44 pp^2 | **Mean abs gap** 2.83 pp | **Accuracy** 57.18% | **Accuracy if calibrated** 58.13% | **Calibration lift** +0.95 pp | **Empirical effect** -3.91 pp | **HGNN effect** -4.40 pp | **Shrinkage** 1.13x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 1,244 | 52.97% | 55.35% | +2.38 pp | 59.81% |
| `0.739-0.764` | 1,432 | 49.09% | 52.50% | +3.41 pp | 55.31% |
| `0.764-0.785` | 1,269 | 49.80% | 51.44% | +1.64 pp | 55.32% |
| `0.785-0.813` | 1,486 | 46.37% | 51.23% | +4.87 pp | 56.53% |
| `>= 0.813` | 1,498 | 49.07% | 50.95% | +1.88 pp | 59.01% |

### LeeSin JUNGLE `ad_off_tank` vs enemy magic

Bruiser jungler resisting magic-heavy enemies.

**Gap MSE** 12.12 pp^2 | **Mean abs gap** 2.80 pp | **Accuracy** 57.73% | **Accuracy if calibrated** 58.56% | **Calibration lift** +0.83 pp | **Empirical effect** -0.92 pp | **HGNN effect** -4.55 pp | **Shrinkage** 4.93x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 773 | 44.37% | 51.03% | +6.65 pp | 54.72% |
| `0.373-0.423` | 720 | 46.11% | 47.88% | +1.77 pp | 56.94% |
| `0.423-0.486` | 668 | 46.11% | 48.01% | +1.90 pp | 59.73% |
| `0.486-0.549` | 513 | 46.00% | 46.68% | +0.68 pp | 58.28% |
| `>= 0.549` | 458 | 43.45% | 46.47% | +3.02 pp | 60.48% |

### Thresh UTILITY `mr_tank` vs enemy magic

MR-tank support anti-magic case.

**Gap MSE** 34.74 pp^2 | **Mean abs gap** 4.99 pp | **Accuracy** 60.42% | **Accuracy if calibrated** 59.89% | **Calibration lift** -0.53 pp | **Empirical effect** +6.29 pp | **HGNN effect** -0.38 pp | **Shrinkage** -0.06x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.373` | 48 | 45.83% | 48.20% | +2.37 pp | 58.33% |
| `0.373-0.423` | 88 | 50.00% | 45.78% | -4.22 pp | 62.50% |
| `0.423-0.486` | 109 | 35.78% | 46.87% | +11.09 pp | 63.30% |
| `0.486-0.549` | 178 | 47.75% | 44.80% | -2.95 pp | 56.74% |
| `>= 0.549` | 330 | 52.12% | 47.82% | -4.30 pp | 61.21% |


## Inspected Lower-Signal Trajectory Tables

### Focus HP `<= 2309` vs enemy burst count

Broad HP-vs-burst check; useful but lower signal than champion-specific rows.

**Gap MSE** 0.26 pp^2 | **Mean abs gap** 0.42 pp | **Accuracy** 57.46% | **Accuracy if calibrated** 57.47% | **Calibration lift** +0.00 pp | **Empirical effect** -5.67 pp | **HGNN effect** -4.31 pp | **Shrinkage** 0.76x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 110,272 | 52.26% | 51.75% | -0.51 pp | 57.53% |
| `1` | 187,700 | 50.80% | 50.56% | -0.24 pp | 57.54% |
| `2` | 83,450 | 49.30% | 49.38% | +0.08 pp | 57.22% |
| `>= 3` | 11,684 | 46.59% | 47.44% | +0.86 pp | 57.26% |

### Focus HP `>= 2478` vs enemy burst count

High-HP slots also drop into burst stacks, so champion/build specificity matters.

**Gap MSE** 1.12 pp^2 | **Mean abs gap** 0.87 pp | **Accuracy** 57.67% | **Accuracy if calibrated** 57.68% | **Calibration lift** +0.01 pp | **Empirical effect** -5.96 pp | **HGNN effect** -4.37 pp | **Shrinkage** 0.73x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 115,235 | 51.09% | 51.33% | +0.24 pp | 57.60% |
| `1` | 197,431 | 49.55% | 50.00% | +0.45 pp | 57.69% |
| `2` | 87,290 | 47.91% | 48.86% | +0.95 pp | 57.70% |
| `>= 3` | 12,550 | 45.14% | 46.96% | +1.82 pp | 57.93% |

### Ahri MIDDLE `ability_power` vs heavy damage-taken count

AP mid vs multiple high-soak enemies; weaker axis than frontline count.

**Gap MSE** 6.00 pp^2 | **Mean abs gap** 2.23 pp | **Accuracy** 57.22% | **Accuracy if calibrated** 57.47% | **Calibration lift** +0.25 pp | **Empirical effect** -2.79 pp | **HGNN effect** -1.83 pp | **Shrinkage** 0.66x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 4,274 | 49.98% | 52.45% | +2.47 pp | 57.81% |
| `1` | 8,146 | 49.18% | 51.56% | +2.38 pp | 57.00% |
| `2` | 4,134 | 51.60% | 50.99% | -0.61 pp | 57.11% |
| `>= 3` | 479 | 47.18% | 50.62% | +3.44 pp | 56.78% |

### Kaisa BOTTOM `on_hit` vs heavy damage-taken count

On-hit marksman vs high-soak enemies; frontline count is the stronger cut.

**Gap MSE** 11.16 pp^2 | **Mean abs gap** 2.82 pp | **Accuracy** 58.37% | **Accuracy if calibrated** 58.61% | **Calibration lift** +0.25 pp | **Empirical effect** -5.50 pp | **HGNN effect** -1.54 pp | **Shrinkage** 0.28x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 3,804 | 46.14% | 48.08% | +1.94 pp | 57.52% |
| `1` | 7,262 | 45.84% | 47.76% | +1.92 pp | 58.92% |
| `2` | 3,583 | 46.05% | 47.55% | +1.50 pp | 58.50% |
| `>= 3` | 411 | 40.63% | 46.54% | +5.91 pp | 55.23% |


## Top-20 Matchup And Synergy Audits

### Yasuo MIDDLE `crit` with ally CC

Yasuo's ult chains off ally knock-ups; scales with team CC.

**Gap MSE** 1.12 pp^2 | **Mean abs gap** 0.92 pp | **Accuracy** 57.71% | **Accuracy if calibrated** 57.72% | **Calibration lift** +0.01 pp | **Empirical effect** +2.90 pp | **HGNN effect** +5.77 pp | **Shrinkage** 1.99x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.374` | 1,204 | 48.84% | 47.13% | -1.71 pp | 60.80% |
| `0.374-0.429` | 1,714 | 49.36% | 48.32% | -1.04 pp | 56.13% |
| `0.429-0.479` | 1,978 | 49.19% | 49.49% | +0.30 pp | 58.19% |
| `0.479-0.539` | 2,274 | 50.88% | 51.25% | +0.37 pp | 58.00% |
| `>= 0.539` | 2,424 | 51.73% | 52.90% | +1.17 pp | 56.64% |

### Jhin BOTTOM `crit` with ally CC

Immobile crit marksman; measured synergy with team CC is near flat.

**Gap MSE** 1.90 pp^2 | **Mean abs gap** 1.21 pp | **Accuracy** 57.53% | **Accuracy if calibrated** 57.62% | **Calibration lift** +0.09 pp | **Empirical effect** +0.88 pp | **HGNN effect** +4.10 pp | **Shrinkage** 4.64x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.374` | 1,020 | 47.25% | 44.94% | -2.31 pp | 57.84% |
| `0.374-0.429` | 2,329 | 47.40% | 45.93% | -1.47 pp | 58.91% |
| `0.429-0.479` | 3,703 | 45.75% | 46.74% | +1.00 pp | 57.06% |
| `0.479-0.539` | 5,293 | 47.38% | 47.76% | +0.37 pp | 57.32% |
| `>= 0.539` | 7,900 | 48.14% | 49.05% | +0.91 pp | 57.44% |

### Lulu UTILITY `utility_protection` with ally damage

Enchanter value rises with carry damage to amplify and peel for.

**Gap MSE** 2.16 pp^2 | **Mean abs gap** 1.37 pp | **Accuracy** 56.64% | **Accuracy if calibrated** 56.79% | **Calibration lift** +0.16 pp | **Empirical effect** +4.96 pp | **HGNN effect** +4.52 pp | **Shrinkage** 0.91x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.739` | 4,674 | 47.67% | 49.55% | +1.88 pp | 58.04% |
| `0.739-0.764` | 3,938 | 52.21% | 51.41% | -0.80 pp | 56.30% |
| `0.764-0.785` | 3,031 | 51.20% | 51.96% | +0.76 pp | 54.90% |
| `0.785-0.813` | 1,904 | 51.00% | 53.00% | +2.00 pp | 55.36% |
| `>= 0.813` | 361 | 52.63% | 54.07% | +1.44 pp | 63.43% |

### Ezreal BOTTOM `attack_damage` vs enemy hard CC

Skillshot poke marksman punished as enemy hard CC stacks.

**Gap MSE** 3.62 pp^2 | **Mean abs gap** 1.53 pp | **Accuracy** 57.74% | **Accuracy if calibrated** 57.68% | **Calibration lift** -0.06 pp | **Empirical effect** -3.60 pp | **HGNN effect** -1.23 pp | **Shrinkage** 0.34x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 8,690 | 48.67% | 49.67% | +1.00 pp | 56.50% |
| `1` | 13,797 | 47.25% | 48.64% | +1.39 pp | 58.53% |
| `2` | 6,856 | 47.56% | 47.94% | +0.37 pp | 57.37% |
| `>= 3` | 1,418 | 45.06% | 48.44% | +3.38 pp | 59.45% |

### Jayce TOP `attack_damage` vs enemy frontline count

Poke bruiser empirically holds up into frontline-heavy teams; model heavily shrinks the effect.

**Gap MSE** 4.58 pp^2 | **Mean abs gap** 1.90 pp | **Accuracy** 57.29% | **Accuracy if calibrated** 57.43% | **Calibration lift** +0.15 pp | **Empirical effect** +7.46 pp | **HGNN effect** +4.82 pp | **Shrinkage** 0.65x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 2,587 | 47.04% | 46.33% | -0.72 pp | 57.60% |
| `1` | 4,260 | 49.18% | 47.01% | -2.16 pp | 56.36% |
| `2` | 2,121 | 50.02% | 48.68% | -1.35 pp | 58.89% |
| `>= 3` | 455 | 54.51% | 51.14% | -3.36 pp | 56.70% |

### LeeSin JUNGLE `attack_damage` vs enemy scaling

Early-tempo bruiser jungler fades as enemy scaling rises.

**Gap MSE** 0.36 pp^2 | **Mean abs gap** 0.48 pp | **Accuracy** 57.49% | **Accuracy if calibrated** 57.56% | **Calibration lift** +0.07 pp | **Empirical effect** -2.54 pp | **HGNN effect** -1.77 pp | **Shrinkage** 0.70x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `<= 0.829` | 3,589 | 52.74% | 52.60% | -0.15 pp | 57.76% |
| `0.829-0.841` | 3,396 | 49.97% | 50.45% | +0.48 pp | 58.19% |
| `0.841-0.852` | 3,792 | 49.34% | 50.42% | +1.08 pp | 56.88% |
| `0.852-0.863` | 3,385 | 50.64% | 50.58% | -0.05 pp | 57.16% |
| `>= 0.863` | 4,083 | 50.21% | 50.83% | +0.62 pp | 57.51% |

### Caitlyn BOTTOM `crit` vs enemy burst count

Immobile siege ADC punished by multiple burst and dive threats.

**Gap MSE** 4.00 pp^2 | **Mean abs gap** 1.47 pp | **Accuracy** 56.75% | **Accuracy if calibrated** 56.84% | **Calibration lift** +0.09 pp | **Empirical effect** -4.99 pp | **HGNN effect** -2.12 pp | **Shrinkage** 0.42x

| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |
|---|---:|---:|---:|---:|---:|
| `0` | 5,009 | 49.73% | 50.58% | +0.85 pp | 56.56% |
| `1` | 9,561 | 49.87% | 49.99% | +0.12 pp | 56.55% |
| `2` | 4,331 | 48.30% | 49.47% | +1.16 pp | 57.31% |
| `>= 3` | 608 | 44.74% | 48.47% | +3.73 pp | 57.57% |


## Fixed Fixture Overall Summary

Detailed audit tables above are historical validation fixtures rendered from
the `val` split.

| Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 46 | 200 | 1.80 pp | 11.09 pp | 6.01 pp^2 | 57.59% | 57.64% | +0.06 pp |

| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,318,331 | 13,183,310 | 46 | 200 | 1.43 pp | 6.67 pp | 3.46 pp^2 | 58.22% | 58.25% | +0.03 pp |
| Validation | 164,792 | 1,647,920 | 46 | 200 | 1.80 pp | 11.09 pp | 6.01 pp^2 | 57.59% | 57.64% | +0.06 pp |
| Test | 164,792 | 1,647,920 | 46 | 200 | 1.87 pp | 11.85 pp | 6.77 pp^2 | 57.34% | 57.40% | +0.06 pp |

Gap MSE is `mean((HGNN_focus_WR - empirical_focus_WR)^2)` across populated threshold bins, rendered as percentage-points squared.

## Reproduction Commands

The stock renderer below regenerates the fixed 46-spec fixture and model probabilities. It does not regenerate the expanded discovery catalog above; that catalog is an excerpt from `/tmp/hgnn_context_discovery_v2.json`, produced by the scratch scanner used for this refresh. Do not point `--output` at this checked-in document unless the discovery section will be re-merged afterward.

```bash
uv run python -m app.ml.context_examples_audit \
  --context-cache-dir app/ml/data/cache \
  --model-cache-dir app/ml/data/cache \
  --model-path app/ml/data/hgnn_production_model.pt \
  --encoder-sidecar-path app/ml/data/semantic_identity_sidecar_compact.npz \
  --prediction-cache app/ml/data/audit_focus_side_probability.npy \
  --audit-split test \
  --output /tmp/hgnn_context_examples_probe.md \
  --json-output /tmp/hgnn_context_examples_probe.json \
  --refresh-predictions \
  --batch-size 24576
```
