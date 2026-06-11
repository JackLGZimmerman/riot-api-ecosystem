# HGNN Group Context Audit

Updated: 2026-06-11.

Status note: this is the lower-noise semantic calibration guardrail. It complements the champion/identity examples in [HGNN_CONTEXT_EXAMPLES_AUDIT.md](HGNN_CONTEXT_EXAMPLES_AUDIT.md); neither alone is sufficient for semantic-boundary promotion.

This audit re-cuts the context examples onto deterministic build/role groups, then measures the HGNN gap against an empirical-Bayes target. Group pooling lowers the bin sampling floor enough for systematic semantic residuals to be visible.

## Source And Model

- Source population verified in ClickHouse: `game_data_filtered.participant_stats` has 16,479,150 participant rows over 1,647,915 valid games.
- Split sizes: train 1,318,331, validation 164,792, test 164,792 games.
- Context cache: `app/ml/data/cache`, with split metadata matching `game_data_filtered.ml_game_player_pivot`.
- Prediction cache regenerated from `app/ml/data/hgnn_production_model.pt` into `app/ml/data/audit_focus_side_probability.npy`.
- Semantic group feature schema: v2, 25 compact per-slot features.

The promoted production checkpoint uses `semantic_moe_num_experts = 128`, `semantic_moe_top_k = 32`, `use_learned_semantic_moe = true`, `use_semantic_group_features = true`, and `semantic_identity_sidecar_compact.npz`.

## Production Result (51 groups, 219 populated bins)

`systematic_gap_mse` is in pp^2 and lower is better. It subtracts the EB target variance from the squared model gap.

| Split | bins | median n | min n | raw MSE | raw floor | EB MSE | EB floor | systematic | clipped | mean abs EB gap | max abs EB gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 219 | 111,380 | 3,071 | 0.80 pp^2 | 0.06 pp^2 | 0.81 pp^2 | 0.04 pp^2 | 0.77 pp^2 | 0.78 pp^2 | 0.6 pp | 4.5 pp |
| Validation | 219 | 13,817 | 348 | 1.30 pp^2 | 0.46 pp^2 | 1.28 pp^2 | 0.22 pp^2 | 1.07 pp^2 | 1.12 pp^2 | 0.8 pp | 3.7 pp |
| Test | 219 | 14,333 | 295 | 1.78 pp^2 | 0.46 pp^2 | 1.43 pp^2 | 0.23 pp^2 | 1.20 pp^2 | 1.25 pp^2 | 0.8 pp | 6.4 pp |

Train->test EB target movement: MSE 0.54 pp^2, mean abs 0.52 pp, max abs 3.82 pp over 219 bins.

## Key Findings

- The fresh 51-group surface is materially broader than the stale 16-group/67-bin summary. Validation systematic gap is 1.07 pp^2 and test systematic gap is 1.20 pp^2.
- The largest residuals are not broad average misses. They concentrate in interpretable semantic bins: MR utility tanks into burst, on-hit junglers into hard CC, MR tanks with middling ally damage, AD fighters top into low enemy range, and attack-damage tops into enemy frontline.
- High-support group trajectories still show real empirical movement of roughly 3-7 pp across context thresholds. The HGNN usually moves in the right direction, but often shrinks the slope.
- The discovery pass found 973 stable validation trajectories before report caps. The group catalog below records the high-support examples rather than reducing the audit to only residual outliers.

## Largest Validation Residuals

Rows are sorted by debiased `systematic_gap_mse`; `EB gap` is HGNN minus EB target.

| systematic | group | bin | n | empirical | EB target | HGNN | EB gap |
|---:|---|---:|---:|---:|---:|---:|---:|
| 14.01 | MR tanks UTILITY vs enemy burst count | `>= 3` | 412 | 46.1% | 50.2% | 46.5% | -3.7 pp |
| 12.88 | On-hit junglers vs enemy hard CC | `>= 3` | 772 | 50.0% | 49.8% | 46.1% | -3.7 pp |
| 12.60 | MR tanks UTILITY with ally damage | `0.764-0.785` | 2,708 | 52.6% | 52.3% | 48.6% | -3.7 pp |
| 11.13 | AD fighters TOP vs enemy range count | `<= 1` | 11,116 | 51.1% | 50.2% | 53.6% | +3.3 pp |
| 7.87 | Attack-damage TOP vs enemy frontline count | `>= 3` | 2,707 | 54.2% | 52.8% | 55.7% | +2.9 pp |
| 7.55 | MR tanks UTILITY vs enemy burst count | `2` | 2,641 | 50.7% | 50.2% | 47.5% | -2.7 pp |
| 7.40 | MR tanks MIDDLE vs enemy magic | `<= 0.373` | 348 | 39.4% | 42.7% | 46.1% | +3.4 pp |
| 6.68 | On-hit junglers vs enemy hard CC | `1` | 6,406 | 50.0% | 50.0% | 47.3% | -2.6 pp |
| 6.64 | MR tanks MIDDLE vs enemy magic | `0.486-0.549` | 769 | 49.5% | 49.2% | 46.2% | -3.0 pp |
| 5.37 | Low own-damage teams vs enemy heal/shield | `<= 0.028` | 12,089 | 49.5% | 49.1% | 51.5% | +2.4 pp |
| 5.37 | Attack-damage junglers vs enemy scaling | `0.829-0.841` | 13,137 | 53.3% | 53.3% | 50.9% | -2.4 pp |
| 5.15 | AD off-tanks JUNGLE vs enemy magic | `>= 0.549` | 8,428 | 47.9% | 47.9% | 45.6% | -2.3 pp |
| 4.49 | Crit carries MIDDLE with ally CC | `<= 0.374` | 5,717 | 49.1% | 49.6% | 47.4% | -2.2 pp |
| 4.33 | MR tanks UTILITY vs enemy magic | `0.486-0.549` | 3,040 | 49.7% | 50.0% | 47.8% | -2.2 pp |
| 4.24 | AD off-tanks JUNGLE vs enemy magic | `0.486-0.549` | 9,119 | 47.0% | 47.4% | 45.3% | -2.1 pp |

## High-Support Empirical Group Catalog

This table excerpts the top 45 historical validation group rows from the 60
selected group trajectories in `/tmp/hgnn_context_discovery_v2.json`; ranking is
by validation discovery score after dedupe and axis caps. Rerun the audit on
`test` for the current v32 test-only protocol.

| Example | Axis | historical val n | min bin n | Stable emp effect | Stable HGNN effect | Stable slope gap | Mean abs gap | Test effect |
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
| crit BOTTOM group | ally tank build count | 184,702 | 5,808 | -4.5 pp | -3.4 pp | +1.1 pp | 0.4 pp | -3.9 pp |
| ar_tank UTILITY group | enemy physical | 47,837 | 5,934 | +4.2 pp | +5.9 pp | +1.7 pp | 1.1 pp | +4.1 pp |
| crit BOTTOM group | ally ad build count | 184,631 | 10,487 | -3.3 pp | -7.4 pp | -4.0 pp | 1.3 pp | -5.0 pp |
| crit BOTTOM group | enemy marksman build count | 182,932 | 7,294 | -4.3 pp | -2.8 pp | +1.5 pp | 0.7 pp | -5.9 pp |
| crit BOTTOM group | enemy burst count | 184,781 | 5,481 | -4.6 pp | -3.9 pp | +0.7 pp | 0.3 pp | -4.9 pp |

## Detailed Group Discovery Tables

### attack_damage all roles group vs enemy marksman build count

Validation support: n=212,774, min stable bin n=8,519. Stable empirical effect -7.3 pp; stable HGNN effect -4.3 pp; stable slope gap +2.9 pp. Test confirmation did not clear the same support/significance gate.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 8,519 | 56.6% | 53.9% | -2.7 pp | yes |
| `1` | 72,220 | 52.7% | 51.9% | -0.8 pp | yes |
| `2` | 96,493 | 50.3% | 49.9% | -0.4 pp | yes |
| `3` | 35,542 | 49.3% | 49.6% | +0.3 pp | yes |
| `4` | 2,329 | 50.8% | 53.4% | +2.6 pp | no |
| `5` | 43 | 55.8% | 54.7% | -1.1 pp | no |

### lethality JUNGLE group vs ally scaling

Validation support: n=62,511, min stable bin n=7,119. Stable empirical effect +7.3 pp; stable HGNN effect +6.2 pp; stable slope gap -1.1 pp. Test confirmation effect +6.4 pp with min bin n=6,742.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.829` | 7,119 | 41.5% | 42.1% | +0.6 pp | yes |
| `20-40% 0.829-0.841` | 9,933 | 44.8% | 46.5% | +1.6 pp | yes |
| `40-60% 0.841-0.852` | 13,675 | 48.1% | 48.2% | +0.1 pp | yes |
| `60-80% 0.852-0.863` | 13,979 | 48.8% | 48.4% | -0.4 pp | yes |
| `80-100% >= 0.863` | 17,805 | 48.8% | 48.3% | -0.5 pp | yes |

### ability_power MIDDLE group vs enemy burst count

Validation support: n=224,155, min stable bin n=6,706. Stable empirical effect -6.6 pp; stable HGNN effect -3.8 pp; stable slope gap +2.8 pp. Test confirmation effect -5.2 pp with min bin n=6,051.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 62,119 | 51.8% | 51.9% | +0.1 pp | yes |
| `1` | 107,356 | 50.3% | 50.7% | +0.4 pp | yes |
| `2` | 47,974 | 48.7% | 49.8% | +1.1 pp | yes |
| `3` | 6,706 | 45.3% | 48.1% | +2.9 pp | yes |
| `4` | 215 | 47.4% | 44.2% | -3.2 pp | no |

### ability_power MIDDLE group vs ally marksman build count

Validation support: n=223,536, min stable bin n=9,268. Stable empirical effect +6.7 pp; stable HGNN effect +5.6 pp; stable slope gap -1.2 pp. Test confirmation effect +7.8 pp with min bin n=7,777.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 9,268 | 45.3% | 47.4% | +2.0 pp | yes |
| `1` | 83,207 | 48.7% | 49.2% | +0.5 pp | yes |
| `2` | 100,599 | 51.4% | 51.7% | +0.4 pp | yes |
| `3` | 30,462 | 52.1% | 52.9% | +0.8 pp | yes |
| `4` | 834 | 53.1% | 52.0% | -1.1 pp | no |

### ability_power all roles group vs enemy burst count

Validation support: n=424,236, min stable bin n=12,854. Stable empirical effect -6.2 pp; stable HGNN effect -3.9 pp; stable slope gap +2.3 pp. Test confirmation effect -5.1 pp with min bin n=11,692.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 117,221 | 51.8% | 52.1% | +0.3 pp | yes |
| `1` | 202,770 | 50.4% | 51.0% | +0.6 pp | yes |
| `2` | 91,391 | 48.8% | 49.9% | +1.1 pp | yes |
| `3` | 12,854 | 45.6% | 48.2% | +2.6 pp | yes |
| `4` | 425 | 49.6% | 43.9% | -5.7 pp | no |

### lethality all roles group vs ally scaling

Validation support: n=110,456, min stable bin n=12,869. Stable empirical effect +6.6 pp; stable HGNN effect +5.4 pp; stable slope gap -1.2 pp. Test confirmation effect +5.7 pp with min bin n=11,924.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.829` | 12,869 | 42.3% | 43.3% | +1.0 pp | yes |
| `20-40% 0.829-0.841` | 16,931 | 45.9% | 47.1% | +1.2 pp | yes |
| `40-60% 0.841-0.852` | 22,798 | 48.4% | 48.5% | +0.1 pp | yes |
| `60-80% 0.852-0.863` | 23,884 | 48.9% | 48.8% | -0.1 pp | yes |
| `80-100% >= 0.863` | 33,974 | 48.9% | 48.8% | -0.2 pp | yes |

### utility_protection UTILITY group vs ally marksman build count

Validation support: n=161,611, min stable bin n=6,053. Stable empirical effect +5.8 pp; stable HGNN effect +3.1 pp; stable slope gap -2.6 pp. Test confirmation effect +7.6 pp with min bin n=4,775.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 6,053 | 45.7% | 48.6% | +2.9 pp | yes |
| `1` | 56,948 | 49.4% | 49.9% | +0.5 pp | yes |
| `2` | 74,052 | 51.5% | 51.5% | -0.0 pp | yes |
| `3` | 24,558 | 51.5% | 51.7% | +0.2 pp | yes |
| `4` | 962 | 47.1% | 48.6% | +1.6 pp | no |

### attack_damage TOP group vs ally scaling

Validation support: n=63,087, min stable bin n=7,286. Stable empirical effect +4.9 pp; stable HGNN effect +0.3 pp; stable slope gap -4.6 pp. Test confirmation effect +3.0 pp with min bin n=7,216.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.829` | 7,286 | 47.0% | 50.6% | +3.6 pp | yes |
| `20-40% 0.829-0.841` | 10,024 | 50.1% | 51.3% | +1.3 pp | yes |
| `40-60% 0.841-0.852` | 13,717 | 51.0% | 51.4% | +0.5 pp | yes |
| `60-80% 0.852-0.863` | 14,023 | 51.3% | 51.3% | -0.0 pp | yes |
| `80-100% >= 0.863` | 18,037 | 51.9% | 50.9% | -1.0 pp | yes |

### ability_power all roles group vs enemy marksman build count

Validation support: n=419,945, min stable bin n=16,326. Stable empirical effect -5.3 pp; stable HGNN effect -2.9 pp; stable slope gap +2.4 pp. Test confirmation effect -6.3 pp with min bin n=14,006.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0` | 16,326 | 54.5% | 53.2% | -1.3 pp | yes |
| `1` | 141,247 | 51.6% | 51.9% | +0.3 pp | yes |
| `2` | 191,680 | 49.5% | 50.3% | +0.8 pp | yes |
| `3` | 70,692 | 49.1% | 50.3% | +1.1 pp | yes |
| `4` | 4,648 | 50.0% | 53.1% | +3.2 pp | no |
| `5` | 68 | 48.5% | 55.0% | +6.5 pp | no |

### ap_off_tank all roles group vs enemy damage

Validation support: n=51,029, min stable bin n=9,410. Stable empirical effect -5.9 pp; stable HGNN effect -4.9 pp; stable slope gap +0.9 pp. Test confirmation effect -5.7 pp with min bin n=8,868.

| Bin | n | Empirical WR | HGNN WR | Gap | Stable |
|---|---:|---:|---:|---:|---|
| `0-20% <= 0.739` | 11,350 | 51.4% | 52.2% | +0.8 pp | yes |
| `20-40% 0.739-0.764` | 10,219 | 49.2% | 49.1% | -0.1 pp | yes |
| `40-60% 0.764-0.785` | 9,410 | 47.1% | 48.0% | +0.9 pp | yes |
| `60-80% 0.785-0.813` | 9,927 | 47.2% | 47.3% | +0.2 pp | yes |
| `80-100% >= 0.813` | 10,123 | 45.5% | 47.3% | +1.7 pp | yes |

## Reproduction

Regenerate focus-slot probabilities:

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

Run the group EB audit:

```bash
uv run python -m app.ml.group_context_audit \
  --context-cache-dir app/ml/data/cache \
  --prediction-cache app/ml/data/audit_focus_side_probability.npy \
  --per-row \
  --json-output /tmp/hgnn_group_context_audit_production.json
```

The expanded discovery pass used the same cache/prediction artifacts plus ClickHouse population checks against `game_data_filtered.participant_stats`, `participant_item_value_totals`, `ml_game_player_pivot`, and the static champion stats JSONL. The catalog sections are excerpts from `/tmp/hgnn_context_discovery_v2.json`; the stock commands above regenerate the probabilities and group EB payload, not that expanded discovery catalog.
