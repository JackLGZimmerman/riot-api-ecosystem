# HGNN Context Examples Audit

Updated: 2026-06-02. Audit target:
`app/ml/data/structured_winrate_model.pt`.

Production semantic context is the threshold-tuned raw identity-conditioned
context head: `use_identity_conditioned_context=true`,
`identity_context_source=raw`, rank `16`, hidden dim `64`, checkpointed by
validation threshold accuracy. This is the naive semantic-context baseline: a
wide draft-safe identity atlas, a small low-rank context interaction, and no
manual champion-specific rules.

Reproducer file: [../context_examples_audit.py](../context_examples_audit.py)

Reproduce:

```bash
.venv/bin/python -m app.ml.context_examples_audit --model-path app/ml/data/structured_winrate_model.pt
```

The audit is model-aligned: bins use draft-time `identity_context` values that
the model can see. It covers all train/val/test side rows (`2,862,626` rows).
Post-game realized damage-share slices are excluded because the cache has no
match id or post-game damage columns to join back to predictions.

## Production Metrics

| Split | Accuracy | Threshold Acc | AUC | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.5735 | 0.5710 | 0.6038 | 0.6747 | 0.2410 | 0.0037 |
| val | 0.5750 | 0.5779 | 0.6014 | 0.6749 | 0.2411 | 0.0252 |
| test | 0.5717 | 0.5743 | 0.5972 | 0.6763 | 0.2418 | 0.0222 |

Selected checkpoint: epoch `3`, validation threshold `0.528`.

## Conventions

| Term | Definition |
| --- | --- |
| `Band` | Global side-row quintile band for the continuous context axis: `0-20`, `20-40`, `40-60`, `60-80`, or `80-100`. |
| `Emp WR` | Actual focus-side win rate. |
| `Model WR` | Mean predicted focus-side win probability. |
| `Base WR` | Mean model probability with the context residual removed. |
| `Gap` | `Model WR - Emp WR`. |
| `Effect` | `80-100` band WR minus `0-20` band WR unless the table says `low-high`. |
| `Delta gap` | `Model effect - Emp effect`. |

Continuous bands are computed globally before identity/build filters are
applied. Discrete count axes keep count bins such as `0`, `1`, and `>=2`. WR,
gaps, and effects are percentage points.

## Summary

| Context | Axis | Direction | Low band | High band | Emp effect | Model effect | Delta gap | Read |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Malphite `ar_tank` | enemy physical share | high-low | `0-20` | `80-100` | +9.04 | +4.69 | -4.35 | under-fit, improved vs shared |
| Malphite TOP `ar_tank` | enemy physical share | high-low | `0-20` | `80-100` | +9.47 | +4.58 | -4.89 | same pattern |
| Low own damage into enemy heal/shield | pressure bands | high-low | `dmg0-20 heal0-20` | `dmg0-20 heal80-100` | -2.28 | -2.28 | +0.00 | captured |
| Sion TOP `ad_off_tank` | enemy damage pressure | low-high | `0-20` | `80-100` | +1.49 | +3.13 | +1.64 | direction captured, tail over-swing |
| DrMundo `ad_off_tank` | enemy magic share | high-low | `0-20` | `80-100` | +2.01 | +2.53 | +0.52 | captured |
| DrMundo `mr_tank` | enemy magic share | high-low | `0-20` | `80-100` | +1.03 | +4.23 | +3.20 | over-reads top quintile |
| Selected enchanters | skirmish allies | high-low | `0` | `>=2` | +2.67 | +2.71 | +0.04 | captured |

## Malphite `ar_tank` Vs Enemy Physical Share

Enemy physical share is the damage-weighted mean enemy `phys_offense_share`.

| Scope | Band | n | Emp WR | Model WR | Base WR | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all roles | `0-20` | 9,871 | 45.34 | 47.22 | 48.53 | +1.88 |
| all roles | `20-40` | 13,478 | 47.34 | 46.85 | 48.54 | -0.48 |
| all roles | `40-60` | 16,369 | 49.16 | 48.60 | 49.46 | -0.56 |
| all roles | `60-80` | 19,442 | 50.31 | 49.79 | 49.57 | -0.52 |
| all roles | `80-100` | 22,884 | 54.38 | 51.91 | 49.36 | -2.47 |
| TOP only | `0-20` | 7,161 | 44.88 | 47.32 | 48.57 | +2.44 |
| TOP only | `20-40` | 9,695 | 47.34 | 46.92 | 48.58 | -0.42 |
| TOP only | `40-60` | 12,449 | 49.47 | 48.72 | 49.56 | -0.75 |
| TOP only | `60-80` | 15,076 | 50.76 | 49.95 | 49.73 | -0.81 |
| TOP only | `80-100` | 17,269 | 54.35 | 51.90 | 49.38 | -2.44 |

The conditioned head moves in the right direction but still under-fits Malphite
against high-physical enemy teams.

## Damage Into Enemy Heal/Shield

Own damage and enemy heal/shield are mean team `identity_context` pressure axes.

| Focus damage band | Enemy heal/shield band | n | Emp WR | Model WR | Base WR | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `0-20` | `0-20` | 113,731 | 49.68 | 49.44 | 51.02 | -0.24 |
| `0-20` | `20-40` | 114,933 | 48.23 | 48.35 | 50.29 | +0.12 |
| `0-20` | `40-60` | 113,775 | 47.58 | 47.54 | 49.58 | -0.05 |
| `0-20` | `60-80` | 115,428 | 47.36 | 47.24 | 49.03 | -0.12 |
| `0-20` | `80-100` | 114,658 | 47.40 | 47.16 | 49.00 | -0.24 |
| `20-40` | `0-20` | 113,937 | 51.70 | 51.43 | 51.34 | -0.27 |
| `20-40` | `20-40` | 114,294 | 50.26 | 50.35 | 50.64 | +0.10 |
| `20-40` | `40-60` | 114,681 | 49.61 | 49.58 | 49.96 | -0.03 |
| `20-40` | `60-80` | 115,002 | 49.07 | 49.13 | 49.37 | +0.06 |
| `20-40` | `80-100` | 114,610 | 49.22 | 49.18 | 49.38 | -0.04 |
| `40-60` | `0-20` | 114,037 | 52.26 | 52.17 | 51.48 | -0.09 |
| `40-60` | `20-40` | 114,350 | 51.34 | 51.07 | 50.78 | -0.27 |
| `40-60` | `40-60` | 114,937 | 50.28 | 50.33 | 50.14 | +0.06 |
| `40-60` | `60-80` | 114,447 | 49.69 | 49.82 | 49.48 | +0.14 |
| `40-60` | `80-100` | 114,753 | 49.88 | 49.92 | 49.54 | +0.04 |
| `60-80` | `0-20` | 114,695 | 52.79 | 52.56 | 51.45 | -0.23 |
| `60-80` | `20-40` | 114,250 | 51.41 | 51.43 | 50.77 | +0.02 |
| `60-80` | `40-60` | 115,011 | 50.70 | 50.76 | 50.17 | +0.06 |
| `60-80` | `60-80` | 113,884 | 49.86 | 50.13 | 49.43 | +0.28 |
| `60-80` | `80-100` | 114,687 | 50.29 | 50.30 | 49.47 | +0.01 |
| `80-100` | `0-20` | 116,125 | 51.90 | 51.91 | 50.71 | +0.01 |
| `80-100` | `20-40` | 114,698 | 51.24 | 50.94 | 50.12 | -0.30 |
| `80-100` | `40-60` | 114,109 | 50.05 | 50.20 | 49.50 | +0.16 |
| `80-100` | `60-80` | 113,775 | 48.68 | 49.35 | 48.62 | +0.66 |
| `80-100` | `80-100` | 113,819 | 49.52 | 49.70 | 48.75 | +0.18 |

Low-damage teams lose `-2.28` empirical WR into the highest enemy heal/shield
band; production predicts `-2.28`.

## Sion TOP `ad_off_tank` Vs Enemy Damage Pressure

Enemy damage is mean enemy `champion_damage_pressure`.

| Band | n | Emp WR | Model WR | Base WR | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0-20` | 948 | 55.49 | 54.25 | 51.47 | -1.23 |
| `20-40` | 922 | 53.69 | 52.05 | 51.21 | -1.63 |
| `40-60` | 946 | 52.22 | 51.50 | 51.07 | -0.72 |
| `60-80` | 1,076 | 50.93 | 51.46 | 51.44 | +0.53 |
| `80-100` | 1,026 | 54.00 | 51.12 | 51.67 | -2.87 |

The expected Sion effect is low-high. The model captures that direction but
over-swings the top quintile.

## DrMundo Vs Enemy Magic Share

Enemy magic share is the damage-weighted mean enemy `magic_offense_share`.

| Build | Band | n | Emp WR | Model WR | Base WR | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ad_off_tank` | `0-20` | 1,344 | 61.83 | 60.22 | 55.41 | -1.61 |
| `ad_off_tank` | `20-40` | 1,526 | 60.42 | 60.20 | 56.11 | -0.22 |
| `ad_off_tank` | `40-60` | 1,657 | 61.98 | 60.42 | 56.31 | -1.56 |
| `ad_off_tank` | `60-80` | 1,803 | 59.29 | 60.61 | 56.08 | +1.32 |
| `ad_off_tank` | `80-100` | 1,676 | 63.84 | 62.75 | 56.22 | -1.10 |
| `mr_tank` | `0-20` | 1,305 | 51.03 | 49.46 | 49.01 | -1.58 |
| `mr_tank` | `20-40` | 2,013 | 50.27 | 49.31 | 49.62 | -0.96 |
| `mr_tank` | `40-60` | 3,329 | 48.96 | 49.91 | 50.00 | +0.94 |
| `mr_tank` | `60-80` | 5,344 | 49.08 | 50.11 | 49.69 | +1.03 |
| `mr_tank` | `80-100` | 8,388 | 52.06 | 53.69 | 50.22 | +1.63 |

`ad_off_tank` is captured. `mr_tank` over-reads the highest enemy-magic band.

## Enchanters With Skirmish-Heavy Allies

Selected enchanters are Sona, Karma, Lulu, and Zilean in `UTILITY` with
`utility_enchanter` or `utility_protection`. Skirmish-heavy allies are Gwen,
Jax, Irelia, Fiora, Udyr, and XinZhao, counted in any focus-team role.

| Support group | Skirmish allies | n | Emp WR | Model WR | Base WR | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected enchanters | 0 | 382,693 | 50.30 | 50.41 | 50.54 | +0.10 |
| selected enchanters | 1 | 73,328 | 52.24 | 51.84 | 51.60 | -0.40 |
| selected enchanters | `>=2` | 3,338 | 52.97 | 53.12 | 52.52 | +0.16 |
| other utility supports | 0 | 1,033,024 | 50.55 | 50.52 | 50.55 | -0.02 |
| other utility supports | 1 | 192,464 | 52.10 | 51.98 | 51.59 | -0.12 |
| other utility supports | `>=2` | 8,392 | 52.40 | 53.42 | 52.68 | +1.03 |

| Enchanter | Skirmish allies | n | Emp WR | Model WR | Base WR | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sona | 0 | 71,304 | 53.27 | 53.12 | 52.83 | -0.15 |
| Sona | 1 | 13,953 | 55.23 | 54.53 | 53.88 | -0.70 |
| Sona | `>=2` | 625 | 55.04 | 55.88 | 54.79 | +0.84 |
| Karma | 0 | 145,241 | 48.79 | 48.80 | 49.08 | +0.02 |
| Karma | 1 | 27,958 | 50.70 | 50.20 | 50.12 | -0.51 |
| Karma | `>=2` | 1,300 | 51.23 | 51.62 | 51.05 | +0.39 |
| Lulu | 0 | 140,056 | 50.15 | 50.43 | 50.77 | +0.28 |
| Lulu | 1 | 26,546 | 52.11 | 51.91 | 51.85 | -0.20 |
| Lulu | `>=2` | 1,186 | 52.45 | 53.02 | 52.76 | +0.57 |
| Zilean | 0 | 26,092 | 51.47 | 51.80 | 51.21 | +0.33 |
| Zilean | 1 | 4,871 | 53.19 | 53.23 | 52.27 | +0.04 |
| Zilean | `>=2` | 227 | 59.91 | 54.71 | 53.40 | -5.20 |

## Armor Tanks Into Enemy Physical Share

Enemy physical share uses global quintile bands. Rows are build `ar_tank`, all
roles.

| Champion | Band | n | Emp WR | Model WR | Base WR | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Maokai | `0-20` | 1,738 | 50.86 | 51.40 | 51.11 | +0.54 |
| Maokai | `20-40` | 2,375 | 51.62 | 50.01 | 51.02 | -1.61 |
| Maokai | `40-60` | 2,826 | 50.88 | 50.55 | 51.44 | -0.33 |
| Maokai | `60-80` | 3,151 | 50.94 | 51.13 | 51.43 | +0.20 |
| Maokai | `80-100` | 3,506 | 53.99 | 52.69 | 51.22 | -1.30 |
| Malphite | `0-20` | 9,871 | 45.34 | 47.22 | 48.53 | +1.88 |
| Malphite | `20-40` | 13,478 | 47.34 | 46.85 | 48.54 | -0.48 |
| Malphite | `40-60` | 16,369 | 49.16 | 48.60 | 49.46 | -0.56 |
| Malphite | `60-80` | 19,442 | 50.31 | 49.79 | 49.57 | -0.52 |
| Malphite | `80-100` | 22,884 | 54.38 | 51.91 | 49.36 | -2.47 |
| Sion | `0-20` | 8,459 | 50.07 | 49.45 | 49.90 | -0.61 |
| Sion | `20-40` | 11,422 | 48.56 | 48.49 | 49.88 | -0.07 |
| Sion | `40-60` | 13,123 | 49.26 | 49.08 | 50.42 | -0.19 |
| Sion | `60-80` | 14,406 | 50.30 | 49.65 | 50.38 | -0.65 |
| Sion | `80-100` | 15,342 | 51.54 | 50.97 | 49.95 | -0.57 |
| Ornn | `0-20` | 7,549 | 51.87 | 50.17 | 50.95 | -1.71 |
| Ornn | `20-40` | 10,221 | 49.56 | 49.42 | 51.07 | -0.15 |
| Ornn | `40-60` | 12,404 | 50.96 | 50.16 | 51.63 | -0.80 |
| Ornn | `60-80` | 13,696 | 51.38 | 50.49 | 51.34 | -0.89 |
| Ornn | `80-100` | 13,774 | 52.66 | 51.69 | 50.85 | -0.97 |
| Nautilus | `0-20` | 9,401 | 47.78 | 48.83 | 49.02 | +1.05 |
| Nautilus | `20-40` | 14,275 | 46.94 | 47.72 | 48.84 | +0.78 |
| Nautilus | `40-60` | 17,799 | 47.82 | 48.56 | 49.41 | +0.74 |
| Nautilus | `60-80` | 20,243 | 49.88 | 49.24 | 49.43 | -0.64 |
| Nautilus | `80-100` | 22,170 | 51.05 | 50.63 | 48.99 | -0.42 |
| Shen | `0-20` | 5,628 | 49.02 | 48.60 | 48.46 | -0.43 |
| Shen | `20-40` | 7,547 | 47.28 | 47.30 | 48.17 | +0.02 |
| Shen | `40-60` | 9,420 | 48.79 | 48.12 | 48.85 | -0.67 |
| Shen | `60-80` | 10,401 | 48.76 | 48.47 | 48.69 | -0.29 |
| Shen | `80-100` | 11,274 | 50.51 | 49.67 | 48.22 | -0.84 |
| Poppy | `0-20` | 4,407 | 49.65 | 50.20 | 50.47 | +0.55 |
| Poppy | `20-40` | 5,773 | 49.64 | 49.09 | 50.51 | -0.56 |
| Poppy | `40-60` | 6,677 | 51.10 | 49.86 | 50.94 | -1.24 |
| Poppy | `60-80` | 7,547 | 50.87 | 50.45 | 50.93 | -0.41 |
| Poppy | `80-100` | 8,169 | 52.76 | 51.86 | 50.50 | -0.90 |

## MR Tanks Into Enemy Magic Share

Enemy magic share uses global quintile bands. Rows are build `mr_tank`, all
roles.

| Champion | Band | n | Emp WR | Model WR | Base WR | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Galio | `0-20` | 1,866 | 38.69 | 42.26 | 44.96 | +3.57 |
| Galio | `20-40` | 2,900 | 40.59 | 42.36 | 45.96 | +1.78 |
| Galio | `40-60` | 3,754 | 41.48 | 42.78 | 46.26 | +1.30 |
| Galio | `60-80` | 5,529 | 44.22 | 43.11 | 46.04 | -1.11 |
| Galio | `80-100` | 7,947 | 48.91 | 47.09 | 46.94 | -1.82 |
| Sion | `0-20` | 696 | 47.13 | 49.08 | 49.74 | +1.96 |
| Sion | `20-40` | 1,281 | 47.23 | 48.94 | 50.58 | +1.71 |
| Sion | `40-60` | 2,604 | 47.81 | 48.30 | 50.05 | +0.49 |
| Sion | `60-80` | 5,052 | 48.75 | 48.48 | 49.76 | -0.28 |
| Sion | `80-100` | 8,060 | 51.09 | 51.88 | 50.32 | +0.79 |
| Ornn | `0-20` | 654 | 48.47 | 49.60 | 50.08 | +1.13 |
| Ornn | `20-40` | 1,192 | 49.75 | 49.42 | 50.64 | -0.33 |
| Ornn | `40-60` | 2,674 | 48.09 | 48.94 | 50.50 | +0.85 |
| Ornn | `60-80` | 5,168 | 50.21 | 49.58 | 50.43 | -0.63 |
| Ornn | `80-100` | 8,167 | 52.58 | 52.84 | 51.13 | +0.27 |
| Chogath | `0-20` | 394 | 48.73 | 49.27 | 49.95 | +0.54 |
| Chogath | `20-40` | 651 | 44.39 | 48.50 | 50.23 | +4.11 |
| Chogath | `40-60` | 1,223 | 47.26 | 48.57 | 50.19 | +1.30 |
| Chogath | `60-80` | 2,044 | 49.07 | 49.54 | 50.45 | +0.47 |
| Chogath | `80-100` | 3,401 | 54.31 | 52.77 | 50.77 | -1.54 |
| Shen | `0-20` | 705 | 46.81 | 47.43 | 47.11 | +0.62 |
| Shen | `20-40` | 1,279 | 44.49 | 47.00 | 47.74 | +2.51 |
| Shen | `40-60` | 2,544 | 46.54 | 47.31 | 48.10 | +0.77 |
| Shen | `60-80` | 4,586 | 47.88 | 46.82 | 47.42 | -1.06 |
| Shen | `80-100` | 7,065 | 50.09 | 50.08 | 48.11 | -0.01 |
| Nautilus | `0-20` | 2,061 | 47.74 | 48.04 | 48.48 | +0.30 |
| Nautilus | `20-40` | 3,839 | 48.14 | 48.06 | 49.22 | -0.08 |
| Nautilus | `40-60` | 5,469 | 48.20 | 48.23 | 49.55 | +0.03 |
| Nautilus | `60-80` | 8,546 | 48.17 | 47.65 | 48.77 | -0.52 |
| Nautilus | `80-100` | 13,060 | 49.86 | 50.42 | 49.35 | +0.56 |
| DrMundo | `0-20` | 1,305 | 51.03 | 49.46 | 49.01 | -1.58 |
| DrMundo | `20-40` | 2,013 | 50.27 | 49.31 | 49.62 | -0.96 |
| DrMundo | `40-60` | 3,329 | 48.96 | 49.91 | 50.00 | +0.94 |
| DrMundo | `60-80` | 5,344 | 49.08 | 50.11 | 49.69 | +1.03 |
| DrMundo | `80-100` | 8,388 | 52.06 | 53.69 | 50.22 | +1.63 |
| Amumu | `0-20` | 294 | 49.66 | 49.60 | 51.56 | -0.06 |
| Amumu | `20-40` | 512 | 49.02 | 49.67 | 52.03 | +0.65 |
| Amumu | `40-60` | 754 | 48.01 | 48.77 | 51.62 | +0.76 |
| Amumu | `60-80` | 1,233 | 50.12 | 48.75 | 51.22 | -1.38 |
| Amumu | `80-100` | 2,102 | 49.86 | 51.87 | 51.49 | +2.01 |
| Maokai | `0-20` | 231 | 49.78 | 49.29 | 49.66 | -0.49 |
| Maokai | `20-40` | 414 | 53.14 | 49.96 | 51.08 | -3.18 |
| Maokai | `40-60` | 689 | 50.07 | 49.30 | 50.56 | -0.77 |
| Maokai | `60-80` | 1,170 | 49.91 | 49.30 | 50.14 | -0.61 |
| Maokai | `80-100` | 1,918 | 51.30 | 52.60 | 50.73 | +1.30 |
| Malphite | `0-20` | 630 | 48.89 | 46.14 | 47.23 | -2.75 |
| Malphite | `20-40` | 1,210 | 46.12 | 45.57 | 47.59 | -0.55 |
| Malphite | `40-60` | 2,717 | 47.00 | 44.88 | 47.51 | -2.12 |
| Malphite | `60-80` | 5,955 | 45.10 | 44.55 | 47.10 | -0.55 |
| Malphite | `80-100` | 10,282 | 46.90 | 46.92 | 47.27 | +0.02 |
