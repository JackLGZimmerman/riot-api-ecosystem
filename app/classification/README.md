# Classification Embeddings

Fixed-size, L2-normalised embeddings for each `(championid, teamposition,
build)` identity. The source is `game_data_filtered.synergy_1vx_temporal`
(`6010`), smoothed with the temporal prior tables (`9000-9040`). Similarity is
cosine similarity over the final embedding vectors.

Run:

```bash
uv run python -m app.classification.embeddings.pipeline
uv run python -m app.classification.embeddings.experiments
```

The current production default is raw metrics plus the ratio-derived feature
subset, `projection_keep_variance=0.91`, `extreme_low_sample_threshold=50`,
and `similarity_threshold=0.82`.

## Evaluation Snapshot

`6010` was rebuilt on 2026-05-26 after the temporal aggregation schema was
corrected. The production snapshot below is from a fresh pipeline rerun against
that table.

| Metric | Current production default |
| --- | ---: |
| PCA axes | 6 |
| Identities | 3128 |
| Groups (non-singleton + singleton) | 24 (24 + 0) |
| Largest group | 494 |
| Coverage (in a non-singleton) | 100.0% |
| Trimmed quality (`mnsx * coverage * diversity`) | 24.79 |
| Worst group minimum pairwise similarity | 0.362 |
| Low-sample dominance | 0.51 |
| TOP/MID mixed identity share | 100% |
| Semantic same-group rate | 71% |
| Anti-pair same-group leakage | 19% |
| Semantic separation | 0.526 |
| `t75` | 0.775 |

The `0.82` threshold remains the inspection lens. The selected embedding
(`raw + ratios, kv=0.91, low_sample=50`) improves semantic recall and quality
without increasing anti-pair leakage. The lower worst within-group minimum is
accepted under the broad-axis audit because the affected groups have clear
role/build reads and the less-popular groups remain specialised. The HTML
report renders only this active threshold; threshold comparisons belong in the
experiment scorecards.

## Optimisation Outcome

The sweep should not be read as "highest score wins", and large clusters should
not be treated as failures by size alone. A 700+ identity group can be useful
when it clearly represents a generic axis, such as all TOP/MID/BOTTOM physical
carry rows across `attack_damage`, `ad_off_tank`, `crit`, `on_hit`, and
`lethality`, or a broad high-magic-damage family across `ability_power` and
`ap_off_tank`. Large groups become a problem only when they cannot be named
without hand-waving, mix unrelated role/build semantics, or inflate diagnostics
without an interpretable League pattern.

For the current goal, semantic value means groups that can be read as coherent
League identities after inspection, with temporal opposites still separated.

| Lever | Best signal from the fresh sweep | Decision |
| --- | --- | --- |
| Feature set | Raw + ratios reached quality `24.79`, semantic same-group `0.71`, anti leakage `0.19`, and `t75=0.77`. The lift came from damage mix, damage/gold, damage-taken/gold, mitigation, neutral-farm share, and champion-damage focus. | Promote ratio-derived features to production. |
| PCA | With ratios, `0.91` is the production balance. `0.88` increased semantic recall to `0.79`, but anti leakage rose to `0.25` and `wmin` fell to `0.228`. `0.92` was safer than `0.88` but weaker than `0.91`. | Keep `0.91`; use lower PCA only for broad audit lenses. |
| `low_sample_threshold` | `50` preserves rare-row signal and keeps anti leakage at `0.19`. Higher values did not improve the promoted feature set, and `75+` reduced semantic hits or raised anti leakage in raw sweeps. | Keep `50`; it stabilises rare rows without erasing specialised off-meta signal. |
| Prior weights | Role/build-heavy priors still find broad archetypes, but do not beat raw + ratios. Sibling/champion-role variants are useful only for narrow hypotheses. | Keep the default role/build-dominant prior balance. |
| Niche derived pools | Support/control, econ/objective, and broad niche-core pools produced high apparent recall by collapsing into very few huge groups and raising anti leakage as high as `0.62`. | Trim them from the default grid until split into narrower probes. |
| Similarity threshold | `0.80` broadens the ratio lens but weakens `wmin`; `0.84` tightens without better semantics. | Use `0.82` for reported groups. |
| Diagnostics | `min_group_sim`, anti-pair leakage, low-sample dominance, and manual group reads caught failures that `net_score` alone missed. Largest-group size is an inspection trigger, not an automatic rejection: a huge group must declare a clear semantic axis and have build/role composition that supports that read. | Optimise by qualitative read plus guardrails, not by a single scalar. |

### Large Group Audit

Large clusters are eligible when their composition supports a plain semantic
read. The promoted ratio run was checked specifically because it prefers broad
contextual axes:

| Run / group | Size | Median / min sim | Composition signal | Read |
| --- | ---: | --- | --- | --- |
| base-003 `#1` | 494 | `0.950 / 0.660` | JUNGLE; AP, tank, AD, crit, lethality, on-hit | broad jungle context axis |
| base-003 `#2` | 375 | `0.928 / 0.404` | UTILITY; tank, protection, enchanter, AP support rows | support utility / control axis |
| base-003 `#3` | 351 | `0.919 / 0.476` | TOP/MID/BOTTOM; `crit`, `attack_damage`, `on_hit`, `lethality`, `ad_off_tank` | physical carry / marksman axis |
| base-003 `#4` | 347 | `0.901 / 0.362` | TOP/MID/BOTTOM; mostly `ad_off_tank`, `attack_damage`, `on_hit` | AD bruiser / skirmisher axis |
| base-003 `#5` | 249 | `0.910 / 0.474` | BOTTOM/MID/TOP; mostly `ability_power`, `ap_off_tank` | AP carry / AP off-tank axis |

These are interpretable broad buckets, not context-free inflation. The lower
worst-pair similarity is accepted because it is concentrated in these broad
groups, while anti-pair leakage remains unchanged and tail groups retain
specialised meaning.

The production model is therefore a broad archetype extractor with specialised
tail behaviour: it captures temporal identity when the temporal profile agrees
with role/build and stat shape, but it does not group on temporal direction
alone. If the next goal becomes pure "scaling identity discovery", the next
experiment should add explicit phase delta/slope features from
`8005_build_temporal_win_rate.sql` rather than relying only on the four raw
phase snapshots.

## Current Top Groups

Current default, threshold `0.82`:

| Rank | Size | Median sim | Roles | Builds | Top champions | Read |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 494 | 0.950 | JUNGLE | ability_power, ar_tank, ad_off_tank, crit, attack_damage | Shaco, Udyr, Shyvana, Volibear, MasterYi, Hecarim, Kindred, Viego | broad jungle context |
| 2 | 375 | 0.928 | UTILITY | ar_tank, utility_protection, utility_enchanter, mr_tank, ability_power | Senna, Ashe, Bard, Nami, Lulu, Seraphine, Zilean, Sona | support / utility-control |
| 3 | 351 | 0.919 | BOTTOM / MIDDLE / TOP | crit, attack_damage, on_hit, lethality, ad_off_tank | Varus, Vayne, Quinn, Akshan, Kaisa, Kalista, Jayce, MissFortune | physical carry / marksman |
| 4 | 347 | 0.901 | TOP / MIDDLE / BOTTOM | ad_off_tank, attack_damage, on_hit, crit, lethality | Irelia, Kled, Olaf, Volibear, Tryndamere, Aatrox, Zaahen, Darius | AD bruiser / skirmisher |
| 5 | 249 | 0.910 | BOTTOM / MIDDLE / TOP | ability_power, ap_off_tank | Kayle, Ahri, AurelionSol, Zyra, Velkoz, Taliyah, Veigar, Brand | AP carry / mage |
| 6 | 184 | 0.910 | MIDDLE / TOP / BOTTOM | ar_tank, mr_tank, utility_enchanter, utility_protection | Shen, Nautilus, Volibear, Nunu, Rammus, Thresh, Taric, Ornn | tank / engage |
| 7 | 148 | 0.939 | UTILITY | ability_power, ap_off_tank | Swain, Zoe, Velkoz, Brand, Mel, Annie, Xerath, Ahri | AP support mage |
| 16 | 49 | 0.987 | JUNGLE | utility_protection, utility_enchanter | Ivern, Taric, Volibear, Zac, Nunu, Amumu, Shaco, Malphite | utility/protection jungle niche |
| 20 | 24 | 0.943 | UTILITY | crit | Graves, Draven, Quinn, Sion, Yasuo, Tristana, Jhin, Kindred | rare crit support niche |

## Temporal Update

Temporal checks now come from `8005_build_temporal_win_rate.sql` instead of
hand-picked examples. Rows were ranked by `late - early_mid` win-rate delta,
requiring at least 500 raw matchups in both endpoint bins.

### Source Signals

| Direction | Identity | Early matchups | Early WR | Late matchups | Late WR | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| late-positive | Ornn TOP ar_tank | 4000 | 0.390 | 6409 | 0.554 | +0.164 |
| late-positive | Camille TOP ad_off_tank | 1949 | 0.353 | 1436 | 0.496 | +0.143 |
| late-positive | Thresh UTILITY mr_tank | 946 | 0.390 | 1383 | 0.531 | +0.141 |
| late-positive | Shen TOP mr_tank | 1054 | 0.365 | 1586 | 0.501 | +0.136 |
| late-positive | Malphite TOP ability_power | 811 | 0.365 | 1088 | 0.489 | +0.124 |
| late-positive | Twitch BOTTOM crit | 4807 | 0.453 | 7639 | 0.555 | +0.103 |
| early-positive | Yorick TOP attack_damage | 614 | 0.635 | 1117 | 0.448 | -0.188 |
| early-positive | Irelia TOP ad_off_tank | 1452 | 0.629 | 1260 | 0.457 | -0.172 |
| early-positive | Irelia MIDDLE on_hit | 4170 | 0.593 | 2223 | 0.430 | -0.163 |
| early-positive | Senna UTILITY crit | 862 | 0.678 | 3202 | 0.529 | -0.148 |
| early-positive | Ziggs BOTTOM ability_power | 3478 | 0.605 | 6050 | 0.468 | -0.137 |
| early-positive | Yasuo MIDDLE crit | 7503 | 0.628 | 11088 | 0.498 | -0.131 |

### Embedding Confirmation

Current default, threshold `0.82`:

| Check | Pair | Cosine sim | Same group? | Read |
| --- | --- | ---: | --- | --- |
| late-positive local | Ornn TOP ar_tank ↔ Ornn TOP mr_tank | 0.997 | yes | tank build variants collapse |
| late-positive local | Shen TOP ar_tank ↔ Shen TOP mr_tank | 0.997 | yes | tank build variants collapse |
| late-positive local | Galio MID ar_tank ↔ Galio MID mr_tank | 0.992 | yes | tank build variants collapse |
| late-positive cross | Ornn TOP ar_tank ↔ Shen TOP ar_tank | 0.929 | yes | broad tank axis now groups cross-champion tanks |
| late-positive ADC | Twitch BOT crit ↔ Jhin BOT crit | 0.919 | yes | broad physical-carry axis now groups late ADC rows |
| early-positive local | Yorick TOP AD ↔ Yorick TOP off-tank | 0.989 | yes | split-push build variants collapse |
| early-positive local | Irelia TOP on-hit ↔ Irelia MID on-hit | 0.999 | yes | role variants collapse |
| early-positive skirmisher | Irelia TOP on-hit ↔ Tryndamere TOP crit | 0.942 | yes | shared early-pressure skirmisher profile |
| early-positive skirmisher | Yasuo MID crit ↔ Yone TOP crit | 0.987 | yes | shared early-pressure skirmisher profile |
| opposite direction | Ornn TOP ar_tank ↔ Yorick TOP AD | 0.395 | no | temporal/archetype opposites separate |
| opposite direction | Twitch BOT crit ↔ Yasuo MID crit | 0.682 | no | late ADC vs early skirmisher separate |

The embedding confirms temporal identities when the row also shares
role/build/stat structure. Temporal direction alone is not used as a grouping
axis, which is the desired behavior for the main classifier.
