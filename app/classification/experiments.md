# Classification Optimisation Experiments

Goal: find `(championid, teamposition, build)` groups that are semantically
readable in League of Legends, especially temporal identities such as late
tanks, early skirmishers, scaling carries, phase-dependent utility picks, and
specialised off-meta role/build rows.

## Run Loop

```bash
uv run python -m app.classification.embeddings.pipeline
uv run python -c 'from pathlib import Path; import app.classification.embeddings.experiments as ex; ex.RAW_CACHE_PATH = Path("/tmp/embed_exp/raw_levels_fresh_20260526.pkl"); ex.run_sweep(output_path=Path("/tmp/embed_exp/scorecard_fresh_20260526.txt"))'
```

If `6010` or the `9000-9040` prior tables were rebuilt, use a fresh
`RAW_CACHE_PATH`; the experiment harness caches raw rows under `/tmp`.

The HTML report is a single-threshold inspection lens. Use
`similarity_threshold` to choose the lens, then open
`app/classification/data/embeddings/groupings_report.html`. Threshold
comparisons should stay in the scorecard rather than being rendered as multiple
threshold sections in the report.

## Replacement Rule

A challenger can replace the active base only if it clears all required
minimums against the current base:

- `net_score`: at least `+0.03` absolute and `+15%` relative improvement.
- Semantic diagnostics: either semantic same-group rate improves by `+0.05`
  absolute or `t75` improves by `+0.05` absolute.
- Guardrails: anti-pair same-group leakage must not increase by more than
  `+0.02`. Worst group minimum pairwise similarity and low-sample dominance
  are hard guardrails for local/raw lenses, but become audit triggers for
  broad-axis derived lenses: a lower `wmin` or low-sample increase above
  `+0.03` is eligible only when the affected large groups have explicit
  role/build reads and the tail groups remain semantically coherent.
  Largest-group size is never a hard rejection by itself. Any group above
  roughly `18%` of identities must have an explicit broad-axis read, such as
  AP/AP-off-tank, AD/crit/on-hit/lethality, tank, jungle, or support utility,
  with build/role composition supporting that read.
- Manual eligibility: less popular groups outside the top 15 must include
  coherent specialised reads, not only leftovers from over-compression.

Use scalar scores to shortlist candidates, then choose by guardrails plus
semantic inspection. Keep the newest base model first. Record at most five
base models; when a sixth base would be added, remove the oldest ledger row.

## Base Model Ledger

Keep the newest base model first. Retain at most five rows.

| Date | Base | Config | Key diagnostics | Decision |
| --- | --- | --- | --- | --- |
| 2026-05-26 | `base-003` | raw + ratio-derived features, `projection_keep_variance=0.91`, `extreme_low_sample_threshold=50`, `similarity_threshold=0.82` | groups `24`, largest `494`, quality `24.79`, worst min sim `0.362`, low-sample dominance `0.51`, semantic same-group `0.71`, anti `0.19`, `t75=0.77`, net `0.28` | Active base. Replaces `base-002`: net `+0.10` (`+56%`), quality `+103%`, semantic same-group `+0.17`, `t75 +0.07`, anti unchanged. Accepted under broad-axis audit: lower worst min sim and low-sample `+0.04` are concentrated in interpretable broad groups, with coherent less-popular groups. |
| 2026-05-26 | `base-002` | raw features, `projection_keep_variance=0.91`, `extreme_low_sample_threshold=50`, `similarity_threshold=0.82` | groups `69`, largest `387`, quality `12.23`, worst min sim `0.440`, low-sample dominance `0.47`, semantic same-group `0.54`, anti `0.19`, `t75=0.70`, net `0.18` | Previous raw base. Replaced `base-001`: net `+0.07`, quality `+25%`, semantic same-group `+0.15`, `t75 +0.13`, anti unchanged, worst min sim slightly improved. |
| 2026-05-26 | `base-001` | raw features, `projection_keep_variance=0.92`, `extreme_low_sample_threshold=100`, `similarity_threshold=0.82` | groups `83`, largest `365`, quality `9.75`, worst min sim `0.430`, low-sample dominance `0.46`, semantic same-group `0.39`, anti `0.19`, `t75=0.57`, net `0.11` | Previous conservative base. |

## Latest Deep Evaluation

Fresh focused grid, using `/tmp/embed_exp/raw_levels_fresh_20260526.pkl` and
the corrected temporal tables. The scorecard now reports `wmin` and
low-sample dominance directly, and `net_score` no longer penalises group size
by itself.

| Candidate | Groups | Largest | Quality | `wmin` | Low-sample dom. | Semantic | Anti | `t75` | Net | Eligibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base-003 default_plus_ratios` | 24 | 494 | 24.79 | 0.362 | 0.51 | 0.71 | 0.19 | 0.77 | 0.28 | Winner: best eligible balance with anti unchanged; promoted after broad/tail audit. |
| `champion_role_heavy + ratios` | 25 | 543 | 19.55 | 0.208 | 0.51 | 0.75 | 0.19 | 0.82 | 0.29 | Near miss: net only `+0.01` over active base and `wmin` falls too far. |
| `kv0.88_ratios_low50` | 18 | 748 | 24.66 | 0.228 | 0.51 | 0.79 | 0.25 | 0.82 | 0.29 | Rejected for promotion: anti leakage `+0.06` and worst-pair coherence too weak despite readable large groups. |
| `ratios_low100` | 24 | 495 | 26.61 | 0.437 | 0.50 | 0.61 | 0.19 | 0.72 | 0.22 | Rejected: quality/wmin improved, but semantic recall and `t75` fell below active base. |
| `lean_raw_only` | 33 | 494 | 18.08 | 0.260 | 0.49 | 0.66 | 0.19 | 0.77 | 0.24 | Audit lens only: useful broad raw compression, but worst-pair drop is not justified by enough semantic lift over ratios. |
| `kv0.92_ratios_low50` | 34 | 494 | 17.23 | 0.330 | 0.50 | 0.64 | 0.19 | 0.75 | 0.23 | Audit lens only: ratios help, but `kv=0.91` is stronger. |
| `kv0.88_raw` | 35 | 529 | 20.12 | 0.321 | 0.50 | 0.62 | 0.19 | 0.68 | 0.24 | Audit lens only: broad groups are readable, but lower `t75` than active base. |
| `base-002 raw kv0.91_low50` | 69 | 387 | 12.23 | 0.440 | 0.47 | 0.54 | 0.19 | 0.70 | 0.18 | Replaced by `base-003`. |
| `support_control` / `niche_core` families | 7-17 | 1573-2076 | 0.00-12.59 | 0.21-0.44 | 0.53-0.54 | 0.84-0.91 | 0.62 | 0.95 | 0.09-0.14 | Rejected: very high same-group rates came from collapse and anti leakage, not useful specialisation. |

### Lever Findings

1. Feature set: ratio-derived features are now production. They add damage mix,
   damage/gold, damage taken/gold, neutral-farm share, mitigation ratio, and
   champion-damage focus. This produced the largest diagnostic gain without
   increasing anti-pair leakage.
2. PCA: `0.91` remains the production balance for the ratio feature set.
   `0.88` improved recall but raised anti leakage to `0.25`; `0.92` lost
   semantic lift and quality.
3. Low-sample threshold: `50` remains the production setting. `100` improved
   quality and `wmin`, but dropped semantic recall to `0.61` and `t75` to
   `0.72`.
4. Prior weights: champion-role-heavy priors produced the top scalar net
   (`0.29`) but only by `+0.01` over active base and with `wmin=0.208`, so it
   does not clear the replacement rule. Role/build-heavy still over-compresses.
5. Niche feature families: support/control, econ/objective, and broad
   niche-core sets were too strong as standalone pools. They created excellent
   apparent recall by collapsing into a few enormous groups and raising anti
   leakage. Keep them out of the default grid until they are split into
   narrower, hypothesis-driven probes.
6. Similarity threshold: `0.82` remains the report threshold. `0.80` broadens
   the ratio lens but weakens `wmin`; `0.84` tightens groups without improving
   the key semantic diagnostics.

### Less Popular Group Audit

The promoted `base-003` was checked below the top groups to confirm specialised
nuance:

| Group | Size | Median / min sim | Read |
| ---: | ---: | --- | --- |
| 15 | 50 | `0.920 / 0.593` | Pantheon/Ashe/Vi/Lucian/Pyke/Draven pick-pressure rows across lethality, on-hit, and AD builds. |
| 16 | 49 | `0.987 / 0.920` | Ivern/Taric-led JUNGLE utility/protection rows with tightly grouped rare supportive jungle builds. |
| 17 | 46 | `0.923 / 0.731` | AP/AP-off-tank TOP/MID control and engage picks: Nunu, Nautilus, Viktor, Lissandra, Sejuani, Ryze. |
| 18 | 41 | `0.940 / 0.770` | Off-meta AD UTILITY pick-pressure identities: Viego, Bel'Veth, Kayn, Shaco, Rek'Sai, Blitzcrank. |
| 20 | 24 | `0.943 / 0.714` | Rare UTILITY crit/marksman support rows such as Graves, Draven, Quinn, Tristana, Jhin, Kindred. |
| 21 | 18 | `0.909 / 0.679` | Small melee skirmisher/off-tank pocket with Yasuo, Yone, Sett, Viego, Jarvan, Shen, Fizz. |
| 22 | 13 | `0.915 / 0.696` | Senna/Nilah AD/crit/on-hit variants isolated as a specialised carry-support/bot-lane identity. |
| 23 | 4 | `0.972 / 0.947` | Fiddlesticks TOP/MID AP/AP-off-tank variants grouped as champion-specific off-meta nuance. |
| 24 | 2 | `0.969 / 0.969` | Dr. Mundo TOP/MID off-tank variants grouped tightly. |

This passes manual eligibility: the tail groups are not random fragments; they
capture off-meta role/build rows and champion-specific variants that a broader
archetype-only model would flatten.

### Large Group Audit

The largest `base-003` groups are broad, but their composition is contextual:

| Group | Size | Median / min sim | Read |
| ---: | ---: | --- | --- |
| 1 | 494 | `0.950 / 0.660` | JUNGLE broad axis across AP, tank, AD, crit, lethality, and on-hit jungle rows. |
| 2 | 375 | `0.928 / 0.404` | UTILITY support-control axis across tank, protection, enchanter, and AP support rows. |
| 3 | 351 | `0.919 / 0.476` | TOP/MID/BOTTOM physical carry axis across crit, AD, on-hit, lethality, and off-tank builds. |
| 4 | 347 | `0.901 / 0.362` | AD bruiser/skirmisher axis across TOP/MID/BOTTOM, led by Irelia, Kled, Olaf, Volibear, Tryndamere, Aatrox, Darius. |
| 5 | 249 | `0.910 / 0.474` | AP/AP-off-tank mage and carry axis across BOTTOM/MID/TOP. |

## Iterations

1. PCA sweep: lower PCA should improve temporal recall, but success requires
   anti-pair leakage near or below 20% unless a specific audit hypothesis is
   being tested.
2. Low-sample threshold sweep: success is fewer rare-row artefacts and stable
   temporal pairs, not simply fewer singletons.
3. Prior weighting sweep: role/build priors should reveal cross-champion
   archetypes; sibling or champion priors should not collapse everything into
   champion-local aliases.
4. Raw/derived sweep: derived metrics are useful only if manual group reads
   stay coherent. Treat very high recall plus anti leakage or unnameable
   groups as overcompression.
5. Similarity threshold sweep: choose the threshold that makes groups
   inspectable. Success means temporal opposites still separated; largest
   group size is audited by semantic read instead of capped.
6. Manual temporal audit: inspect `groupings_report.html` and the temporal pair
   checks. A winning run must produce groups that can be named without forcing
   the interpretation.

## Success

A successful iteration has:

- Coherent group reads with at least a few surprising but sensible identities.
- Temporal anchors preserved: late tanks and early skirmishers cluster locally;
  opposite temporal directions do not cluster.
- Anti-pair leakage around 20% or lower.
- No unnameable mega-cluster dominating the report.
- Few low-sample artefacts and few meaningless singletons.

Use scalar scores to shortlist candidates, then choose by semantic inspection.
