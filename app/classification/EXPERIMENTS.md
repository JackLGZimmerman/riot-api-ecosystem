# Classification Tuning

**Goal**: tune specialists for semantic coherence, metric recoverability, and
stable grouping behavior.

This file documents evaluation methods plus dated promotion records for
specialist tuning passes. Keep records short and reproducible: include the
command family, PCA boundary, group-shape evidence, semantic read, and rejected
nearby alternatives.

## Run Loop

```bash
# Production: rebuild specialists end-to-end and write the HTML report.
uv run python -m app.classification.embeddings.pipeline

# Sweep one specialist.
uv run python -m app.classification.embeddings.tune \
    --name <specialist> --kvs <kv...> --ts <threshold...>

# Inspect the picked candidate.
uv run python -m app.classification.embeddings.inspection.base \
    --name <specialist> --kv <kv> --t <threshold>

# Inspect a feature/threshold variant without editing config.
uv run python -m app.classification.embeddings.inspection.base \
    --name <specialist> --kv <kv> --t <threshold> \
    --features <feature1> <feature2> <feature3>

# Compare raw-vs-transformed replacement metrics.
uv run python -m app.classification.embeddings.inspection.base \
    --name <specialist> \
    --compare-features <raw_feature> <transformed_feature> \
    --denominator-check <numerator_feature> <denominator_feature>
```

The HTML report under `app/classification/data/embeddings/` is the production
view for the picked config. The CLI inspector is the faster debugging view:
retained PCA axes, feature z-scores, and build/role/champion composition.
The inspector lives at `app/classification/embeddings/inspection/base.py`.
Use `inspection/specialist_configs.txt` to preserve tuned per-specialist CLI
arguments.

`tune.py` reuses `/tmp/embed_exp/raw_levels.pkl`. Move or delete that cache
after rebuilding `6010` or `9000-9040`; a stale cache can fail with missing
columns. Adding a column to `ALL_METRICS` in `config.py` requires the source
column to exist in `6010` and the `9000-9040` priors first, or load fails with
`UNKNOWN_IDENTIFIER`.

## Required Evidence

Every candidate configuration evaluation needs these checks:

1. **PCA evidence**: retained axes, explained variance, and top loadings.
2. **Group evidence**: per-phase group counts, coverage, largest-group share,
   and median within-group cosine.
3. **Semantic evidence**: top z-scores plus champion/build/role composition for
   every retained group that affects the decision.
4. **Stability evidence**: nearby `t`/`kv` variants do not flip the semantic
   read unless the change is intentionally selecting a new axis.
5. **Replacement evidence**: when a derived or normalised feature stands in for
   an absolute metric, run tail correlation and top/bottom-50 Jaccard checks.

## 2026-05-29 Adaptive Prior Cascade Smoothing (before/after)

**Hypothesis (from the ML side):** the legacy `additive` smoothing in
`posteriors.py` pools every prior level (`sibling`, `champion_role`,
`role_build`, `champion_build`, `build`) into one weighted mixture. Even a
well-sampled specific prior is then contaminated by broad ones. The fix is an
adaptive cascade: shrink toward only the highest-priority prior level whose own
sample size clears a confidence threshold; broader levels are fallback only.
This is validated here before any ML change (see
[../ml/documentation/README.md](../ml/documentation/README.md)).

**Why classification first.** Most 1vx identities are low-sample (baseline
`matchups` median 9, p25 = 2), so the prior choice dominates the embedding. The
grouping rubric below scores the effect without retraining the win-rate model.

**Sampling threshold (assumption).** "Sufficient sampling" = the prior level's
own `matchups >= prior_confidence_matchups` (`tau`). The gate is on the prior's
matchups, metric-independent, evaluated in `PRIOR_LEVELS` contextual-relevance
order; the first level clearing `tau` is selected, the broadest valid level is
the fallback. Selection is mutually exclusive (every row picks exactly one
level); at `tau=50`, sibling 23%, champion_role 60%, role_build 16%,
champion_build 0.4%, build 0% (never needed).

**Modes** (`EmbeddingConfig.smoothing_mode`, `cascade_match_weight`):

- `additive` — pooled value, pooled weight (legacy).
- `cascade` — single selected value, that level's weight only (less total
  shrinkage).
- `cascade` + `cascade_match_weight` — single selected value, but the additive
  mixture's *total* weight. Controls for shrinkage magnitude so the only change
  is prior *value* (single vs pooled): the clean test of contamination.

**Evaluation.** Each of the 24 specialists scored at its *promoted* (kv, t) via
`inspection/registry_audit.sweep_specialist` (same scoring used for tuning).
`mean_score` excludes the coverage/budget sentinel penalties; `n` is specs not
penalised. Harness: `/tmp/embed_exp/eval_smoothing.py`.

| Smoothing | mean_score | median | cov | med_cos | mean top‑\|z\| | penalised | over‑budget | weak | small |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `additive` (before) | 4.509 (n=23) | 4.677 | 0.993 | 0.9773 | 1.088 | 1 (`vision`) | 0 | 0 | 12 |
| `cascade` tau=30 | 4.524 (n=23) | 4.622 | 1.000 | 0.9779 | 1.145 | 1 (`takedown_shape`) | 4 | 5 | 37 |
| `cascade` tau=50 | 4.530 (n=23) | 4.646 | 1.000 | 0.9757 | 1.152 | 1 (`takedown_shape`) | 4 | 3 | 44 |
| `cascade` tau=100 | 4.553 (n=24) | 4.662 | 1.000 | 0.9806 | 1.149 | 0 | 0 | 4 | 23 |
| `cascade`+match tau=50 | 4.543 (n=24) | **4.701** | 1.000 | **0.9817** | **1.159** | 0 | 0 | 3 | 27 |
| `cascade`+match tau=100 | **4.560** (n=24) | 4.680 | 1.000 | 0.9810 | 1.149 | 0 | 0 | 4 | 28 |

**Reads.**

1. Plain `cascade` at low `tau` over-fragments (`takedown_shape` → [18,18,16,16]
   vs budget 9). This is *under-shrinkage*, not contamination: matching the
   weight fixes it ([6,7,6,7]). Raising `tau` to 100 also fixes it by routing
   more rows to broader, higher-sample priors.
2. The controlled test (`cascade`+match) isolates contamination: at identical
   shrinkage magnitude, using the single highest-confidence prior value instead
   of the pooled value sharpens group signatures (mean top‑\|z\| 1.088 → 1.159,
   +6.5%), tightens within-group cosine (0.9773 → 0.9817), and gives full
   coverage with 0 over-budget and 0 penalised specs.
3. `vision` is the clearest win: under `additive` its weak late group is dropped
   (coverage 0.83, penalised); under every cascade it reaches full coverage as a
   coherent group.
4. Per-spec mean Δ(top‑\|z\|) = **+0.072** (`additive` → `cascade`+match tau=50).
   Largest gains: `farming` +0.31, `jungle_control` +0.26, `utility_pickmaking`
   +0.21, `epic_objectives` +0.20, `map_control` +0.18, `damage_efficiency`
   +0.14. Many specialists *consolidate* into fewer, tighter groups (`farming`
   4→3, `damage_profile` 7→5, `economy_scaling` 8→6). Minor regressions:
   `sustained_damage` −0.03, `defensive_statline` −0.02, `resistances` −0.01.

**Conclusion.** A consistent but modest improvement. It supports the
contamination hypothesis: removing broad-prior contamination yields sharper,
tighter, more consolidated groups and recovers a previously-dropped group, with
no over-budget or coverage failures. Adopted as the default: `smoothing_mode =
"cascade"`, `cascade_match_weight = True`, `prior_confidence_matchups = 50`.

**Migration (default flipped to cascade).** `EmbeddingConfig` now defaults to the
cascade, and `SPECIALISATIONS.md` is regenerated under it. One targeted re-tune
was needed: `sustained_damage` `t` 0.65 → 0.55 (clears its small-group `Watch`).
After that, the regenerated audit is **467 Excellent / 3 Watch / 3 Weak** across
473 retained groups, all 24 specialists OK and at full coverage. The 6 residuals
are intentional and match the project's own rubric, not tuning failures:

- `enchanters` 3 Weak — the high/low spec's large low pool is a no-read baseline
  (share ~0.84, max \|z\| 0.31–0.34, a hair under the 0.35 bar). The cascade
  sharpened the enchanter group, which flattened the majority pool. No (kv, t)
  changes this (it is single-axis high/low); valid per the "No-Read Pools" rule,
  like the `vision` late-phase exception.
- `map_control` 3 Watch — small but coherent macro reads (size 7–30, median
  cosine 0.96–1.00, max \|z\| 0.55–0.67). Valid per "small coherent groups are
  valid specialist reads; there is no size floor". Lowering `t` only trades one
  `Watch` for a lost read, so the config is left at `t=0.72`.

`additive` remains available via config for comparison/rollback.

**ML carry-over (done).** The per-side fallback was carried into `app/ml`: each
under-sampled `1v1`/`2vx` pair is now shrunk toward a composite of its two
sides' solo priors (`0.5 + (wr_blue - wr_red)/2` for matchups; the pair-average
for synergies) instead of a flat `0.5`. Testing there surfaced a second, larger
issue: the 45 interaction features were overfitting an **unregularised**
logistic regression (train 0.70 / val,test 0.53, intercept −48). On the full
1.95M-game cache:

| ML config | val acc | test acc | val auc | val tail-ECE |
| --- | ---: | ---: | ---: | ---: |
| `1vx`-only baseline | ~0.57 | ~0.57 | ~0.59 | — |
| interactions, no L2 (before) | 0.534 | 0.534 | 0.544 | 0.317 |
| interactions, L2 only | 0.568 | 0.570 | — | — |
| interactions, L2 + per-side fallback | **0.569** | **0.570** | **0.596** | **0.112** |

L2 is the dominant fix (recovers to the baseline); the per-side fallback adds a
small consistent AUC/ranking gain on top — the same modest-but-consistent
pattern the cascade showed here. The overfit gap collapsed (0.167 → 0.065) and
calibration improved sharply. See
[../ml/documentation/README.md](../ml/documentation/README.md).

### 1v1 / 2vx extension assessment

The classification grouping pipeline embeds the full ~60-metric *behavioural*
profile of single identities from `6010`. `matchup_1v1` and `synergy_2vx` carry
only `matchups`/`wins`/`win_rate` — there is no per-pair behavioural suite to
embed, so interaction pairs cannot be grouped by the same machinery. Extending
the *grouping* eval to 1v1/2vx is therefore a category mismatch and was not
built; the transferable piece is the cascade + per-side fallback applied to
interaction *win-rate* smoothing, which belongs to the ML pipeline
(`build_dataset.py`, currently a flat shrink toward 0.5 with `prior_strength=20`).

**Efficient prior generation.** The `9000-9040` tables are all plain `GROUP BY`
rollups of `6010` (e.g. `9010` aggregates across builds), and `load.py` already
derives them on the fly when a table is missing. So the five materialised prior
tables can be replaced by one rollup query (or `GROUPING SETS`) over the base
aggregation. The same pattern extends to interactions: given one base
interaction aggregation, every side/level prior is a `GROUP BY` over it — no new
per-level materialised tables required.

## 2026-05-29 Full Registry Quality Audit

> Historical record. This audit was run under `additive` smoothing. The registry
> has since migrated to the cascade (see the Adaptive Prior Cascade section
> above); the regenerated `SPECIALISATIONS.md` now carries 3 intentional `Weak`
> (`enchanters` no-read baseline) and 3 intentional `Watch` (`map_control` small
> coherent reads), and `sustained_damage` moved to `t=0.55`. The configs and
> "no Watch/Weak/Reject/OVER" result below describe the pre-cascade state.

Target specialists: every active `SpecialistSpec` and `SingularMetricSpec`.
This pass supersedes the 2026-05-28 promotion notes where they conflict with
the active registry.

Commands used:

```bash
uv run python -m app.classification.embeddings.inspection.specialisations_markdown

uv run python -m app.classification.embeddings.tune \
    --name <specialist> --kvs <nearby kv values> --ts <nearby thresholds>

uv run python -m app.classification.embeddings.inspection.base \
    --name <specialist> --kv <kv> --t <threshold> --phase <phase> \
    --features <candidate feature set>
```

Audit rubric changes:

- `SPECIALISATIONS.md` now records coverage and dropped-group counts beside
  phase-local group counts.
- Quality text is generated per retained group from metric z-score strength,
  secondary metric support, within-group median cosine, size, and whether a
  large cluster is an intentional baseline contrast.
- Singular metrics are audited by phase-local unique value counts and top/bottom
  tail composition. All six active singular metrics had dense orderings
  (`>=3092` unique values in every phase), so no clustering was added.

Promoted specialist changes:

| Specialist | Promoted Config | Evidence | Rejected Alternative |
| --- | --- | --- | --- |
| `sustained_damage` | `kv=0.80`, `t=0.65`; features `totaldamagedealttochampions`, champion damage per gold, champion damage per death, champion-damage focus | `phase_g=[6,5,5,6]`, full coverage, no dropped groups. Direct volume/focus axes removed low-coherence damage-type groups while keeping carry, support-focus, low-volume, and high-output reads. | Prior type-share set overlapped `damage_profile` and produced low-coherence mid/mid-late groups. `t=0.70` over-split early-mid into 7 groups against a 6-group budget. |
| `vision` | `kv=0.85`, `t=0.35` | `phase_g=[3,3,3,2]`. Early/mid phases retain lane-low, support-volume, and jungle ward-action reads. Late phase coverage intentionally falls to `0.30` because the weak background group is dropped instead of labelled. | `t=0.45` kept full coverage but retained a shallow early-mid ward-ratio side group with max `|z|=0.40`. |
| `farming` | `kv=0.95`, `t=0.60` | `phase_g=[4,4,4,4]`, full coverage. Merges duplicate lane-farm fragments while retaining lane-farm, no-farm support, jungle-farm, and low-lane/high-neutral side reads. | `t=0.68` left duplicate lane-farm signatures in mid-late and late. |
| `epic_objectives` | `kv=0.95`, `t=0.15` | `phase_g=[2,2,2,2]`, full coverage. Keeps a clear low/no objective baseline and a high objective-control group in every phase. | `t=0.20` split late into a lower-coherence epic-monster-share side group that did not add enough semantic separation. |
| `resistances` | `kv=0.90`, `t=0.60`; features `armor`, `magicresist` | `phase_g=[2,2,2,2]`, full coverage. Raw resistance investment gives stable high/low groups with clean role/build reads. | Armor/MR-per-gold ratios created shallow side fragments; they are not robust enough to keep as group-forming axes. |
| `ability_power` | `kv=0.90`, `t=0.70`; replace champion damage per gold with magic total-damage share | `phase_g=[2,2,2,2]`, full coverage. The spec now asks AP/magic investment only, while damage efficiency stays in `damage_efficiency`. | Including champion damage per gold leaked an efficiency axis and split utility/tank magic side groups. |
| `attack_damage` | `kv=0.85`, `t=0.65`; replace critical strike ceiling with physical total-damage share | `phase_g=[2,2,2,2]`, full coverage. Raw AD, AD per gold, physical champion share, and physical total share form clean high/low investment groups. | Critical strike ceiling produced low-median side groups and is better represented as the existing singular metric. |
| `on_hit_carry` | `kv=0.85`, `t=0.70`; replace champion damage per gold with physical total-damage share | `phase_g=[6,6,6,5]`, full coverage. Keeps no-hit, AD physical, attack-speed carry, tank physical, attack-speed AP/on-hit, and hybrid side groups without duplicate high damage-per-gold fragments. | The old damage-per-gold axis duplicated `damage_efficiency` and created repeated small high-output fragments. |

Unchanged specialists reviewed as excellent under the stricter group rubric:
`early_agency`, `durability`, `self_sustain`, `damage_profile`,
`damage_efficiency`, `burst_skirmish`, `takedown_shape`,
`utility_pickmaking`, `economy_scaling`, `jungle_control`, `structure`,
`siege_pressure`, `map_control`, `crowd_control`, `enchanters`, and
`defensive_statline`.

Final audit result:

- `SPECIALISATIONS.md` contains no `Watch`, `Weak`, `Reject`, or `OVER`
  retained-group quality outcomes.
- Every retained specialist group is within the per-phase budget or documented
  as semantically justified by the stricter quality text.
- The only dropped retained-label opportunity is intentional: `vision` late
  phase drops one weak background group rather than treating it as a meaningful
  category.

## 2026-05-28 Stat Investment Specialists

Target specialists: `resistances`, `ability_power`, and `attack_damage`.

Commands used:

```bash
uv run python -m app.classification.embeddings.tune \
    --name <specialist> --kvs <nearby kv values> --ts <nearby thresholds>

uv run python -m app.classification.embeddings.inspection.base \
    --name <specialist> --kv <kv> --t <threshold> --phase <phase>

uv run python -m app.classification.embeddings.inspection.base \
    --name <specialist> --compare-features <raw> <ratio> \
    --denominator-check <raw> goldearned --include-bottom-tails
```

### Resistances

Promoted config: `kv=0.90`, `t=0.75`.

Evidence:

- Current `kv=0.85` retained one PCA axis (`0.899`) and produced only a
  high/low split (`phase_g=[2,2,2,2]`). The high group was real, but it merged
  armor-heavy and MR-heavy identities.
- `kv=0.90` retained two axes (`0.899, 0.058`). PC1 is total resistance
  investment; PC2 separates MR-per-gold / MR-heavy identities from armor-heavy
  identities.
- `kv=0.90, t=0.75` produced `phase_g=[6,6,6,7]`, coverage `1.00`,
  largest-group share `0.74`, median within-group cosine `0.98`.
- Early-mid retained a no-read pool plus mixed tank, MR-skew, armor-skew,
  moderate MR, and moderate armor groups. Mid phase kept the same semantic
  pattern, with the extra lower-z groups acting as side-stat reads rather than
  role-only leakage.

Rejected:

- `kv=0.97` retained a third axis (`0.042`) and split early-mid into 11 groups
  with coverage `0.87`; many were tiny threshold fragments with duplicated
  resistance z-score reads.
- `t=0.80` had clean medians but over-split the high-resistance mass compared
  with `t=0.75`.

Replacement check:

- `armor_to_goldearned_ratio` and `magicresist_to_goldearned_ratio` are not
  replacements for raw stats. Mean top50 Jaccard was `0.487` for armor and
  `0.434` for MR; minimum top-tail Spearman was `0.601` and `0.530`.
- Denominator correlations were near zero (`armor` vs `goldearned`
  mean Pearson `-0.059`; `magicresist` vs `goldearned` `-0.046`), so the ratios
  add a per-gold efficiency axis instead of simply reweighting the raw stat.

### Ability Power

Promoted config: `kv=0.99`, `t=0.80`.

Evidence:

- Current `kv=0.85` retained one PCA axis (`0.980`) and produced only a
  high/low split (`phase_g=[2,2,2,2]`).
- `kv=0.99` retained the second axis (`0.980, 0.020`). PC1 is AP investment;
  PC2 separates raw AP from AP-per-gold efficiency.
- `kv=0.99, t=0.80` produced `phase_g=[4,4,4,4]`, coverage `1.00`,
  largest-group share `0.59`, median within-group cosine `0.99`.
- The semantic read is high AP, low/no AP, and incidental raw AP with weaker
  per-gold efficiency. Mid and late both exposed the incidental AP groups
  without creating tiny support fragments.

Rejected:

- `kv=0.99, t=0.70` was stable but under-specified in mid phase
  (`phase_g=[4,3,4,4]`), merging an incidental AP side group back into the
  no-read pool.

Replacement check:

- `abilitypower_to_goldearned_ratio` is not a raw-AP replacement. Mean top50
  Jaccard was `0.563`, and minimum top-tail Spearman was `0.676`.
- `abilitypower` vs `goldearned` had mean Pearson `-0.267`, so the ratio is a
  different semantic question: AP density rather than absolute AP.

### Attack Damage

Promoted config: `kv=0.99`, `t=0.70`.

Evidence:

- Current `kv=0.85` retained one PCA axis (`0.982`) and produced only a
  high/low split (`phase_g=[2,2,2,2]`).
- `kv=0.99` retained the second axis (`0.982, 0.018`). PC1 is AD investment;
  PC2 separates absolute AD from AD-per-gold efficiency.
- `kv=0.99, t=0.70` produced `phase_g=[5,4,6,6]`, coverage `1.00`,
  largest-group share `0.57`, median within-group cosine `0.97`.
- The semantic read is low/no AD, high AD carries/fighters, and smaller
  on-hit/raw-stat side groups.

Rejected:

- `kv=0.99, t=0.80` increased median cosine to `0.98` and group count to
  `phase_g=[8,6,6,7]`, but the new groups were mostly tiny utility/support
  fragments with weak z-scores rather than new AD semantics.

Replacement check:

- `attackdamage_to_goldearned_ratio` is strongly not a raw-AD replacement.
  Mean top50 Jaccard was `0.362`, and minimum top-tail Spearman was `0.370`.
- `attackdamage` vs `goldearned` had mean Pearson `+0.615`; the denominator is
  materially correlated with the numerator, so the ratio changes the question
  and must stay paired with raw `attackdamage`.

## Replacement Metrics

Use these checks when a transformed feature is intended to preserve the same
"who is high/low" read as an absolute metric.

### Tail Correlation

Compute Spearman and Pearson between raw and transformed values in the relevant
tails, usually top 10% and top 5% by the raw metric. If the low tail is a
meaningful negative archetype, repeat for bottom 10% and bottom 5%.

Thresholds:

- Accept a transformed feature as a replacement only when relevant-tail
  Spearman is `>= 0.90` and relevant-tail Pearson is `>= 0.80`.
- If Spearman passes but Pearson fails, document that rank is preserved but
  scale or magnitude changed.
- If Spearman is `< 0.85`, treat the transformed feature as a different
  semantic question.

Spearman is the primary gate because ranking stability matters most for group
identity. Pearson is the scale-sensitivity check.

### Jaccard Similarity Coefficient

Compare identity sets per phase:

- top 50 raw identities vs top 50 transformed identities,
- bottom 50 raw identities vs bottom 50 transformed identities.

Thresholds:

- Mean top50 Jaccard `>= 0.70` across phases is relevant overlap.
- Mean top50 Jaccard `>= 0.80` is strong overlap.
- Mean top50 Jaccard `< 0.60` means the transformed feature changed the read and
  must not replace the raw metric without reframing the specialist question.
- If the low tail has semantic meaning, bottom50 Jaccard uses the same
  thresholds: `>= 0.70` relevant, `>= 0.80` strong, `< 0.60` changed question.
- If the bottom is a no-read pool, record bottom50 Jaccard but do not let it
  override a strong top-tail decision.

For top/bottom 50 sets, Jaccard `0.70` is about 41 shared identities out of 50.

### Denominator Sanity

For ratios, correlate the denominator with the raw numerator and inspect PCA
loadings before accepting the ratio as part of the evaluation.

- A near-zero denominator correlation may be harmless, but it probably does not
  create a useful new axis by itself.
- A strongly correlated denominator can erase, invert, or over-normalise the
  original read.
- A ratio should be accepted only if PCA and group inspection show the intended
  semantic axis.

## Feature-Set Diagnostics

### Single-Feature Specialists

A single feature L2-normalises to a sign, so clustering can only produce a
coarse high/low split. Use a single-feature specialist only when an ordinal
above/below read is the intended behavior. Do not tune `t` or `kv` expecting
multi-archetype structure without adding another meaningful axis.

### Correlated Magnitude Pairs

Two raw volume features that rise together usually load onto one magnitude axis.
To split composition rather than intensity, test whether a ratio between the two
raw features or a pair of consistently normalised component metrics creates a
second interpretable PCA axis. Reject variants that keep one dominant axis and
only change the high/low threshold.

### Numerator-Sharing Normalisers

Adding `x / denominator` next to raw `x` often reweights the same numerator
axis rather than adding a new direction. Treat it as suspect until PCA loadings
show an independent axis and group inspection shows a new semantic read.

### Sparse Zero-Inflated Columns

Before adding sparse metrics, print per-column standardised variance after the
same transforms used by the embedding matrix. If an individual column explodes
because the median/MAD scale fallback is inappropriate, combine related sparse
metrics on raw values first and standardise the combined column once. A sparse
column creating a large PCA direction is not sufficient evidence by itself.

### Algebraic Features

Do not reject a feature only because it is a linear combination of other
features. Keep it only if removal merges or weakens interpretable groups; remove
it if it leaves the same semantic partition unchanged.

## Group Diagnostics

### PCA Plateaus

`kv` lives on plateaus defined by the cumulative-variance curve. Values inside
one step give identical embeddings. Only crossing into the next axis changes
clustering, and crossing can over-fragment. Record the variance boundary when it
affects a decision.

### Threshold Fragments

A higher `t` or forced low-variance axis can create more tiny groups without new
top z-scores or champion semantics. Reject that change even when median cosine
improves. Prefer a lower threshold when the higher threshold only subdivides an
already interpretable group.

### No-Read Pools

A largest group with uniformly negative z-scores across specialist features is
a background pool, not a positive archetype. It is valid to keep a no-read pool;
do not tune until it becomes a fake positive class.

### Role/Build Leakage

If a group is explained only by `teamposition` or `build`, it is not adding
behavior beyond the identity key. Keep it only when specialist metrics provide
an independent label.
