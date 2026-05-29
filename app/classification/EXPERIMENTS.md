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
