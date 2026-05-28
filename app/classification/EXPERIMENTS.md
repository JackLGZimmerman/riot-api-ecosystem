# Classification Tuning

**Goal**: tune specialists for semantic coherence and recoverable metric reads.
As a rough scale check, expect kept groups near `1.5 * active_metrics`; do not
promote a config only because the sweep score looks better. Inspect the groups
and prefer stable, named behavior over threshold fragments.

**CRITICAL**: Do not add experiment outcomes here, only processes during experimentation that supported generating more performant and effective groupings.

## Run Loop

```bash
# Production: rebuild specialists end-to-end and write the HTML report.
uv run python -m app.classification.embeddings.pipeline

# Sweep one specialist.
uv run python -m app.classification.embeddings.tune \
    --name epic_objectives --kvs 0.95 0.97 --ts 0.66 0.70 0.72 0.76

# Inspect the picked candidate.
uv run python -m app.classification.embeddings.inspection.base \
    --name epic_objectives --kv 0.97 --t 0.72

# Inspect a feature/threshold variant without editing config.
uv run python -m app.classification.embeddings.inspection.base \
    --name structure --kv 0.85 --t 0.70 \
    --features structure_takedowns structure_losses structure_damage \
        structure_takedowns_to_structure_damage_ratio structure_net_control \
        structure_damage_to_goldearned_ratio structure_damage_to_deaths_ratio
```

The HTML report under `app/classification/data/embeddings/` is the production
view for the picked config. The CLI inspector is the faster debugging view:
retained PCA axes, feature z-scores, and build/role/champion composition.
The inspector lives at `app/classification/embeddings/inspection/base.py`.
Use `inspection/specialist_configs.txt` to preserve or copy the tuned
per-specialist CLI arguments.

`tune.py` reuses `/tmp/embed_exp/raw_levels.pkl`. Move or delete that cache
after rebuilding `6010` or `9000-9040`; a stale cache can fail with missing
columns, as happened when `truedamagetaken` was added after the cache was
written. Adding a column to `ALL_METRICS` in `config.py` requires the source
column to exist in `6010` (and the `9000-9040` priors) first, or load fails
with `UNKNOWN_IDENTIFIER`; rebuild those tables from their `*_schema.sql` +
`*_build.sql` before deleting the cache.

## Separating Overlapping Archetypes

When two archetypes share one signature but should be different specialist
reads (e.g. two groups that share `kills_to_assists` but differ on
survivability), find the axis that actually separates them before editing a
feature set. `inspection/discriminate.py` ranks candidate features by the
standardized mean gap between two named champion sets:

```bash
uv run python -m app.classification.embeddings.inspection.discriminate \
    --phase mid --set-a <squishy champs> --set-b <durable champs> \
    --features <candidate1> <candidate2> ...
```

Process:

1. Name a clean exemplar set for each archetype (champions you are confident
   belong to A vs B). The gap is only as good as these sets.
2. Rank candidate features; gold-normalised ratios usually beat raw volumes
   because they control for game stage/farm.
3. Take the top-gap feature into `inspection.base --features ...` and confirm
   the clustering actually splits A and B into separate coherent groups, not
   just that the means differ. A large gap that does not survive clustering is
   not a usable axis.
4. If both archetypes share the same separating subspace, one specialist whose
   clustering emits both groups is usually cleaner than two specialists that
   re-embed the same axes.

## Semantic Review Loop

The latest `epic_objectives` / `structure` pass used a stricter review loop
than "sweep, pick, promote":

1. Start with the current config and inspect every retained group with
   `inspection.base`, including PCA axes, feature z-scores, role/build mix, and
   top champions.
2. Small groups are okay when their champion mix and
   z-scores are clear; size floors are only a reporting/stability tool, not a
   semantic filter.
3. Compare materially different feature vocabularies:
   - remove duplicated derived features,
   - remove suspicious ratios,
   - try atomic raw metrics when a ratio may be hiding multiple behaviours.
4. For each group, ask whether the label can be recovered from the original
   metrics. A group with weak top z-scores, overlapping champion composition,
   or only role/build separation is a threshold slice, not a specialist
   archetype.
5. Prefer the simplest candidate whose groups can be explained. A sweep pick
   with more groups or higher median cosine can be rejected if the extra groups
   are mostly fragments.

During the review, a small scratch scorer counted groups with a meaningful
top feature z-score, but it was only a triage aid. Manual champion validation
remains the promotion gate.

## Promotion Rules

- Kept groups should land near `1.5 * active_metrics` unless the specialist is
  intentionally sparse or has a clear continuum/no-read pool.
- Median within-group cosine should clear the specialist floor, usually
  `>= 0.85`; higher medians are not meaningful by themselves.
- Top z-scores should explain the group composition in domain terms.
- Prefer fewer, larger interpretable groups over many tiny threshold splits.
- If a largest group has uniformly negative z-scores, treat it as a continuum
  or "no distinctive read" pool instead of a positive archetype.
- There is no size floor; small coherent groups are valid specialist reads.
  Tune group count with features and `t`, not by dropping small groups.
- Do not accept `tune.py`'s first-ranked row automatically. It sorts by generic
  shape metrics; the final choice can use a lower threshold if the higher
  threshold mostly creates extra fragments.

## General Heuristics

**Axis redundancy**: features that share a numerator often overweight one
intensity axis. A duplicate can still be useful if it improves separation, but
verify that PCA retains another interpretable direction.

**Low-magnitude cosine trap**: near-zero identities can form a tight, coherent
cluster after L2 normalisation. Symptoms are PC1 dominance, full coverage, and
a largest group with negative z-scores on every feature. Raising `t` can split
the pool, but only promote the split if it exposes a new semantic read.

**Bimodal role traps**: ratios that are near 0 for most roles and near 1 for
one role often re-encode `teamposition`. Prefer atomic volumes when the ratio
collapses the specialist into "role X vs not-X".

**Identity-key leakage**: do not split a group only by role or build when those
fields are already in `(championid, teamposition, build)`. A specialist should
add behavior not already recoverable from the key.

**PCA plateaus**: `kv` lives on plateaus defined by the cumulative-variance
curve. Values inside one step give identical clustering; only crossing into the
next axis changes anything, and crossing usually over-fragments. If a step is
tight (e.g. 0.876 → 0.913 at a 3→4 axis boundary), there is no useful middle
ground to sweep.

**`min_median_sim` is not a weak-archetype filter**: median within-group cosine
and top-feature z-score strength are not correlated. A tight low-magnitude
continuum can sit at `med=0.93` while a coherent but weakly-differentiated
archetype sits at `med=0.86`. To drop a weak-z group, change features or `t`,
not `min_median_sim`.

**Algebraic features can still be axes**: a feature that is a linear
combination of others (e.g. `net_control = takedowns - losses`) can still
become an independent embedding direction after L2 normalisation. Before
removing one as "redundant", check whether removal merges previously-distinct
champion archetypes — not just whether PCA still retains the same number of
axes.

**Sparse zero-inflated metrics need scale control before they can join a
specialist**: raw `lifesteal`/`omnivamp`/`spellvamp`/`physicalvamp` are mostly
zero. After signed-log1p + median/MAD standardisation, a column whose median is
0 gets `MAD -> 1.0` (the fallback), so its variance is set by the non-zero tail
rather than normalised to unit scale. Always print per-column standardised
variance before adding such a feature: in the vamp case `spellvamp` came out at
`var=6e8` (a smoothing/MAD artifact) and `physicalvamp` at `var=0` (always
zero), while `lifesteal`/`omnivamp` were a usable `~2.5`. The fix is to combine
the related sparse metrics into **one** feature and standardise the combined
column once — summing all four vamp sources on *raw* values yields `vamp_sustain`
at `var≈1.15` (unit scale), because the tiny spellvamp/physicalvamp contributions
ride on the larger lifesteal/omnivamp mass instead of each blowing up alone. Rule:
combine before standardising; never standardise a sparse zero-inflated column on
its own, and do not assume a "broken" raw column must be dropped — inside a sum it
is harmless.
