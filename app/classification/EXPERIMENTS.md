# Classification Tuning

Goal: tune relevant specialists for ~6-14 kept groups. (The key is semantic
meaning in the groups)

## Run Loop

```bash
# Production: rebuild specialists end-to-end
uv run python -m app.classification.embeddings.pipeline

# Lean specialist sweep (writes /tmp/embed_exp/tune_specialists.txt):
uv run python -m app.classification.embeddings.tune --lo 6 --hi 14
# Scope to one spec:
uv run python -m app.classification.embeddings.tune \
    --name sustained_damage --kvs 0.80 0.85 0.90 --ts 0.65 0.70 0.75

# Inspect the picked sustained-damage candidate:
uv run python -m app.classification.embeddings.inspect_sustained_damage --kv 0.85 --t 0.70

# Inspect the picked burst-damage candidate:
uv run python -m app.classification.embeddings.inspect_burst_damage --kv 0.85 --t 0.60

# Inspect the picked vision candidate:
uv run python -m app.classification.embeddings.inspect_vision --kv 0.85 --t 0.68

# Inspect the picked farming candidate:
uv run python -m app.classification.embeddings.inspect_farming --kv 0.95 --t 0.88
```

The HTML report under `app/classification/data/embeddings/` is a
single-threshold inspection lens for the picked config.

## Tuning Process

`tune.py` reuses a pickled raw cache at `/tmp/embed_exp/raw_levels.pkl`. Delete
it after rebuilding the `6010` or `9000-9040` tables to force a refresh.

Specialist sweep output per candidate: `spec kv t | g cov lg% med`.
`--min-median` overrides the spec's coherence floor without editing config.

## Specialist Feature Refinement

Feature candidates live in `DERIVED_METRIC_FUNCS` in `config.py`. Iterate:

1. Sweep → pick in-band config.
2. Inspect group semantics: build/role/champion composition + feature z-scores vs global mean.
3. Prune features that co-load across unrelated groups or dominate a PCA axis without adding a distinct specialist read.
4. Re-sweep — removing a feature concentrates the embedding, so the threshold typically needs to increase to maintain group count.

For sustained-damage inspection, use
`uv run python -m app.classification.embeddings.inspect_sustained_damage --kv <kv> --t <threshold>`.
It prints retained PCA axes, then each group's top build/role/champion counts
alongside the highest-|z| features vs global mean. Success = each group's top
z-score aligns with its champion composition.

Replicate this inspection file pattern for future specialist tests: keep the
script narrow to the specialist's default feature set, but allow feature
overrides while experimenting with candidate axes.

**Axis redundancy**: a raw volume feature can be useful when it contributes a
unique direction for the specialist. If it shares a numerator with ratio
features already in the set and only adds collinear variance, drop one side of
the pair so PCA does not overweight duplicate information.

**Low-magnitude cosine trap**: features where most identities sit near zero
(e.g. vision metrics outside support/jungle) produce a large *tightly-coherent*
generic cluster after L2 normalisation, because near-zero vectors all point in
similar directions. `min_median_sim` cannot distinguish "tight because semantic"
from "tight because all-near-zero" — both look coherent. Symptoms:

- PC1 explains >70% of variance (features collinear on the intensity axis).
- The largest kept group has high median cosine *and* uniformly negative z on
  every feature (no positive read on anything).
- Coverage stays at 1.0 even at high `min_median_sim`, because the all-near-zero
  cluster is structurally cohesive.

Mitigations tried, in order of cost:

1. Drop the most collinear feature (e.g. raw volume that duplicates a ratio's
   numerator). Boosts PC2's share, splits the low-magnitude pool into shallower
   sub-strata.
2. Raise the similarity threshold (`t`) further. Splits the diffuse pool into
   smaller fragments, but does not eliminate the dominant low-magnitude cluster.
3. Treat the spec as a continuum (per [Promotion Rules](#promotion-rules)) and
   accept that the largest kept group is the "no distinctive read" archetype.

For `vision`, mitigation 1 was applied (drop `total_wards_placed` — PC1 share
77.9%, PC2 17.3%) and `t` was tuned for cluster cardinality rather than
maximally splitting the low-magnitude pool. At `t=0.68` the result is 5 groups:
supports, junglers + roamers, off-role enchanters in MID/TOP, the Fiddlesticks
Scarecrow-Effigy passive anomaly (per-action z=+4.5), and a 54% continuum group
for the "no distinctive read" lane carries. Raising `t` further (e.g. `t=0.93`)
splits the continuum into 12 groups but mostly fragments the same low-magnitude
pool — fewer interpretable named archetypes for more cardinality.

**Picking `t` for low-magnitude-trapped specs**: at low `t` (~0.68) the
continuum collapses into one labelled group and the small high-signal
archetypes stand alone — preferable when downstream consumers want named
buckets. Higher `t` over-splits the continuum without surfacing new semantic
reads.

**Bimodal-ratio role traps**: ratios that approach 0 for some roles and ~1
for others (e.g. `jungle_to_lane_minions_ratio` — ~0 for non-junglers, ~0.9
for junglers) act as a 1-bit role indicator after standardisation. They
dominate PC1 (95%+) and re-encode information the role/build identity already
provides, so the spec collapses to "role X vs not-X". Mitigation: replace the
ratio with the two atomic volumes it was hiding (e.g. `totalminionskilled`,
`neutralminionskilled`). Each role then lives at its own (lane, jungle)
coordinate and PCA spreads variance across two orthogonal axes
(`farming`: PC1=53%, PC2=45%) instead of collapsing to one. Trade: more raw
features, but the embedding finds within-role splits (AP lane farmers vs
crit/AD lane carries) that the ratio version could not.

**Shared-numerator collapse**: when multiple ratios share a numerator (e.g.
`total_farm`, `total_farm_to_gold_ratio`, `total_farm_to_deaths_ratio` all
have `totalminionskilled + neutralminionskilled` on top), they reduce to a
single "intensity" axis (PC1≈97%) regardless of how distinct the denominators
seem semantically. Drop the raw volume if a ratio captures it; keep at most
two of the ratios when the denominators truly span different signals
(gold vs deaths).

**Don't split on an axis the identity key already encodes**: `farming` at high
`t` (0.94+) splits the lane mega-cluster into "AP lane farmers" vs "crit/AD
lane carries" — a clean visible split, but build is already part of the
identity key, so this contributes nothing the consumer doesn't already get
from `(championid, teamposition, build)`. The promoted config uses `t=0.88`
(8 groups, largest 47%); the lane mega-cluster has its own positive read
("farms hard, build-agnostic") and the remaining 7 groups surface
*off-axis* archetypes the identity key can't infer: off-role lane hybrids,
carry junglers playing lane, jungle champs picked as support, and aggressive
solo-lane carries. Rule of thumb: before raising `t` to split a mega-
cluster, check what dimension the split is on. If it's role or build, the
identity key already has it and you're just fragmenting.

## Promotion Rules

Specialist promotion requires:

- Kept groups in 6-14.
- Cluster-shape specs: coverage `>= 0.85`, median within-group cosine
  `>= 0.97`, largest non-absence group `<= ~25%`.
- Continuum specs: median within-group cosine `>= 0.85`; coverage may drop,
  `-1` is a meaningful "no distinctive read" signal.
- Top groups read coherently against domain knowledge.
