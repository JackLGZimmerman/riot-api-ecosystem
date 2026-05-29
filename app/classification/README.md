# Classification Embeddings

This directory builds phase-aware identity descriptors for draft prediction.
The working identity is `(championid, teamposition, build)`, and each output is
preserved by temporal phase so a downstream model can compare what an identity
usually contributes in early/mid/late game contexts.

The goal is not to replace the draft model. It is to provide extra structured
signals that help separate games that otherwise sit in the low-confidence
`0.45-0.55` band. The embedding layer turns smoothed historical metrics into:

- **Specialist group labels** for multi-metric behaviours such as durability,
  map control, damage profile, utility, economy, objectives, and structure play.
- **Singular metric orderings** for one-dimensional behaviours whose useful
  meaning is relative rank, such as movement speed or low death rate.

Run:

```bash
uv run python -m app.classification.embeddings.pipeline       # run specialists + report
uv run python -m app.classification.embeddings.specialists    # specialists only
uv run python -m app.classification.embeddings.singular_metrics # scalar orderings only
uv run python -m app.classification.embeddings.tune           # specialist sweep
```

Source data is `game_data_filtered.synergy_1vx_temporal` (`6010`), smoothed
with the `9000-9040` prior tables. Embeddings are L2-normalised and grouped by
average-link agglomerative clustering on cosine distance.

Temporal bins are preserved through grouping generation. Matrices are shaped as
`(identity, phase, feature)`, PCA is fit once across all identity-phase rows to
maintain a shared latent semantic space, and clustering is then run
independently within each phase. Temporal embeddings are not flattened, pooled,
or averaged into a single identity embedding before grouping.

## Latest Audit

The full 2026-05-28 registry audit is in
[SPECIALISATIONS.md](SPECIALISATIONS.md). It evaluates every active
`SpecialistSpec` and `SingularMetricSpec`, including gold/death normalisation
checks, phase-local group tables, top/bottom identities, and a one-line quality
read per group.

Main takeaways:

- Most normalised variants changed the identity tail, so they should usually be
  treated as new semantic axes rather than replacements for raw metrics.
- `enchanters`, `durability`, `self_sustain`, `takedown_shape`,
  `utility_pickmaking`, `jungle_control`, `siege_pressure`, `map_control`,
  `resistances`, and `ability_power` currently have the cleanest group shapes.
- The follow-up tuning pass promoted `burst_skirmish`, `economy_scaling`,
  `early_agency`, `damage_profile`, `damage_efficiency`,
  `defensive_statline`, `attack_damage`, and `on_hit_carry` by lowering PCA
  retention enough to remove low-variance threshold fragments while preserving
  clear semantic group reads.

Future performance work should validate specialist labels with target encoding
or regularised one-hot features in the downstream draft model. If another broad
spec regresses into high-cardinality fragments, split it into narrower
questions or lower PCA retention before treating its labels as cheap
categoricals.

## Outputs

Each specialist is a separate embedding whose feature set is chosen for the
independent directions retained by PCA. Groups whose median pairwise cosine
sits below `min_median_sim` are dropped (the identity gets a `-1` label).
Small coherent groups are valid specialist reads; there is no size floor.

Active registry, see `SPECIALISTS` in [embeddings/config.py](embeddings/config.py):

### Specialist Composition

Per-specialist labels are saved as `npz` files in
`data/embeddings/cache/specialists/<name>.npz` with `keys`, `key_columns`, and
`labels` arrays. `labels` is shaped `(identity, phase)` and uses `-1` for
identity-phase rows that fell into a dropped group; `phases` names the temporal
axis. These embedding/report artifacts are generated outputs and are ignored by
git. Downstream code should intersect labels within the same phase.

Specialist label numbers are identifiers, not magnitudes. A label value of `3`
is not stronger than `2`; it only means "member of specialist group 3 for this
phase". Downstream models should one-hot, target-encode, or otherwise encode
group membership per specialist and phase, rather than treating raw label IDs as
ordered values.

### Singular Metrics

Singular metrics are saved as `npz` files in
`data/embeddings/cache/singular_metrics/<name>.npz`. They keep a single
semantic feature as a phase-relative ordering instead of forcing it through a
cluster. Each file contains:

- `standardised_values`: the transformed feature values used for ordering.
- `ranks`: 1-based phase-local ranks, with ties averaged.
- `percentiles`: phase-local percentile ordering in `[0, 1]`.
- `scores`: centered percentiles in `[-1, 1]`, where positive means stronger in
  the configured semantic direction.
- `higher_is_more`: whether larger raw values define the positive direction.

Use singular metric `scores` as continuous vector inputs alongside encoded
specialist membership. They are intentionally separate from `SpecialistSpec`
because metrics like `movementspeed` do not have a natural companion set and
their useful signal is the relative ordering of identities in the same phase.

### Downstream Vector Shape

The intended model-facing view is one row per `(identity, phase)` with:

- encoded membership for every available specialist group in that phase,
- continuous singular metric scores for every configured singular metric,
- optional raw or calibrated priors only when they are split-safe.

This shape makes sense as long as the downstream join keeps the phase axis
intact. Avoid flattening all phases into one identity vector unless the model is
explicitly designed to consume temporal summaries. Also avoid training on
classification artifacts built from validation/test rows; `win` is available in
the raw catalogue but should be treated as a target/prior with strict split
discipline, not as an ordinary behavioural feature.

### Adding A Specialist

1. Add a `SpecialistSpec` to `SPECIALISTS` in [embeddings/config.py](embeddings/config.py).
2. If the spec needs a derived feature not yet in `DERIVED_METRIC_FUNCS`,
   add it there.
3. Sweep `kv` × `t` with `tune.py --name <name>`.
4. Run `uv run python -m app.classification.embeddings.specialists` and
   inspect the report.
5. Use the generic inspector for PCA axes and feature z-scores:
   `uv run python -m app.classification.embeddings.inspection.base --name <name>`.

Prefer features that add a unique axis for the specialist. Raw metrics are
allowed, but avoid pairing a raw numerator with a ratio that already contains
the same information unless the PCA inspection shows a distinct retained
direction.

### Adding A Singular Metric

Add a `SingularMetricSpec` to `SINGULAR_METRICS` in
[embeddings/config.py](embeddings/config.py). Use this for a feature where the
phase-relative order is meaningful by itself and clustering would mostly create
arbitrary buckets. Set `higher_is_more=False` when the positive semantic signal
is a lower value, such as death rate.

See [EXPERIMENTS.md](EXPERIMENTS.md) for tuning workflow.
