# Classification Embeddings

Per-identity classification at the `(championid, teamposition, build)` key. The
pipeline runs a set of **specialist** embeddings. Each asks one narrow
behavioural question over its own small feature subset and emits labels for each
identity in each temporal bin.

Run:

```bash
uv run python -m app.classification.embeddings.pipeline       # run specialists + report
uv run python -m app.classification.embeddings.specialists    # specialists only
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

## Specialists

Each specialist is a separate embedding whose feature set is chosen for the
independent directions retained by PCA. Groups whose median pairwise cosine
sits below `min_median_sim` are dropped (the identity gets a `-1` label).
Small coherent groups are valid specialist reads; there is no size floor.

Active registry, see `SPECIALISTS` in [embeddings/config.py](embeddings/config.py):

### Composition

Per-specialist labels are saved as `npz` files in
`data/embeddings/cache/specialists/<name>.npz` with `keys`, `key_columns`, and
`labels` arrays. `labels` is shaped `(identity, phase)` and uses `-1` for
identity-phase rows that fell into a dropped group; `phases` names the temporal
axis. These embedding/report artifacts are generated outputs and are ignored by
git. Downstream code should intersect labels within the same phase.

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

See [EXPERIMENTS.md](EXPERIMENTS.md) for tuning workflow.
