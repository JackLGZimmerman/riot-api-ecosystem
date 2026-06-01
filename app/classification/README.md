# Classification Embeddings

This package builds non-temporal identity descriptors for draft prediction.

> An identity is the tuple `(championid, teamposition, build)`.

The classification path builds one descriptor per identity. It now reads
timeline checkpoints as historic averages, but there is still no phase axis in
the Python matrices, specialist labels, or singular metric scores.

## Data Source

`embeddings/load.py` aggregates rows directly from:

- `game_data_filtered.participant_stats`
- `game_data_filtered.participant_item_value_totals`
- `game_data_filtered.ml_game_split`
- `game_data.tl_participant_stats` for final participant stat snapshots and
  checkpoint snapshots at 3, 4, 5, 7, 10, 12, 15, 20, 22, and 25 minutes
- `game_data.participant_challenges` for lane pressure, solo-kill, damage,
  gold/minute, and turret/plate context

The prior hierarchy is derived in memory from the baseline aggregate. The old
`synergy_1vx_temporal`, `synergy_1vx_temporal_prior_*`, and
`participant_scaling_weights` tables are not part of the rebuild path.

## Run

```bash
uv run python -m app.classification.embeddings.pipeline
uv run python -m app.classification.embeddings.dense
uv run python -m app.classification.embeddings.relationship_details
uv run python -m app.classification.embeddings.inspection.relationship_detail_probe
uv run python -m app.classification.embeddings.specialists
uv run python -m app.classification.embeddings.singular_metrics
uv run python -m app.classification.embeddings.tune
```

## Outputs

Specialist labels are saved as ignored generated files under
`app/classification/data/embeddings/cache/specialists/<name>.npz`.
Each file contains `keys`, `key_columns`, and one `labels` array shaped
`(identity,)`; `-1` means the identity fell into a dropped low-coherence group.

Singular metrics are saved under
`app/classification/data/embeddings/cache/singular_metrics/<name>.npz`.
Each file contains `standardised_values`, `ranks`, `percentiles`, and `scores`,
all shaped `(identity,)`.

Dense HGNN identity descriptors are saved to
`app/classification/data/embeddings/cache/identity_semantic_embedding.npz`.
They are keyed by `(championid, teamposition, build)` and are intended to be
passed to `app/ml/hgnn_model.py` as `identity_semantic` with shape
`(batch, 10, 64)`. The HGNN projects this vector into the node state and fuses
it with the existing champion/role/build identity embedding and 1vX posterior.

Relationship-detail vectors are saved under
`app/classification/data/embeddings/cache/relationship_details/`. The 1v1 file
is directional and is passed as `m1v1_detail` with shape `(batch, 25, 16)`;
the 2vX file is symmetric and is passed as `s2vx_detail` with shape
`(batch, 20, 16)`. These vectors enrich the HGNN relationship residual head
with historic gold, CS, XP, damage, solo-kill, level-lead, and plate pressure.
Checkpoint timing is carried by the per-identity semantic descriptors, while
relationship details let a matchup such as Sion TOP ar_tank vs Yone TOP crit
carry its high-stomp lane profile rather than only a small win-rate edge.

Label numbers are identifiers, not magnitudes. Downstream models should one-hot,
target-encode, or otherwise encode group membership, and should use singular
metric `scores` as continuous inputs.

## Adding A Specialist

1. Add a `SpecialistSpec` to `SPECIALISTS` in
   [embeddings/config.py](embeddings/config.py).
2. Add any needed derived metric to `DERIVED_METRIC_FUNCS`.
3. Sweep with `uv run python -m app.classification.embeddings.tune --name <name>`.
4. Run `uv run python -m app.classification.embeddings.specialists`.
5. Inspect with
   `uv run python -m app.classification.embeddings.inspection.base --name <name>`.

Prefer features that add a unique axis for the specialist. Raw metrics are
allowed, but avoid pairing a raw numerator with a ratio that already contains
the same information unless PCA inspection shows a distinct retained direction.
