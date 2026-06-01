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
`(batch, 10, 64)`. The broad node-state path remains disabled in the production
config, but the profile-v2 interaction uses a rank-4 projection of this
historical descriptor to condition profile sensitivity without exposing
current-match postgame fields.

Matchup-profile descriptors are saved alongside, to
`app/classification/data/embeddings/cache/identity_profile_embedding.npz`
(also written by `python -m app.classification.embeddings.dense`). They are
9 interpretable `[0, 1]` axes per identity: three champion damage-type shares,
two resistance fractions, one expected champion-damage pressure, and three
damage-pressure-weighted type axes. These are raw smoothed identity values, not
the standardized dense-embedding features, so the share/fraction semantics
survive. They are passed to the HGNN as `identity_profile` with shape
`(batch, 10, 9)` and feed the per-player antisymmetric cross-team interaction
term, which lets an identity's resistance profile interact with the opposing
team's damage-weighted aggregate damage-type composition (e.g. an armor tank
gaining win rate as the enemy team's physical share rises). Profile v2 also
concatenates the low-rank semantic context above and explicit resistance × enemy
offense products to the profile-head input, without using the current game's
realized diagnostics.

## Semantic Meaning For HGNN

The classification layer gives the win model two kinds of meaning:

- A dense semantic descriptor captures many historical behavioural signals for
  the identity, such as lane pressure, damage timing, economy, durability, and
  objective pressure. This vector is intentionally compressed before HGNN uses
  it, so it can condition a decision without becoming a large memorization path.
- An interpretable matchup profile keeps a small set of named axes in their
  natural units. The production profile currently exposes damage-type mix,
  resistance mix, and expected champion-damage pressure. The HGNN can then learn
  shared rules such as "armor-heavy identities benefit against damage-weighted
  physical enemy compositions" rather than memorizing one champion pair at a
  time.

This split is the pattern for adding thousands of future contextual groups.
Each new group should describe a reusable semantic axis of champion decisions:
anti-burst, anti-poke, side-lane pressure, engage reliance, true-damage threat,
healing denial, scaling curve, jungle tempo, or any other stable pre-game
context. If the axis has clear units, add it as a named profile feature or an
explicit product with enemy/team context. If the axis is broader or correlated
with many stats, let it enter the dense semantic descriptor and pass through the
same low-rank bottleneck.

The important boundary is that these descriptors are historical identity
summaries keyed by `(championid, teamposition, build)`. They may estimate what a
champion usually does in a matchup context, but they must not include the current
match's realized damage, taken damage, mitigation, final items, or outcome. That
keeps the model deployable at draft time while still giving it enough semantic
texture to specialize globally across many champion/build/role contexts.

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
