# Classification Metrics

This package keeps the champion historical metrics used for downstream semantic
representation work.

An identity is `(championid, teamposition, build)`. The preserved data surface is
intentionally small:

- raw full-game source metrics from participant stats
- final participant stat snapshots where relevant
- derived ratios and signed differences in `DERIVED_METRIC_FUNCS`
- team-participation / role-matchup context features in `context_features.py`
  (included by default for the full-game 215-metric surface; disable with
  `include_context_features=False` for legacy profile-only matrices)
- standalone static, full-game, and temporal identity autoencoder baselines

Challenge-derived data is forbidden here. Do not add
`game_data.participant_challenges` joins or `challenge_*` feature columns.

## Metric Catalogue

The core definitions live in [embeddings/config.py](../embeddings/config.py):

- `RATE_METRICS`
- `LARGEST_AVG_METRICS`
- `FINAL_SNAPSHOT_AVG_METRICS`
- `PER_MINUTE_METRICS`
- `RATE_LIKE_METRICS`
- `ALL_METRICS`
- `DERIVED_METRIC_FUNCS`
- `raw_and_derived_metric_names()`

`embeddings/load.py` loads and caches aggregate rows. `embeddings/matrices.py`
builds standardised matrices from smoothed rows and computes derived metrics from
the preserved source columns. The per-encoder metric surfaces are listed in
[ENCODER_METRICS.md](ENCODER_METRICS.md).

## Run

Build the default all-metrics matrices:

```bash
uv run python -m app.classification.embeddings.pipeline
```

Train the full-game identity autoencoder with all 215 default metrics:

```bash
uv run python -m app.classification.full_game_encoder \
  --csv path/to/profiles.csv \
  --epochs 7200 \
  --batch-size 1024 \
  --device auto \
  --noise-std 0.003 \
  --latent-dropout 0.10 \
  --latent-decorrelation-weight 0.0005 \
  --neighbor-k 10 \
  --latent-output app/classification/data/embeddings/cache/full_game_identity_latents.csv
```

The default autoencoder uses a 640-d latent vector, latent BatchNorm, and a
215-column full-game input surface with a 160-wide metric bottleneck and a
`(512, 384)` decoder.
Decoder-side latent dropout plus a light latent
decorrelation penalty maximize non-trivial semantic grouping capacity while
preserving metric reconstruction accuracy. Autoencoder details and smoke-test
commands live in
[AUTOENCODER_README.md](AUTOENCODER_README.md).
For clustering, use the exported `latent_*` columns keyed by
`champion_id`, `teamposition_id`, and `build_id`; keep `--neighbor-k 10` on
regeneration runs so semantic-neighborhood preservation is visible next to MSE.
For HGNN semantic-context experiments, package static, full-game, and temporal
latents plus support into the encoder sidecar artifact and enable the learned
semantic MoE production path.

If the embedding pipeline reports a stale `classification_identity_base`
catalogue hash or missing context columns, rebuild the classification
ClickHouse tables with `build_classification_tables()` before training the
215-metric encoder. The builder recreates its materialized schemas on rebuild,
so added metrics are materialized instead of being hidden by an old
`CREATE TABLE IF NOT EXISTS` table.

## Tests

```bash
uv run pytest tests/classification -q
uv run ruff check app/classification tests/classification
```
