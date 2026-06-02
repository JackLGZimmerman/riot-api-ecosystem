# Classification Metrics

This package keeps the champion historical profile metrics used for downstream
semantic representation work.

An identity is `(championid, teamposition, build)`. The preserved data surface is
intentionally small:

- raw full-game source metrics from participant stats
- final participant stat snapshots where relevant
- derived ratios in `DERIVED_METRIC_FUNCS`
- the standalone champion semantic autoencoder baseline

Challenge-derived data is forbidden here. Do not add
`game_data.participant_challenges` joins or `challenge_*` feature columns.

## Metric Catalogue

The core definitions live in [embeddings/config.py](embeddings/config.py):

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
the preserved source columns.

## Run

Build the raw/derived metric matrices:

```bash
uv run python -m app.classification.embeddings.pipeline
```

Train the champion semantic autoencoder:

```bash
uv run python -m app.classification.champion_semantics \
  --csv path/to/profiles.csv \
  --epochs 7200 \
  --batch-size auto \
  --device auto \
  --noise-std 0.003 \
  --latent-dropout 0.05 \
  --latent-decorrelation-weight 0.0005 \
  --latent-output app/classification/data/embeddings/cache/champion_latents.csv
```

The default autoencoder uses a 512-d latent vector, latent BatchNorm, and a
80-wide metric bottleneck. Decoder-side latent dropout plus a light latent
decorrelation penalty maximize non-trivial semantic grouping capacity while
preserving metric reconstruction accuracy. Autoencoder details and smoke-test
commands live in
[AUTOENCODER_README.md](AUTOENCODER_README.md).

## Tests

```bash
uv run pytest tests/classification -q
uv run ruff check app/classification tests/classification
```
