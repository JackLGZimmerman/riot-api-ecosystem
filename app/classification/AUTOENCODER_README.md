# Champion Semantic Autoencoder

This module is a small PyTorch baseline for learning champion identity
representations from normalized historical profile rows.

The implementation lives in `app/classification/champion_semantics.py`. It is
standalone and is not wired into the HGNN training or serving path yet.

## Goal

Learn a compact latent vector for each `(champion_id, teamposition_id, build_id)`
profile row. The latent vectors can later feed semantic grouping or
classification experiments.

This baseline only trains the representation.

## Inputs

The training data is a pandas `DataFrame` or CSV with:

- `champion_id`: non-negative integer embedding ID
- `teamposition_id`: non-negative integer embedding ID
- `build_id`: non-negative integer embedding ID
- normalized metric columns

Metric columns should describe normalized historical profile behavior only. Do
not include row-support/count metadata such as `matchups`.

Metric columns are configurable. If they are omitted, the module uses the
classification full-game metric catalogue:

- every raw source metric in `ALL_METRICS`
- every derived metric in `DERIVED_METRIC_FUNCS`

Derived metrics can be requested directly if their source columns are present in
the frame. The dataset also accepts `smoothed_<metric>` columns for source
metrics.

## Architecture

`ChampionEncoder` builds four vectors:

- champion identity embedding
- team position embedding
- build embedding
- metrics embedding from a small MLP over the normalized metric vector

Those four vectors are concatenated and passed through a fusion MLP. The fusion
output is normalized and becomes the champion semantic latent vector. The
default is latent `BatchNorm`, which encourages more dimensions to carry
population-level variation for later grouping. `LayerNorm` and no normalization
remain available for ablation runs.

`ChampionAutoencoder` wraps the encoder and adds a decoder MLP that reconstructs
the clean normalized metric vector. Training primarily uses MSE reconstruction
loss, with a tiny default latent decorrelation penalty so semantic dimensions do
not collapse into redundant copies. The decoder also sees dropout-corrupted
latent vectors during training, while extracted latents remain clean. This
spreads reconstruction pressure across more independent semantic dimensions.

Optional denoising can corrupt only the metric input while still reconstructing
the clean metrics:

- `noise_std`: add Gaussian noise
- `mask_prob`: randomly zero metric inputs

## Defaults

- `latent_dim=512`
- `champion_embedding_dim=16`
- `teamposition_embedding_dim=4`
- `build_embedding_dim=8`
- `metrics_embedding_dim=80`
- `metrics_hidden_dims=(224, 112)`
- `fusion_hidden_dims=(128,)`
- `decoder_hidden_dims=(256, 256)`
- `dropout=0.0`
- `latent_dropout=0.05`
- `latent_norm="batch"`
- `noise_std=0.003`
- `mask_prob=0.0`
- `latent_decorrelation_weight=0.0005`

If vocab sizes are omitted in `train_from_dataframe_or_csv(...)`, they are
inferred as `max(id) + 1` from the input frame.

## Python Usage

```python
from torch.utils.data import DataLoader
import pandas as pd

from app.classification.champion_semantics import (
    ChampionProfileDataset,
    evaluate_autoencoder,
    extract_champion_latents,
    train_from_dataframe_or_csv,
)

metric_columns = ("damage_per_min", "gold_per_min", "cc_per_min")
frame = pd.read_csv("profiles.csv")

model, history = train_from_dataframe_or_csv(
    frame,
    metric_columns,
    epochs=7200,
    batch_size="auto",
    device="auto",
    noise_std=0.003,
    mask_prob=0.0,
    latent_decorrelation_weight=0.0005,
)

dataset = ChampionProfileDataset(frame, metric_columns)
dataloader = DataLoader(dataset, batch_size=int(history[-1]["batch_size"]), shuffle=False)
scores = evaluate_autoencoder(model, dataloader, "auto", neighbor_k=10)
latents = extract_champion_latents(model, dataloader, "auto")
```

`extract_champion_latents(...)` returns one row per dataloader sample:

- `champion_id`
- `teamposition_id`
- `build_id`
- `latent_0 ... latent_n`

## CLI Usage

Train from a CSV and write latent vectors:

```bash
uv run python -m app.classification.champion_semantics \
  --csv path/to/profiles.csv \
  --metric-columns damage_per_min gold_per_min cc_per_min \
  --epochs 7200 \
  --batch-size auto \
  --device auto \
  --noise-std 0.003 \
  --latent-dropout 0.05 \
  --latent-decorrelation-weight 0.0005 \
  --latent-output app/classification/data/embeddings/cache/champion_latents.csv
```

`device="auto"` uses CUDA when available. On CUDA, training now enables
automatic mixed precision, TF32 matmul, pinned DataLoader memory, and
non-blocking host-to-device copies. `--batch-size auto` probes CUDA memory and
uses the largest training batch that fits the current model and dataset. Use
`--max-batch-size` to cap that probe, or `--no-amp` / `--no-pin-memory` when
debugging numerical or transfer behavior. Use `--latent-norm layer` or
`--latent-norm none` only for ablation runs; the BatchNorm latent improved
reconstruction, latent rank, and metric-neighborhood preservation in screening.

With the current cached baseline matrix on an RTX 5070 Ti, the auto probe fits
the whole 80% training split in one batch:

- rows: `5518`
- features: `147`
- auto batch size: `4414`
- semantic-cardinality recipe: `metrics_embedding_dim=80`,
  `metrics_hidden_dims=(224, 112)`, `noise_std=0.003`,
  `latent_dropout=0.05`, `latent_decorrelation_weight=0.0005`,
  `mask_prob=0.0`
- previous LayerNorm recipe, 7200 epochs: validation MSE `0.01780`,
  validation MAE `0.09147`, latent effective rank `18.80`
- BatchNorm + 80-wide metrics without latent decorrelation, 5000 epochs, 3
  seeds: mean validation MSE
  `0.01287`, mean validation MAE `0.07903`, mean latent effective rank
  `35.39`, mean participation rank `17.34`, mean neighbor recall@10 `0.670`
- BatchNorm + 80-wide metrics with `latent_decorrelation_weight=0.0005`, 5000
  epochs, 3 seeds: mean validation MSE `0.01255`, mean validation MAE
  `0.07762`, mean latent effective rank `37.04`, mean participation rank
  `18.93`, mean neighbor recall@10 `0.669`
- BatchNorm + 80-wide metrics with `latent_dropout=0.05` and
  `latent_decorrelation_weight=0.0005`, 5000 epochs, 3 seeds: mean validation
  MSE `0.01205`, mean validation MAE `0.07678`, mean latent effective rank
  `39.45`, mean participation rank `20.53`, mean neighbor recall@10 `0.671`,
  mean distance correlation `0.801`
- BatchNorm + 96-wide metrics with `latent_dropout=0.05` and
  `latent_decorrelation_weight=0.0005`, 5000 epochs, 1 seed: validation MSE
  `0.01133`, latent effective rank `38.31`, neighbor recall@10 `0.677`
- BatchNorm + 112-wide metrics, 5000 epochs, 3 seeds: mean validation MSE
  `0.01079`, mean latent effective rank `32.99`, mean neighbor recall@10
  `0.679`
- BatchNorm + 256-wide metrics, 5000 epochs, 1 seed: validation MSE
  `0.00887`, latent effective rank `30.19`, neighbor recall@10 `0.692`

The 80-wide recipe with light decorrelation and latent dropout maximizes
independent latent variation while still improving reconstruction over the older
developed encoder. Heavier decorrelation produced larger rank numbers, but
reduced metric-neighborhood preservation, so it is not the default. The 96-,
112-, and 256-wide recipes are useful accuracy-biased ablations, but they expose
fewer independent latent directions than the selected 80-wide dropout recipe.
`neighbor_recall@k` and `distance_corr` are diagnostics for whether latent space
preserves metric-space neighborhoods; they are not clustering outputs.

Rejected production probes at 2500 epochs:

- latent Gaussian noise (`0.01` to `0.05`) and heavier latent dropout (`0.075`)
  increased validation MSE without increasing effective rank.
- metric-similarity preservation (`0.003` to `0.03`) improved distance
  correlation/neighbor recall, but reduced latent effective rank into the
  low-to-mid 30s, so it compressed the semantic space rather than maximizing
  grouping capacity.

Full-batch training does fewer optimizer steps per epoch than smaller batches,
so use a higher epoch count when `--batch-size auto` resolves to the whole
training split.

When `--metric-columns` is omitted, the default full-game metric catalogue is
used. The CSV must contain those metric columns or the source columns needed to
derive them.

## Test Commands

Run the focused autoencoder tests:

```bash
uv run pytest tests/classification/test_champion_semantics.py -q
```

Run all classification tests:

```bash
uv run pytest tests/classification -q
```

Lint the module and focused tests:

```bash
uv run ruff check app/classification/champion_semantics.py \
  tests/classification/test_champion_semantics.py
```

## CLI Smoke Test

This creates a tiny normalized synthetic CSV, trains for two epochs, and writes
latents to `/tmp/champion_latents.csv`.

```bash
uv run python - <<'PY'
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 24
pd.DataFrame(
    {
        "champion_id": np.arange(n) % 4,
        "teamposition_id": np.arange(n) % 3,
        "build_id": np.arange(n) % 2,
        "damage_per_min": rng.normal(0.0, 1.0, n),
        "gold_per_min": rng.normal(0.0, 1.0, n),
        "cc_per_min": rng.normal(0.0, 1.0, n),
    }
).to_csv("/tmp/champion_semantics_smoke.csv", index=False)
PY

uv run python -m app.classification.champion_semantics \
  --csv /tmp/champion_semantics_smoke.csv \
  --metric-columns damage_per_min gold_per_min cc_per_min \
  --epochs 2 \
  --batch-size 8 \
  --device auto \
  --noise-std 0.01 \
  --mask-prob 0.1 \
  --latent-dropout 0.05 \
  --latent-decorrelation-weight 0.0005 \
  --latent-output /tmp/champion_latents.csv
```
