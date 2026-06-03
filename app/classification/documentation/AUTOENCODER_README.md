# Full-Game Autoencoder

This module is a small PyTorch baseline for learning full-game behavioral
identity representations from normalized historical profile and context rows.

The implementation lives in `app/classification/full_game_encoder.py`. It is
standalone; exported latents can be used by the optional HGNN identity-encoder
sidecar path and semantic-context head, which remain disabled by default.

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

Metric columns should describe normalized historical behavior only. Do not
include row-support/count metadata such as `matchups`.

Metric columns are configurable. If they are omitted, the module uses the
classification full-game metric catalogue:

- every raw source metric in `ALL_METRICS` (66)
- every derived metric in `DERIVED_METRIC_FUNCS` (89)
- every context feature in `CONTEXT_FEATURE_NAMES` (60)

That is `215` full-game inputs by default. The complete list for this encoder
and its sibling encoders lives in [ENCODER_METRICS.md](ENCODER_METRICS.md).

The derived metrics cover three per-identity families:

- **Ratios / normalisations** — one metric over another, e.g.
  `totaldamagedealttochampions_to_goldearned_ratio` (damage per gold) or the
  physical/magic/true `*_share` splits of dealt and taken damage.
- **Differences** — signed decompositions, e.g. `self_heal`
  (`totalheal - totalhealsonteammates`), `net_kills`, `net_combat_damage`,
  `structure_net_control`.
- **Team-aggregate participation / matchup** — these need the four teammates and
  the same-role opponent, so they are *not* derivable from a plain identity row.
  They live in `context_features.py` (`<metric>_team_share` participation,
  `<metric>_team_concentration` carry, `<metric>_vs_role_opponent_diff/advantage`
  matchup).

Derived metrics can be requested directly if their source columns are present in
the frame. The dataset also accepts `smoothed_<metric>` columns for source
metrics.

`full_game_metric_columns()` now includes the 60 context features. They are not
derivable during dataset construction, so the input frame must already carry
those (smoothed) columns, built with
`EmbeddingConfig(include_context_features=True)`. Use
`full_game_metric_columns(include_context=False)` or CLI `--profile-only` only
for legacy 155-column profile-only frames.

## Architecture

`FullGameEncoder` builds a metrics embedding from a small MLP over the
normalized full-game metric vector and always fuses it with champion, role, and
build identity embeddings. This branch is fixed at the `(champion, role, build)`
grain by construction: the latent represents the full identity together with its
behavior. Champion static stats live separately in the champion-level
`static_identity_encoder.py`.

The concatenated identity-plus-metrics vector is passed through a fusion MLP. The
fusion output is normalized and becomes the full-game latent vector. The default
is latent `BatchNorm`, which encourages more dimensions to carry population-level
variation for later grouping. `LayerNorm` and no normalization remain available
for ablation runs.

`FullGameAutoencoder` wraps the encoder and adds a decoder MLP that reconstructs
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

- `latent_dim=640`
- `champion_embedding_dim=16`
- `teamposition_embedding_dim=4`
- `build_embedding_dim=8`
- `metrics_embedding_dim=160`
- `metrics_hidden_dims=(320, 160)`
- `fusion_hidden_dims=(128,)`
- `decoder_hidden_dims=(512, 384)`
- `dropout=0.0`
- `latent_dropout=0.10`
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

from app.classification.full_game_encoder import (
    FullGameProfileDataset,
    evaluate_autoencoder,
    extract_full_game_latents,
    train_from_dataframe_or_csv,
)

metric_columns = ("damage_per_min", "gold_per_min", "cc_per_min")
frame = pd.read_csv("profiles.csv")

model, history = train_from_dataframe_or_csv(
    frame,
    metric_columns,
    epochs=7200,
    batch_size=1024,
    device="auto",
    noise_std=0.003,
    mask_prob=0.0,
    latent_decorrelation_weight=0.0005,
)

dataset = FullGameProfileDataset(frame, metric_columns)
dataloader = DataLoader(dataset, batch_size=int(history[-1]["batch_size"]), shuffle=False)
scores = evaluate_autoencoder(model, dataloader, "auto", neighbor_k=10)
latents = extract_full_game_latents(model, dataloader, "auto")
```

`extract_full_game_latents(...)` returns one row per dataloader sample:

- `champion_id`
- `teamposition_id`
- `build_id`
- `latent_0 ... latent_n`

## CLI Usage

Train from a CSV and write latent vectors:

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

When `--metric-columns` is omitted, the CLI expects all 215 default full-game
metrics. Use `--profile-only` for old 155-column matrices, or provide
`--metric-columns` for targeted experiments and smoke tests.

For semantic grouping, treat the `champion_id`, `teamposition_id`, and
`build_id` columns as the identity key and cluster only the `latent_*` columns.
The default BatchNorm latent is the grouping surface; if a downstream clustering
method needs exactly equal feature scale, z-score the exported latent matrix
outside the encoder on the training/export set. Keep `--neighbor-k 10` enabled
for regeneration runs so metric-neighborhood recall and distance correlation are
reported alongside reconstruction loss and latent rank.

For HGNN semantic-context experiments, export all three identity latent blocks
and support in the sidecar artifact. The recommended model variant is the
all-three sidecar run with `use_identity_semantic_context_head=True`, so each
slot can be evaluated against support-weighted ally and enemy latent context.

`device="auto"` uses CUDA when available. On CUDA, training now enables
automatic mixed precision, TF32 matmul, pinned DataLoader memory, and
non-blocking host-to-device copies. The default batch size is `1024`; in current
screening this gave much better fixed-epoch convergence than full-batch
training while preserving BatchNorm latent quality. `--batch-size auto` still
probes CUDA memory and uses the largest training batch that fits the current
model and dataset. Use `--max-batch-size` to cap that probe, or `--no-amp` /
`--no-pin-memory` when debugging numerical or transfer behavior. Use
`--latent-norm layer` or `--latent-norm none` only for ablation runs; the
BatchNorm latent improved reconstruction, latent rank, and metric-neighborhood
preservation in screening.

Historical full-game screening on an RTX 5070 Ti, before the current 215-feature
default, found that the auto probe fit the whole 80% training split in one
batch:

- rows: `5518`
- features: `147` (pre-expansion baseline; the legacy profile-only surface is
  now `155`, and the default all-metrics surface is `215`)
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

That older 80-wide recipe maximized independent latent variation on the
pre-expansion surface. Current 215-feature screening selected the 160-wide
metric branch with a 640-d latent, batch size `1024`, a wider `(512, 384)`
decoder, and `latent_dropout=0.10`: across 3 local 300-epoch seeds it reduced
mean clean MSE while raising latent effective rank versus the previous
215-feature default. Mean neighbor recall@10 was slightly lower, so larger
fusion layers and latents were rejected when they spent too much metric-space
neighborhood preservation for rank. Heavier decorrelation produced larger rank
numbers in earlier probes, but reduced metric-neighborhood preservation, so the
light `0.0005` penalty remains the default.
`neighbor_recall@k` and `distance_corr` are diagnostics for whether latent space
preserves metric-space neighborhoods; they are not clustering outputs.

Rejected production probes at 2500 epochs:

- latent Gaussian noise (`0.01` to `0.05`) and heavier latent dropout (`0.075`)
  increased validation MSE without increasing effective rank.
- metric-similarity preservation (`0.003` to `0.03`) improved distance
  correlation/neighbor recall, but reduced latent effective rank into the
  low-to-mid 30s, so it compressed the semantic space rather than maximizing
  grouping capacity.

Full-batch training does fewer optimizer steps per epoch than smaller batches.
Use a higher epoch count when `--batch-size auto` resolves to the whole training
split; for fixed-epoch regeneration, prefer the default `1024`.

When `--metric-columns` is omitted, the default all-metrics catalogue is used.
The CSV must contain the 60 context columns directly, plus either direct raw /
derived profile metric columns or the source columns needed to derive the
profile metrics.

## Current Regeneration Checks

Local checks on 2026-06-03:

| Encoder | Inputs evaluated | Rows | Output |
| --- | ---: | ---: | --- |
| Full-game | 215 | 5518 identities | 3-seed mean clean MSE `0.03787`; clean MAE `0.14123`; neighbor recall@10 `0.711`; distance corr `0.764`; latent effective rank `42.23` |
| Static identity, default | 47 | 171 champions | 3-seed mean clean MSE `0.000078`; champion recovery `1.000`; latent effective rank `21.53` |
| Static identity, compact check | 47 | 171 champions | clean MSE `0.03368`; champion recovery `1.000` at `latent_dim=16` |
| Temporal | 47 buckets x 51 metrics | 5518 identities | 3-seed mean all-row masked MSE `0.05082`; train masked MSE `0.06081`; latent effective rank `72.13` |

The full-game check used the default 640-d latent recipe for 300 epochs on CUDA
with batch size `1024` after rebuilding the local context base to the current
60-feature schema. The reported row is the mean across seeds `37`, `41`, and
`43`; the baseline matrix now validates at `(5518, 215)`.

The temporal tensor cache contained `217184` observed bucket cells
(`11076384` observed scalar values), or `0.8374` bucket coverage. The untrained
temporal masked MSE was `3.4782`, so the trained score is a large
reconstruction improvement.

## Issues And Fixes

- Full-game 215-column evaluation depends on current ClickHouse base-table
  schemas. A stale catalogue hash or missing context columns means old 147/155
  outputs are not valid for the all-metrics surface. Rerun
  `build_classification_tables()`; the builder recreates materialized tables
  before loading so schema changes cannot survive behind
  `CREATE TABLE IF NOT EXISTS`.
- Context features cannot be derived from a plain per-identity CSV. Build frames
  with `EmbeddingConfig(include_context_features=True)` or use `--profile-only`
  for legacy 155-column data.
- Static identity reconstruction is dictionary recovery, not a held-out
  generalization test. Use `champion_recovery_accuracy` alongside MSE and keep
  `validate_static_input_columns` in front of any imported feature table.
- Temporal reconstruction is mask-aware; comparing its MSE to full-game MSE is
  not apples-to-apples. Report observed bucket coverage and validation masked
  MSE together.

## Test Commands

Run the focused autoencoder tests:

```bash
uv run pytest tests/classification/test_full_game_encoder.py -q
```

Run all classification tests:

```bash
uv run pytest tests/classification -q
```

Lint the module and focused tests:

```bash
uv run ruff check app/classification/full_game_encoder.py \
  tests/classification/test_full_game_encoder.py
```

## CLI Smoke Test

This creates a tiny normalized synthetic CSV, trains for two epochs, and writes
latents to `/tmp/full_game_identity_latents.csv`.

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
).to_csv("/tmp/full_game_encoder_smoke.csv", index=False)
PY

uv run python -m app.classification.full_game_encoder \
  --csv /tmp/full_game_encoder_smoke.csv \
  --metric-columns damage_per_min gold_per_min cc_per_min \
  --epochs 2 \
  --batch-size 8 \
  --device auto \
  --noise-std 0.01 \
  --mask-prob 0.1 \
  --latent-dropout 0.10 \
  --latent-decorrelation-weight 0.0005 \
  --neighbor-k 2 \
  --latent-output /tmp/full_game_identity_latents.csv
```

## Sibling Encoders

The full-game branch is one of three independent identity encoders. The two
behavioral branches key on the `(champion, role, build)` identity; the static
branch is **champion-level** (champion base stats are a function of champion
only). None consumes win rates, matchup/synergy priors, support counts, or
challenge data. The full-game branch's default context features are *observed
behavioral* team-share and role-matchup quantities, not win-rate or synergy
priors; they can be disabled only for legacy profile-only runs.

| Encoder | Module | Identity grain | Input surface | Latent | Identity embeddings |
| --- | --- | --- | --- | --- | --- |
| Full-game | `full_game_encoder.py` | champion/role/build | 215 metrics (66 raw + 89 derived + 60 context); legacy 155 profile-only with `--profile-only` | 640 | always (champion+role+build) |
| Static identity | `static_identity_encoder.py` | champion | 47 deterministic champion static features | 128 | none (champion = its static vector) |
| Temporal | `temporal_autoencoder.py` | champion/role/build | `(47 buckets x 51 metrics)` standardized trajectory tensor | 416 | always (champion+role+build) |

Each encoder is fixed at its own grain; there is no switch to toggle. The two
behavioral branches always embed the full `(champion, role, build)` identity
alongside the behavioral signal — they must represent the full identity of every
row they ingest — so each latent is a complete identity-plus-behavior
representation. The static branch is champion-level: champion identity enters
only through the deterministic static stat vector.

For identity-level semantic groups, use full-game latents as the primary
per-identity surface because they directly reconstruct the 215-column behavioral
profile at the `(champion, role, build)` grain. Static latents are champion-only
auxiliary groups; temporal latents are trajectory groups and are best compared
with masked MSE and latent-rank diagnostics before mixing them into a shared
clustering run.

### Static Identity Autoencoder

Champion-level dictionary branch: champion base stats (plus level-18 derived
stats) as the only input, one row per champion via `static_identity_frame`.
Role and build never parametrise this encoding, so there are no role/build
inputs and no role/build reconstruction. The input is screened by
`validate_static_input_columns`, which rejects any empirical prior, win-rate,
matchup/synergy, or support-count column. Reconstruction is continuous MSE with
latent BatchNorm and a light latent decorrelation penalty. Input noise and
decoder-side latent dropout default to off because the source is a deterministic
dictionary, not a noisy behavioral sample.

Static stats are a **fixed champion dictionary**, so the encoding only matters if
the original champion is recoverable from it — there is no held-out split.
`evaluate_static_autoencoder` reports `champion_recovery_accuracy`: the fraction
of champions whose decoded static vector is nearest their own true stats among
all champions. On the current 171 local dictionary champions, recovery is
`1.000` from 1200 epochs with the default 128-d latent (3-seed mean clean MSE
`0.000078`) and also at `latent_dim=16` (clean MSE `0.03368`), so every champion
is recoverable through a tight bottleneck below the 47 input features.

```bash
uv run pytest tests/classification/test_static_identity_encoder.py -q
```

### Temporal Autoencoder

Reconstructs the mask-aware temporal tensor from
`app/classification/embeddings/temporal.py`; buckets a short-lived identity never
reached contribute no loss. By default, the same mask is also passed into the
encoder so unobserved buckets are zeroed before latent construction. Latent
BatchNorm only. `evaluate_temporal_autoencoder` reports masked MSE plus the same
latent diagnostics (`effective_rank`, `participation_rank`, `mean_abs_corr`).

The full-game semantic recipe was screened here and **not** adopted. Older
2-seed screening at 600 epochs on a held-out 20% of 5518 identities found:

- baseline (no recipe): masked MSE `0.1858`, latent effective rank `43.33`
- decoder-side `latent_dropout=0.05`: masked MSE `0.1942`, effective rank
  `41.13` — strictly worse on both
- `latent_decorrelation_weight=1e-3`: masked MSE `0.1859`, effective rank
  `43.50` — neutral

Current 215-feature-era screening selected a much wider temporal default
(`metric_embed_dim=96`, `hidden=1536`, `latent_dim=416`, `dropout=0.02`,
`zero_unobserved_input=True`). Across 3 local 200-epoch seeds on the cached
train temporal tensor, the model measured mean all-row masked MSE `0.05082`,
mean train masked MSE `0.06081`, and mean latent effective rank `72.13`.
Smaller 384-d latents were a lower-loss frontier point on one seed but retained
fewer semantic dimensions; no-dropout variants reduced MSE slightly while
dropping effective rank. Decoder-side latent dropout and latent decorrelation
still default to off (`latent_dropout=0.0`, `latent_decorrelation_weight=0.0`);
they remain available for ablation.

```bash
uv run pytest tests/classification/test_temporal.py -q
```
