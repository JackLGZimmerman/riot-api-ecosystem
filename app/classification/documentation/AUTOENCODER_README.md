# Full-Game Autoencoder

PyTorch baseline that learns full-game behavioral identity representations from
normalized historical profile/context rows. Implementation:
`app/classification/full_game_encoder.py`. Standalone — exported latents feed the
HGNN semantic MoE sidecar path used by production testing.

## Goal

One compact latent per `(champion_id, teamposition_id, build_id)` profile row,
for downstream semantic grouping / classification experiments. This baseline
only trains the representation.

## Inputs

Training data is a pandas `DataFrame`/CSV with integer embedding IDs
(`champion_id`, `teamposition_id`, `build_id`) plus normalized metric columns.
Metric columns describe **normalized behavior only** — never row-support/count
metadata (e.g. `matchups`), win/matchup/synergy priors, or challenge data.

When `--metric-columns` is omitted, the default is the full-game catalogue of
**215 inputs** = 66 raw (`ALL_METRICS`) + 89 derived (`DERIVED_METRIC_FUNCS`) +
60 context (`CONTEXT_FEATURE_NAMES`). Full list: [ENCODER_METRICS.md](ENCODER_METRICS.md).

- Raw/derived metrics are per-identity (ratios, signed differences).
- The 60 context features (`<metric>_team_share`, `_team_concentration`,
  `_vs_role_opponent_diff/advantage`) need the four teammates + same-role
  opponent, so they are **not** derivable from a plain identity row — build the
  frame with `EmbeddingConfig(include_context_features=True)`.
- `full_game_metric_columns(include_context=False)` / CLI `--profile-only` selects
  the legacy 155-column profile-only surface.

## Architecture

`FullGameEncoder`: MLP over the normalized metric vector, always fused with
champion/role/build identity embeddings, at the `(champion, role, build)` grain.
`FullGameAutoencoder` adds a decoder MLP reconstructing the clean metric vector
(MSE) with a light latent-decorrelation penalty and decoder-side latent dropout;
extracted latents stay clean. Optional denoising on the metric input only:
`noise_std` (Gaussian), `mask_prob` (zeroing). Latent norm defaults to
`BatchNorm` (`layer`/`none` available for ablation).

## Defaults

`latent_dim=640`, `champion_embedding_dim=16`, `teamposition_embedding_dim=4`,
`build_embedding_dim=8`, `metrics_embedding_dim=160`,
`metrics_hidden_dims=(320, 160)`, `fusion_hidden_dims=(128,)`,
`decoder_hidden_dims=(512, 384)`, `dropout=0.0`, `latent_dropout=0.10`,
`latent_norm="batch"`, `noise_std=0.003`, `mask_prob=0.0`,
`latent_decorrelation_weight=0.0005`. Omitted vocab sizes are inferred as
`max(id)+1` from the frame.

## Throughput Default

Use `batch_size=5518` / `--batch-size 5518` for every full-game autoencoder
experiment on the current 5518-row identity table unless the experiment is
explicitly a batch-size sweep. On the local RTX 5070 Ti, the documented
215-metric, 640-latent recipe peaked at `60,478` samples/s with the whole
identity table in one batch:

| Batch size | Samples/s |
| ---: | ---: |
| `512` | `38,349` |
| `1024` | `40,819` |
| `2048` | `47,037` |
| `4096` | `56,535` |
| `5518` | **`60,478`** |

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
    frame, metric_columns, epochs=7200, batch_size=5518, device="auto",
    noise_std=0.003, mask_prob=0.0, latent_decorrelation_weight=0.0005,
)

dataset = FullGameProfileDataset(frame, metric_columns)
dataloader = DataLoader(dataset, batch_size=int(history[-1]["batch_size"]), shuffle=False)
scores = evaluate_autoencoder(model, dataloader, "auto", neighbor_k=10)
latents = extract_full_game_latents(model, dataloader, "auto")
```

`extract_full_game_latents(...)` returns one row per sample: `champion_id`,
`teamposition_id`, `build_id`, `latent_0 ... latent_n`.

## CLI Usage

```bash
uv run python -m app.classification.full_game_encoder \
  --csv path/to/profiles.csv \
  --epochs 7200 \
  --batch-size 5518 \
  --device auto \
  --noise-std 0.003 \
  --latent-dropout 0.10 \
  --latent-decorrelation-weight 0.0005 \
  --neighbor-k 10 \
  --latent-output app/classification/data/embeddings/cache/full_game_identity_latents.csv
```

- Omitting `--metric-columns` expects all 215 default metrics (CSV must carry the
  60 context columns + raw/derived profile columns or their sources);
  `--profile-only` selects the legacy 155-column surface.
- `device="auto"` uses CUDA (AMP + TF32 + pinned/non-blocking transfers) when
  available. Use `--batch-size 5518` for current full-game experiments;
  `--batch-size auto` probes the largest fitting batch and `--max-batch-size`
  caps it for explicit sweeps; `--no-amp` / `--no-pin-memory` for debugging.
- For semantic grouping, key on the three id columns and cluster only `latent_*`;
  keep `--neighbor-k 10` so neighbor recall / distance correlation are reported.
  `neighbor_recall@k` and `distance_corr` are neighborhood-preservation
  diagnostics, not clustering outputs.

## Current Regeneration Checks (2026-06-03)

| Encoder | Inputs | Rows | Output |
| --- | ---: | ---: | --- |
| Full-game | 215 | 5518 identities | 3-seed mean clean MSE `0.03787`; MAE `0.14123`; neighbor recall@10 `0.711`; distance corr `0.764`; latent eff. rank `42.23` |
| Static identity (default) | 47 | 171 champions | 3-seed mean clean MSE `0.000078`; champion recovery `1.000`; latent eff. rank `21.53` |
| Static identity (compact) | 47 | 171 champions | clean MSE `0.03368`; recovery `1.000` at `latent_dim=16` |
| Temporal | 47×51 | 5518 identities | 3-seed mean masked MSE `0.05082`; train masked MSE `0.06081`; latent eff. rank `72.13` |

Full-game check: default 640-d recipe, 300 epochs, CUDA, batch 1024, seeds
37/41/43; baseline matrix validates at `(5518, 215)`. Temporal cache held
`217184` observed bucket cells (`0.8374` coverage); untrained masked MSE `3.4782`.

## Issues & Fixes

- 215-column eval depends on current ClickHouse base schemas; a stale catalogue
  or missing context columns invalidates old 147/155 outputs. Rerun
  `build_classification_tables()` (it recreates materialized tables before load).
- Context features can't be derived from a plain CSV — use
  `EmbeddingConfig(include_context_features=True)` or `--profile-only`.
- Static reconstruction is dictionary recovery, not held-out generalization;
  report `champion_recovery_accuracy` and keep `validate_static_input_columns` in
  front of any imported feature table.
- Temporal MSE is mask-aware; not comparable to full-game MSE. Report observed
  bucket coverage with masked MSE.

## Test / Lint Commands

```bash
uv run pytest tests/classification/test_full_game_encoder.py -q
uv run pytest tests/classification -q
uv run ruff check app/classification/full_game_encoder.py \
  tests/classification/test_full_game_encoder.py
```

### CLI Smoke Test

Tiny synthetic CSV, two epochs, latents to `/tmp/full_game_identity_latents.csv`:

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
  --epochs 2 --batch-size 8 --device auto \
  --noise-std 0.01 --mask-prob 0.1 --latent-dropout 0.10 \
  --latent-decorrelation-weight 0.0005 --neighbor-k 2 \
  --latent-output /tmp/full_game_identity_latents.csv
```

## Sibling Encoders

Three independent identity encoders. The two behavioral branches key on
`(champion, role, build)`; the static branch is **champion-level**. None consumes
win rates, matchup/synergy priors, support counts, or challenge data. Each is
fixed at its own grain (no toggle).

| Encoder | Module | Grain | Input surface | Latent |
| --- | --- | --- | --- | ---: |
| Full-game | `full_game_encoder.py` | champion/role/build | 215 metrics (66 raw + 89 derived + 60 context); 155 with `--profile-only` | 640 |
| Static identity | `static_identity_encoder.py` | champion | 47 deterministic champion static features | 128 |
| Temporal | `temporal_autoencoder.py` | champion/role/build | `(47 buckets × 51 metrics)` standardized trajectory tensor | 416 |

Use full-game latents as the primary per-identity grouping surface; static
latents are champion-only auxiliaries; temporal latents are trajectory groups
(compare with masked MSE + latent-rank first).

HGNN consumes these exported latents as frozen identity sidecars, not as labels.
The semantic group feature tensor is layered later in `app/ml/semantic_group_features.py`
as a compact, versioned summary of latent and static semantic groupings; the
HGNN learned MoE relationship head then learns each identity's response to its
own, allied, and enemy group summaries.

### Static Identity Autoencoder

Champion-level dictionary branch: champion base stats (+ level-18 derived) as the
only input, one row per champion via `static_identity_frame`. No role/build
inputs or reconstruction. `validate_static_input_columns` rejects any prior,
win-rate, matchup/synergy, or support-count column. Continuous MSE, latent
BatchNorm, light decorrelation; input noise / decoder latent dropout default off
(deterministic source). Since stats are a fixed dictionary there is no held-out
split — `evaluate_static_autoencoder` reports `champion_recovery_accuracy`
(fraction of champions nearest their own decoded vector). 171 champions recover
at `1.000` with the 128-d latent and at `latent_dim=16`.

```bash
uv run pytest tests/classification/test_static_identity_encoder.py -q
```

### Temporal Autoencoder

Reconstructs the mask-aware temporal tensor from
`app/classification/embeddings/temporal.py`; unreached buckets contribute no loss
and are zeroed before encoding (`zero_unobserved_input=True`). Latent BatchNorm
only; `evaluate_temporal_autoencoder` reports masked MSE plus latent diagnostics.
Default: `metric_embed_dim=96`, `hidden=1536`, `latent_dim=416`, `dropout=0.02`;
decoder latent dropout / decorrelation default off (available for ablation).

```bash
uv run pytest tests/classification/test_temporal.py -q
```
