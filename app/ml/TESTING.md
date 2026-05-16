# Hyper-parameter Testing Procedure

Maintenance: update this file when the sweep protocol or run provenance changes. Keep it procedural; put result tables and recommendations in `OPTIMISATIONS.md`, and only copy stable defaults into `README.md`.

Run isolated training jobs on the current token model. Each trial uses a fresh Python process, seed `42`, BF16 AMP, fused AdamW, `torch.compile(mode="reduce-overhead")`, and final test evaluation from that trial's best validation-loss checkpoint.

Live training defaults `TrainConfig.checkpoint_dir` to `app/ml/data/`. Sweeps override it to a run-specific subdirectory under `app/ml/data/checkpoints/` so each trial's `best.pt` is preserved instead of overwriting the live run.

## Capacity x Regularization Grid Sweep

Script: `app/ml/run_grid_sweep.py`. Runs a 5x5 grid (25 trials) crossing a joint capacity tier (`d_model` / `n_layers` / `dim_feedforward`) with a joint regularization tier (`dropout` / `attention_dropout` / `weight_decay`). Every other axis stays at the current `ModelConfig` / `TrainConfig` defaults (`n_heads=4`, `pooling="gated"`, `head_hidden=256`, `head_dropout=0.0`, `lr=5e-5`, `warmup_steps=125`, `batch_size=16384`, `target_min/max=0.15/0.85`, `attention_diagnostics_interval=0`, `train_monitor_samples=50_000`).

Tiers:

```text
capacity        d_model  n_layers  dim_feedforward
  C0              192        3          768
  C1 (default)    256        3         1024
  C2              256        4         1536
  C3              384        3         1536
  C4              384        4         2048

regularization  dropout  attention_dropout  weight_decay
  R0              0.05         0.05            1e-3
  R1 (default)    0.15         0.10            5e-3
  R2              0.25         0.15            1e-2
  R3              0.35         0.20            2e-2
  R4              0.45         0.25            5e-2
```

Per-trial horizon: `epochs=150`. Prior LR sweep best epochs ranged epoch `111`-`146`, so 150 is a known-sufficient horizon; no early stopping.

Run the full sweep:

```bash
CLICKHOUSE_HOST=localhost python -m app.ml.run_grid_sweep
```

Output layout (one directory per trial under the sweep root):

```text
app/ml/data/checkpoints/capacity_reg_grid_<YYYYMMDD_HHMMSS>/
  t00_d192_l3_ff768_dp0.05_ad0.05_wd1e-03/
    best.pt
    metrics.jsonl
    metrics_latest.json
    tb/<metrics_stem>_<YYYYMMDD_HHMMSS>/   # TensorBoard event files
  t01_...
  ...
  sweep_summary.json
```

Inspect all 25 trials together in TensorBoard:

```bash
uv run tensorboard --logdir app/ml/data/tensorboard
# or, for one sweep only:
uv run tensorboard --logdir app/ml/data/checkpoints/capacity_reg_grid_<ts>
```

Per-trial TensorBoard mirrors the curated scalar tags from `app/ml/utils/tensorboard.py`: train and validation `loss/*`, `quality/*` (accuracy, AUC), `calibration/*` (Brier, ECE), `central_475_525/*`, `generalization/*`, `predictions/*`, `attention/*`, `throughput/*`, `optimization/*`, plus the final `test/*` row. The full per-epoch field set is available in `metrics.jsonl`.

Per-trial wall time scales with capacity tier. At the current default-tier throughput of `~8s` per epoch:

```text
C0: ~6s/epoch  -> ~15 min/trial
C1: ~8s/epoch  -> ~20 min/trial   (current default tier)
C2: ~13s/epoch -> ~33 min/trial
C3: ~14s/epoch -> ~36 min/trial
C4: ~23s/epoch -> ~58 min/trial
```

Expected total sweep wall time: `~11`-`13` hours for the full 5x5 grid.

The parent process dispatches one subprocess per trial and waits for it before starting the next, so CUDA memory is fully released between trials. On completion the parent prints a table sorted by best validation loss and writes `sweep_summary.json` containing per-trial `best_val` (epoch, val loss/accuracy/Brier/ECE plus matching `train_monitor_*` from the same epoch) and `final_test` rows.

## Selecting The Winner

Rank by validation loss from the best validation checkpoint. Validate the rank with Brier and ECE: a tie on val loss should defer to the lower Brier and lower ECE. Cross-check against the final-test row from the same trial; a winner whose validation rank does not survive on test is a noise hit, not a recommendation.

Read `gen_*` from the same epoch (`gen_loss_gap`, `gen_brier_gap`) to interpret what the winner is doing. If the winning cell sits on an edge of the grid (e.g. capacity `C4` x regularization `R4`), schedule a follow-up sweep that extends that edge before recommending defaults.

## Recording Results

When the sweep finishes, copy a compact protocol + table + conclusion block into `OPTIMISATIONS.md` under a dated section (see existing sweep entries for the format). Capture, at minimum, per-trial: best epoch, val loss/accuracy/Brier/ECE, test loss/accuracy/Brier/ECE, and median `samples/s`. Promote the winning configuration into `README.md` only if it beats the current defaults on validation loss without regressing Brier, ECE, or throughput.
