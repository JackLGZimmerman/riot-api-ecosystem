# Hyper-parameter Testing Procedure

Maintenance: keep this file procedural. Per-sweep result tables and recommendations go in `OPTIMISATIONS.md`; stable defaults are promoted into `README.md` only after they beat the current live defaults.

## Invariant

A sweep trial always inherits live `ModelConfig` / `TrainConfig` defaults from [config.py](../config.py) and overrides **only** the axes under test plus per-trial isolation paths (`checkpoint_dir`, `metrics_dir`, `tensorboard_dir`, `tensorboard_run_name`). Do not copy hyperparameter values into this file or into sweep scripts — read them from `ModelConfig()` / `TrainConfig()` at trial construction time. This keeps every sweep comparable to live training without manual sync.

Per-trial isolation: each trial runs in a fresh Python subprocess so CUDA memory, `torch.compile` cache, and global RNG state cannot leak between runs. Live training's `TrainConfig.checkpoint_dir` defaults to `app/ml/data/`; sweeps override it to a run-specific subdirectory under `app/ml/data/checkpoints/` so each trial's `best.pt` is preserved instead of overwriting the live run.

## Sweep Layout

One subprocess per trial. Override only the axes under test plus per-trial isolation paths. Each trial writes:

```text
app/ml/data/checkpoints/<sweep_name>_<YYYYMMDD_HHMMSS>/
  <trial_name>/
    best.pt
    metrics.jsonl
    metrics_latest.json
    trial.log
  <sweep_name>_summary.md
```

Point TensorBoard at the shared root to surface every trial:

```bash
uv run tensorboard --logdir app/ml/data/tensorboard
```

## Selecting The Winner

Rank by validation loss from the best-validation-loss checkpoint. Validate the rank with Brier and ECE: a tie on val loss should defer to lower Brier and lower ECE. Cross-check against the final-test row from the same trial; a winner whose validation rank does not survive on test is a noise hit, not a recommendation.

Read `gen_*` from the same epoch (`gen_loss_gap`, `gen_auc_gap`) to interpret what the winner is doing. If the winning cell sits on an edge of the swept axis, schedule a follow-up sweep that extends that edge before recommending defaults.

For multi-seed sweeps, compare mean ± std across seeds, not single-seed numbers. A per-axis edge that is smaller than the within-seed std of either competing arm is noise.

## Recording Results

When the sweep finishes, copy a compact protocol + table + conclusion block into `OPTIMISATIONS.md` under a dated section (see existing sweep entries for the format). The protocol block in `OPTIMISATIONS.md` should snapshot the *actual* live-default values used at sweep time — that file is a historical record, unlike this one. Capture, at minimum, per trial: best epoch, val loss/accuracy/Brier/ECE, test loss/accuracy/Brier/ECE, and median `samples/s`.

Promote the winning configuration into live `ModelConfig` / `TrainConfig` defaults and update `README.md` only if it beats the current defaults on validation loss without regressing Brier, ECE, or throughput.
