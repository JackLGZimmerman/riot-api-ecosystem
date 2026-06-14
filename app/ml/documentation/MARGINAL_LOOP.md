# Pregame Marginal Signal Loop

Living design document for the iterative loop extracting remaining draft-time
signal from the 1vX surface through the leakage-free pregame marginal path
(`HGNN_BUILD_INTENT.md` Phase A+). Each iteration: design → execute →
measure (test acc / NLL via `python -m app.ml.marginal_eval`) → re-direct.

Hard constraints (every iteration):

- No leakage of the game outcome into accepted scoring. Accepted modes never
  read a held-out row's observed `build_id`; all aggregations are
  `split = 'train'` only.
- Draft-time information only: champions, roles, bans, train-only build
  catalog/candidate worlds, and patch/season metadata only through an explicit
  runtime provider. No observed final builds, player information, summoner
  spells, runes, rank, or PUUID.
- Detailed historical aggregations over draft-visible keys are allowed.
- The model serves the RL drafting environment; the conditional
  `P(win | draft, build)` surface must remain scoreable per concrete world.

## State Entering Iteration 1 (2026-06-12)

| Path | Artifact | Test acc | Test NLL |
| --- | --- | ---: | ---: |
| Oracle observed-build (diagnostic) | 6-seed bias-only | 0.58367 | 0.66964 |
| Marginal `W=128, k=3` (raw) | 6-seed (loop baseline) | 0.56306 | 0.68065 |
| Modal `W=1` (raw) | 6-seed | 0.55859 | 0.68259 |
| Marginal `W=128, k=3` | 3-seed affine (superseded) | 0.56186 | 0.68152 |
| Modal `W=1` | 3-seed affine (superseded) | 0.55798 | 0.68432 |

The 6-seed refresh alone improved the marginal row by +0.12pp acc / −0.0009
NLL over the 3-seed record. Raw (uncalibrated) W=128 ECE is 0.001; the
train-fitted bias slightly hurts test NLL, so raw metrics are the baseline.

Known structural facts that shape direction:

- `P(build | champ, role)` is deterministic given inputs the model already
  sees — unconditioned marginalisation cannot add information, only honest
  evaluation. Observed final build labels remain oracle diagnostics only.
- ~40–45% of the final-label win association is outcome-side inflation
  (15-min label check); intent share is ≤~57%. Phase B (time-capped label)
  is the only lever on the label itself; it is expensive and stays gated.
- 1vX prior-quality enrichment is closed by three bounds (EXPERIMENTS.md,
  2026-06-12); do not revisit (champ, role, build)-keyed feature residuals.
- Patch residuals require a real serving provider before promotion. The
  observed-build patch-restore run is closed as a tiny diagnostic gain; the
  accepted pregame W=128 seed-9 rescore regressed below the modal floor.
- Higher-order train-only relationship residuals are the next plausible
  draft-generic probe, but must clear logit-only and shuffled controls before
  cache/model wiring.
- The `W=512,k_slot=3` catalog sweep improved retained mass but regressed
  accuracy/NLL below the W=128 baseline and modal floor, so catalog-size
  expansion is closed as a standalone lever.

## Iteration 1 — Design

Run serial GPU work only (WSL2: one cache-heavy job at a time). Sidecar audits
may run in parallel, but implementation remains top-level owned.

### Track R1 — baseline refresh on the promoted 6-seed artifact (closed)

The modal and `W=128,k_slot=3` records have been refreshed on the promoted
6-seed artifact and now define the raw loop baseline. The pre-registered
full-test `W=512,k_slot=3` uncalibrated sweep was run under
`app/ml/data/experiments/20260613_2308_w512_catalog/` and rejected:

1. `marginal --worlds 512 --k-slot 3 --no-calibrate` — retained mass improved
   to mean `0.7823`, but raw metrics regressed to `0.55453` / `0.68357`.

Executed directly by the orchestrator as background runs (run-only work is
not delegated). Output paths are timestamped under
`app/ml/data/experiments/<timestamp>_<lever>/`.

### Current loop directions

1. Patch-side prior audit: observed-build seed4-9 is closed as small diagnostic
   lift, and the seed9 W=128 accepted marginal rescore is negative. Keep the
   patch lever rejected unless a future same-seed control plan is explicitly
   reopened.
2. Higher-order relationship probe: frozen-logit residual over train-only 1v1
   and 2vX aggregates first, with LOO train features and shuffled controls.
   Exact 2v1 remains strict-gated follow-up material; exact 3v1 is closed until
   a coarser backoff table exists.
3. Marginal catalog sweep: `W=512,k_slot=3` is rejected; do not expand to
   `W=512,k_slot=5` unless a separate model-side change first improves raw
   W=128 metrics.
4. Time-capped build-label scout: label-only diagnostics from timeline item
   purchases; never predictor input.
5. Encoder architecture scout: no rebuild unless low-cost residual probes beat
   controls and remain draft-servable.

### Iteration acceptance / decision rules

- Baseline = raw `W=128,k=3` 6-seed row (`0.563064` / `0.680652`).
- World-count changes must improve both raw accuracy and raw NLL against the
  W=128 baseline before they stay open.
- Every experiment, including negative/insignificant runs, is timestamped in
  `EXPERIMENTS.md` with artifact path, allowed inputs, forbidden inputs not
  used, metrics, controls, and verdict.

## Iteration Log

| Iter | Lever | Test acc | Test NLL | Verdict |
| --- | --- | ---: | ---: | --- |
| 1 | 6-seed baseline refresh | 0.56306 | 0.68065 | raw W=128 baseline accepted |
| 2 | patch restore observed-build diagnostic | 0.57373 | 0.67476 | tiny diagnostic lift; not accepted pregame |
| 3 | patch restore seed9 W=128 marginal | 0.55075 raw / 0.55187 cal | 0.68509 raw / 0.68492 cal | rejected; below modal floor, artifact `20260613_2238_patch_seed9_w128` |
| 4 | relation aggregate coverage probe | n/a | n/a | exact 1v1/2vX support sufficient for frozen probe; exact 3v1 closed, artifact `20260613_2306_relation_table_probe` |
| 5 | W=512,k=3 catalog sweep | 0.55453 | 0.68357 | rejected; below W=128 baseline and modal floor, artifact `20260613_2308_w512_catalog` |
