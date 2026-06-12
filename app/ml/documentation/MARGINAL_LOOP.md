# Pregame Marginal Signal Loop

Living design document for the iterative loop extracting remaining draft-time
signal from the 1vX surface through the leakage-free pregame marginal path
(`HGNN_BUILD_INTENT.md` Phase A+). Each iteration: design → execute →
measure (test acc / NLL via `python -m app.ml.marginal_eval`) → re-direct.

Hard constraints (every iteration):

- No leakage of the game outcome into accepted scoring. Accepted modes never
  read a held-out row's observed `build_id`; all aggregations are
  `split = 'train'` only.
- Draft-time information only: champions, roles, summoner spells, runes. No
  player information of any kind (draft-generic constraint).
- Detailed historical aggregations over draft-visible keys are allowed.
- The model serves the RL drafting environment; the conditional
  `P(win | draft, build)` surface must remain scoreable per concrete world.

## State Entering Iteration 1 (2026-06-12)

| Path | Artifact | Test acc | Test NLL |
| --- | --- | ---: | ---: |
| Oracle observed-build (diagnostic) | 6-seed bias-only | 0.58367 | 0.66964 |
| Marginal `W=128, k=3` | 3-seed affine (superseded) | 0.56186 | 0.68152 |
| Modal `W=1` | 3-seed affine (superseded) | 0.55798 | 0.68432 |

Known structural facts that shape direction:

- `P(build | champ, role)` is deterministic given inputs the model already
  sees — unconditioned marginalisation cannot add information, only honest
  evaluation. The only way to *close* oracle gap pregame is conditioning the
  build prior on draft-visible information beyond (champ, role).
- Runes and summoner spells are draft-visible (admissible by constraint) and
  are strong build-intent proxies; they enter the model today only through
  the `_nobuild`-keyed loadout priors, so a build-conditioned rune
  aggregation is genuinely new signal.
- ~40–45% of the final-label win association is outcome-side inflation
  (15-min label check); intent share is ≤~57%. Phase B (time-capped label)
  is the only lever on the label itself; it is expensive and stays gated.
- 1vX prior-quality enrichment is closed by three bounds (EXPERIMENTS.md,
  2026-06-12); do not revisit (champ, role, build)-keyed feature residuals.

## Iteration 1 — Design

Two parallel tracks, GPU serialised (WSL2: one cache-heavy job at a time).

### Track R1 — baseline refresh on the promoted 6-seed artifact (run-only)

The recorded marginal numbers are for the superseded 3-seed artifact, and the
`W=512` sweep was deferred. Three serial runs, full test split:

1. `modal --split test` (calibrated) — modal floor on the 6-seed artifact.
2. `marginal --worlds 128 --k-slot 3` (calibrated) — the loop's baseline.
3. `marginal --worlds 512 --k-slot 3 --no-calibrate` — measures the retained
   mass → metric slope; if flat vs `W=128`, the world-count lever is closed
   and conditioning is the only open lever.

Executed directly by the orchestrator as background runs (run-only work is
not delegated). Outputs: `app/ml/data/experiments/loop1_*.json`.

### Track B1 — keystone-conditioned build prior (implementation)

Replace world weights `P(b | champ, role)` with
`P(b | champ, role, keystone)` at eval time. The keystone
(`participant_perk_ids.primary_perk_1`) is chosen pregame, is admissible,
and discriminates build intent within a champion-role (e.g. AP vs AD or
tank vs damage keystones). Mixture weights move toward the true conditional;
the scored worlds and the model stay fixed.

Mechanics (decision-complete; sub-agent executes):

- Train-only counts `(championid, teamposition, primary_perk_1, build) → n`
  from `participant_stats ⋈ ml_game_split ⋈ participant_perk_ids ⋈
  participant_item_value_totals`, memory-capped, cached locally.
- Conditioned vector = nested EB: child counts restricted to the parent's
  retained labels, smoothed toward the parent distribution
  (`p = (n_b + τ_c·p_parent) / (N_child + τ_c)`, `τ_c=50`), child cell gated
  at `N_child ≥ 50` else the parent vector is served unchanged. Conditioning
  can only reweight parent-vetted profiles, never introduce new builds.
- Per-slot keystone arrays for each split, built from a pivot query that
  mirrors the cache builder's `ORDER BY matchid` row order, hard-validated
  by exact champion-id equality against the cache; stored with a
  champion-array checksum so stale arrays fail loudly.
- `marginal_eval --condition keystone` (default `none`); payload reports
  conditioned-slot share and child-support distribution. Calibration train
  scoring uses the train keystone array (train rows see their own pregame
  keystones — still draft-time information).

Leakage audit: keystone is selected before the game starts; the aggregation
is train-split only; test rows contribute only their own pregame keystone as
a lookup key. The `puuid` in the join is row alignment only (same as the
loadout features); no player identity is emitted.

### Iteration 1 acceptance / decision rules

- Baseline = R1's `W=128` calibrated row (6-seed artifact).
- Conditioned marginal must improve test NLL vs that baseline without ECE
  collapse; accuracy is secondary (promotion priority NLL-first).
- If keystone conditioning pays: iteration 2 sweeps `τ_c`/child gate on
  train scoring, then adds the summoner-spell pair and primary/sub style as
  further conditioning axes.
- If it is flat: the conditioning axis is probably mass-limited — check the
  conditioned-slot share and child-support stats before declaring the axis
  closed; the remaining lever would be Phase B (time-capped label), which
  needs an explicit go decision.
- All accepted results recorded in `EXPERIMENTS.md` with build source
  `pregame_marginal_build`.

## Iteration Log

| Iter | Lever | Test acc | Test NLL | Verdict |
| --- | --- | ---: | ---: | --- |
| 1 | 6-seed baseline refresh + W=512 + keystone conditioning | pending | pending | pending |
