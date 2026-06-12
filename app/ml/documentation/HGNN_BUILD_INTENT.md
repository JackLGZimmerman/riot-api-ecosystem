# HGNN Build Intent Plan

Last updated: 2026-06-12.

This document is the implementation plan for adding build intent to the HGNN and
RL/search stack without leaking completed-game item information. It is grounded
in the current v32 model surface (see `HGNN_CURRENT.md`) and the recorded
experiment history (`EXPERIMENTS.md`).

**Status (2026-06-12): Phase A is implemented.** The catalog/contracts live in
[build_catalog.py](../build_catalog.py), the cache-side marginalised eval in
[marginal_eval.py](../marginal_eval.py) (`python -m app.ml.marginal_eval`), the
runtime batched marginal path in [predictor.py](../predictor.py)
(`predict_marginal`), and the catalog-backed RL pool in
[pool.py](../../rl/pool.py) (`python -m app.rl.pool generate`). The dead
`use_final_build_labels=False` arm, the id-0 build default, and
`build_pool_from_priors` are deleted. Measured results are recorded in
`EXPERIMENTS.md` ("Pregame Build Marginalisation"). Sections below are the
plan of record; the "Grounding" section describes the pre-implementation tree.

## Goal

Maximise the useful effect of build information while keeping every accepted
test, serving, and RL/search path constrained to draft-phase information.

The production signal is not an observed held-out build label and not a single
synthetic "unknown" build. The only normal pregame build signal is a train-only
historical distribution:

```text
P(build | champion_id, teamposition)
```

The HGNN should score concrete train-supported builds, then average the
resulting probabilities:

```text
P(win | draft) =
  sum over joint assignments b = (b_slot0..b_slot9):
      P_HGNN(win | draft, b) * prod_slot P(b_slot | champion_slot, role_slot)
```

Do not average build ids, build embeddings, sidecar tensors, semantic features,
logits, or hidden states as a shortcut. The HGNN is nonlinear, and the sidecar
and semantic lookup surfaces are keyed by concrete `(champion, role, build)`
identities. Production marginalisation must happen over output probabilities
(for the ensemble: `sigmoid(scale * mean_logit + bias)` per world, then the
weighted average).

**Training keeps conditioning on observed final builds.** That is what makes
`P_HGNN(win | draft, build=b)` a scoreable conditional at all. The leakage
boundary is split-scoped, not global: train rows keep their observed labels;
accepted test/serving scoring must never read a held-out row's observed build.
Any guard that bans final build labels from the cache outright would destroy
the conditional model this plan depends on.

## Grounding: What The Code Did Pre-Implementation

Verified against the working tree on 2026-06-11, before Phase A landed. Kept as
the record of why each change was made; line anchors refer to that tree.

- **Build label definition.** The label is the argmax over 11 item-value
  categories of the *final inventory*
  ([5132_participant_item_value_totals_schema.sql](../../../database/clickhouse/schema/5132_participant_item_value_totals_schema.sql),
  `highest_value_label`). The pivot
  ([6900_ml_game_player_pivot_build.sql](../../../database/clickhouse/schema/6900_ml_game_player_pivot_build.sql))
  and the train-only prior
  ([6003_1vx_aggregations_build.sql](../../../database/clickhouse/schema/6003_1vx_aggregations_build.sql))
  both join it. `synergy_1vx` is already `split = 'train'` only. A
  time-capped label (Phase B) is derivable from
  `game_data.tl_item_purchased` (926M rows) joined through
  `item_value_map_dict` — verified working in the signal inventory below.
- **Model vocab.** `HGNNConfig.n_builds` defaults to 11; `build_vocab` is the
  sorted distinct train labels recorded at cache build
  ([build_dataset.py](../build_dataset.py) `_identity_meta`). One reserve
  embedding row (`index n_builds`) exists for unknown ids — it is randomly
  initialised and *untrained*.
- **Cache.** `npy-memmap-v32`. Slot order is fixed per side
  (TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY, blue 0–4 / red 5–9), so `role =
  POSITIONS[slot % 5]` — an eval harness can recover roles from slot index
  without new arrays. Both splits currently store observed final `build_id`s;
  the train side is correct by design, the test side must simply not be read
  by accepted scoring.
- **Dead draft-safe path.** `DatasetConfig.use_final_build_labels=False`
  ([config.py:59](../config.py#L59)) requires `synergy_1vx` rows with
  `build = 'unknown'`, which no SQL build produces — `_identity_meta` would
  raise. At runtime that mode maps every slot to the untrained reserve
  embedding with 0.5/0 priors ([predictor.py:197](../predictor.py#L197)).
  This path is dead and is also a *bad* baseline; replace it with marginal
  mode rather than fixing it.
- **id-0 default.** `_team_tuples` defaults a missing build to id `0`
  ([predictor.py:49](../predictor.py#L49)), which is a real vocab label.
  Must become an explicit error.
- **RL reward is already probability-weighted.** `resolve_rewards` weights the
  win matrix by `np.outer(blue_weights, red_weights)` from config
  probabilities ([reward.py:199-204](../../rl/reward.py#L199-L204)). The real
  gaps are upstream: pool weights are raw matchup counts with only a
  `min_matchups` gate and no smoothing, min-share, or fallback
  ([pool.py](../../rl/pool.py) `build_pool_from_priors`), and the sampler's
  joint top-K renormalises silently, hiding pruned mass
  ([reward.py:148-154](../../rl/reward.py#L148-L154)).
- **Vocab ordering hazard.** `WinRatePredictor.build_labels` is derived by
  sorting distinct prior keys ([predictor.py:147](../predictor.py#L147)) while
  embedding indices come from `model.config.build_vocab`
  ([predictor.py:144](../predictor.py#L144)). They coincide today only because
  both are sorted over the same set. The catalog must make
  `model.config.build_vocab` the single canonical ordering and assert
  equality at load.
- **Serving contract.** `load_predictor` fail-fasts on checkpoints with
  loadout/patch heads (`runtime_unsupported_inputs`), and the promoted
  production ensemble *has* those heads. Accepted marginalised metrics must
  therefore be computed by a cache-side eval harness (where
  `loadout_features.npy` / `patch_features.npy` exist), not through
  `WinRatePredictor`. Extending the runtime to supply patch features
  (deterministic from current patch) is a separate, deferred decision.
- **Existing tests.** `tests/ml/test_build_dataset.py`, `test_dataset.py`,
  `test_predictor.py`, `test_train_defaults.py`, `test_train_calibration.py`,
  `test_encoder_sidecar.py`, `tests/rl/test_reward.py` all exist and cover
  the touched surfaces.

## Measured Signal Inventory (2026-06-11)

Every claim in this section was measured against ClickHouse train-split data,
not assumed. Method: within each conditioning cell, the games-weighted
variance of sub-cell win rates is computed and the expected sampling-noise
contribution `(B-1) * pbar * (1-pbar) / N` is subtracted ("debiased
variance", reported in pp²; the same approach as the group audit in
`EXPERIMENTS.md`). Sub-cells gated at `n >= 20`, cells at `n >= 500`–`1000`.
Reproduction note: row-level scans of `participant_item_value_totals` joined
to splits are cheap, but any aggregation over `tl_item_purchased` (926M rows)
must run on a sampled, staged subset with `max_memory_usage` capped — an
uncapped run OOM-kills the ClickHouse container.

| Measurement (train split) | Value | Decision it drives |
| --- | --- | --- |
| `P(b\|champ,role)`: weighted top-1 share / effective builds | 0.783 / 1.95 | Marginalisation ≠ modal scoring: the modal *joint* world carries only ~9% mass (measured below). |
| Labels with ≥5% share per champ-role (weighted) | ~2.1 | `K_slot=3` per-slot truncation is near-lossless. |
| Debiased win-rate variance of the 11-label final build within (champ, role) | 4.78pp² (2.19pp RMS) | Large oracle signal by this project's standards — the conditional `P(win\|draft,build)` surface is worth building. |
| Same signal at the 4-group `BUILD_GROUPS` grain | 3.30pp² | Groups keep only ~69% of it: keep the 11-label vocab; groups stay smoothing parents, not a replacement vocab. |
| Secondary label (2nd-highest value column) within (champ, role, primary) | +6.99pp² raw association | Headroom past the primary label exists — but see the contamination split below before trusting it. |
| Hybrid ratio `v2/v1` quartiles within (champ, role, primary) | +7.21pp² raw association | Same caveat: shape discriminates, causality unknown at this grain. |
| Win/loss TV of the final label within (champ, role) | 0.0217 | An equivalent restatement of the 2.19pp RMS, not an independent check — see consequence 1. |
| 15-min purchase-derived label: agreement with final label | 66.7% | The final label is a noisy proxy for intent. |
| 15-min vs final label debiased variance, identical games | 3.06pp² vs 5.36pp² | ≥~43% of the final-label association is outcome-side inflation; ≤~57% is plausibly draft intent. |
| 8-min label | unusable | ~17% coverage, selection-biased subsample (52.6% WR). |
| Retained joint mass, real priors (mean over sampled drafts): `W=1` / `K=3,W=128` / `K=3,W=512` / `K=3,W=2048` | 0.09 / 0.66 / 0.75 / 0.78 | Sets the enumeration defaults below; a ≥0.95 joint-mass target is unreachable at any practical `W`. |
| Deterministic per-(champ,role) features (averaged historical label, shape, or group) | 0 by construction | A feature that is a function of (champ, role) alone adds no draft-time information beyond the champion identity the model already has. Rejected. |

Two analytical consequences:

1. **Association measures cannot separate causation.** Within a cell, the
   win/loss TV of the label distribution and the across-label win-rate
   variance are the same association expressed in different units
   (TV 0.0217 ↔ 2.19pp RMS here). A diagnostic comparing `P(b|champ,role)` to
   `P(b|champ,role,win)` therefore cannot tell "build choice wins games" from
   "winning shapes the final label". The only data-driven separator available
   is the **time-capped label benchmark** (label recomputed from purchases in
   the first 15 minutes via `tl_item_purchased` mapped through
   `item_value_map_dict`), which replaces the TV diagnostic as the required
   contamination measurement.
2. **Pregame marginalisation adds no information beyond champion+role.**
   `P(b | champ, role)` is a deterministic function of inputs the model
   already sees, so accepted passive metrics can at best match a well-trained
   no-build model. Phase A's value is honest evaluation (the recorded
   production test metrics are oracle-conditioned) plus the conditional
   surface RL/search needs — not headline-metric gain.

## Known Modelling Risk: Final Build Is Post-Treatment

`highest_value_label` is computed from the *final* inventory, which is partly
an outcome of the game: losing players finish fewer items, defensive pivots
correlate with the game going badly. The signal inventory quantifies this:
roughly 40–45% of the final label's win association evaporates when the label
is frozen at 15 minutes, and the 15-min label itself still embeds in-game
adaptation, so the true draft-intent share is at most ~57%. Marginalising the
conditional against a pregame prior also evaluates it under a different
conditioning distribution than it was trained on; expect accepted metrics to
sit below the oracle-conditioned reference, by design.

Required handling:

1. **Treat the safe-ablation result as the verdict**, not a tuning failure.
   If marginalisation loses NLL to the no-build baselines, contamination (not
   smoothing or pruning) is the prime suspect — the measurements above say
   the room for a passive-metric *gain* was never there.
2. **Phase B's label is intent-proxied, not richer-final.** Enriching the
   *final*-inventory label (secondary set, margin, shape) amplifies exactly
   the contaminated component — those facets are even more exposed to
   item-completion progress than the primary label. Any Phase B label
   redefinition derives from time-capped purchases (15-minute cutoff over
   `tl_item_purchased` + `item_value_map_dict`; the same item-classification
   approach as
   [8005_scaling_item_time_bins.sql](../../../database/clickhouse/schema/analytics_builds/8005_scaling_item_time_bins.sql)).
   This changes the label definition end-to-end
   (5132 → 6003 → 6900 → cache → sidecars → retrain).

## Phasing

Phase the work so the expensive, model-touching steps are gated on evidence
from the cheap ones. This is the main lever for minimising performance loss
and wasted effort.

### Phase A — primary-label marginalisation (no retrain, no cache rebuild)

Marginalise over the existing 11-label primary vocab using the already-trained
production ensemble. Everything needed already exists: the conditional model
(trained on final labels), train-only priors (`synergy_1vx`), and
`(champion, role, build)`-keyed sidecar/semantic lookups. Phase A is a pure
serving/eval/RL change plus a small catalog module:

- Catalog + prior vector derived from `synergy_1vx` train rows (counts per
  build / row sum per `(champ, role)`). **No new ClickHouse tables.**
- Cache-side marginalised eval harness for accepted test metrics.
- Batched marginal path in the predictor for RL/serving.
- Source labels, guards, deletion of the dead `use_final_build_labels=False`
  runtime arm and the id-0 default.
- Oracle ceiling and safe ablations (the decision data for Phase B).

### Phase B — intent-proxied label (conditional, expensive)

The measured evidence has already shaped Phase B: the headroom past the
primary label is real (secondary +6.99pp², shape +7.21pp² raw association)
but those facets of the *final* inventory are the most outcome-inflated, so
"richer final profiles" is rejected as the Phase B direction. Phase B, if
taken, redefines the label from 15-minute time-capped purchases (primary
label first; optionally a time-capped secondary), then: extend the label
definition in SQL, bump the cache format (v33), regenerate the sidecar and
semantic-context artifacts at the new identity grain, retrain 3 seeds,
re-promote via `promote.py`, and re-run `verify_equivalence.py`-style
no-regression checks.

Go/no-go inputs, all recorded in `EXPERIMENTS.md` first: Phase A's accepted
marginal metrics and oracle references, plus a train-side check that a model
conditioned on the time-capped label retains its oracle separation (the
aggregate-level expectation is ~3.06pp² vs the final label's 5.36pp²). Do not
start Phase B work, including contract fields it would need, before that.

## Core Data Contracts

### BuildProfile

A train-supported build atom the HGNN can score. Phase A: exactly one profile
per retained `(champion, role, primary_label)`; `profile_id ==
(champion_id, teamposition, primary_label)` canonically serialised.

Required fields (Phase A):

- `champion_id`
- `teamposition`
- `primary_label` (one of the 11 item-value categories)
- `hgnn_build_id` — index into `model.config.build_vocab`; validated against
  the loaded checkpoint at startup, never recomputed by re-sorting
- `support_count`, `support_share`, `support_tier`
- `catalog_version`

Phase B replaces `primary_label`'s source with the 15-minute time-capped
label (and may add a time-capped `secondary_label`). Final-inventory
`secondary_label_set` / `shape_bucket` / `margin_bucket` fields are rejected
— see the signal inventory: those facets are the most outcome-inflated.

### BuildPriorVector

The normal pregame representation: a probability distribution over retained
`BuildProfile` rows for one `(champion_id, teamposition)`.

Required fields:

- `champion_id`, `teamposition`
- `profile_ids`, `hgnn_build_ids`, `probabilities`, `support_counts`
- `retained_mass`, `pruned_mass` (over the pre-pruning empirical distribution)
- `fallback_source` (`champion_role` | `role` | `global`)
- `smoothing_strength`, `catalog_version`

No normal production mass is assigned to an `"unknown"` build, and nothing
maps to the untrained reserve embedding row. If a champion-role has no
retained support, fall back to the role-level prior over labels, then global;
fallback profiles still resolve to real vocab ids. An unmappable build is a
hard failure or an explicitly-labelled emergency diagnostic, never a modelled
option.

Support and smoothing policy (initial defaults, tunable on train only):

- `profile_min_count=20`, `profile_min_share=0.01`
- `rl_core_min_count=50`, `rl_core_min_share=0.02`
- `tau=20` empirical-Bayes smoothing toward the fallback distribution:

```text
p_i = (n_i + tau * q_i) / (sum_j n_j + tau)
```

where `q_i` is the role-level (else global) label distribution restricted and
renormalised to the retained profiles. Smoothing happens after pruning;
`retained_mass`/`pruned_mass` are reported from raw counts so pruning is
visible.

### Source Labels

Every cache metadata block, eval payload, experiment artifact, and RL/search
output that involves a build assignment carries one of:

- `pregame_marginal_build`: accepted passive prior over train-supported
  profiles.
- `rl_candidate`: concrete profile chosen by RL/search from the train catalog.
- `train_observed_build`: observed label on a *train* row — valid for training
  and train-side calibration fitting only.
- `oracle_observed_build`: observed held-out (test) label — diagnostics only.

Accepted test/serving modes reject `oracle_observed_build` mechanically (a
validation function in the contract module, called by the eval harness and
predictor), not by convention.

## Passive Prediction

### Algorithm

For one draft (10 `(champion, role)` slots):

1. Build one `BuildPriorVector` per slot, truncated to its top `K_slot`
   profiles (default 3; the 11-label vocab makes per-slot distributions
   concentrated).
2. Enumerate joint assignments in descending product mass with a best-first
   heap over the 10 independent per-slot distributions (exact lazy top-W
   enumeration — no need to materialise `K_slot^10` candidates). Default world
   cap `W=512`; stop early once cumulative retained joint mass ≥ 0.90 (rare).
   Measured against real priors: mean retained joint mass is ~0.09 at `W=1`,
   ~0.66 at `K=3, W=128`, ~0.75 at `K=3, W=512`, plateauing near the per-slot
   truncation ceiling (~0.78 for `K=3`) — a ≥0.95 target is unreachable at
   any practical `W`, so completeness is reported, not chased.
3. Score all retained worlds in **one batched forward pass**. Only
   build-dependent inputs vary per world; precompute per `(slot, candidate)`
   once and assemble world tensors by indexing:
   - `build_id` (vocab index),
   - the 1vX prior pair `(win_rate, p1_cnt)` for the hypothesised key,
     smoothed with the standard runtime smoothing (no LOO at runtime),
   - sidecar blocks and semantic-context rows for `(champ, role, label)`.
   Champion ids, loadout/patch features (cache-side eval), and everything
   role-derived are shared across worlds.
4. Average sigmoid probabilities with the *unnormalised* retained joint
   weights divided by retained mass, and report `retained_joint_mass` and
   tail mass rather than hiding pruned probability behind silent
   renormalisation.
5. If retained joint mass is below a floor (default 0.35 — calibrated so the
   p10 draft at `K=3, W=512` passes; 0.5 would flag roughly a third of
   normal drafts), emit a low-confidence diagnostic on the result payload.
6. Calibrate the marginal probability with a fresh affine logit calibration
   fit on the **train split scored by the same marginalisation procedure**
   (source label `pregame_marginal_build`). Do not reuse the production
   ensemble's scale/bias blindly — it was fitted under observed-build
   conditioning; refitting on marginal train logits is cheap and removes a
   known mismatch.

### Where accepted metrics are computed

Through a cache-side harness (new module under `app/ml/`), not through
`WinRatePredictor`: it loads the v32 cache, reconstructs roles from slot
order, replaces test-side `build_id`/prior columns with hypothesised
candidates, and reuses `build_hgnn_inputs` plus the on-device sidecar gather
from `train.py`. This keeps loadout/patch heads served exactly as trained and
avoids the runtime fail-fast.

Cost envelope: 329,586 test games × 512 worlds ≈ 169M forward rows — on the
order of an hour of forward-only GPU time at current throughput (`W=128` ≈
42M rows stays the cheap iteration setting). Expose `--worlds` and
`--mass-floor`; report the retained-mass distribution alongside metrics.

### Baseline definition

The "safe no-build baseline" for promotion is **not** the dead unknown-build
arm (untrained embedding, empty priors). Use three reference points:

1. The marginalisation itself with `W=1` (modal build per slot) — isolates the
   value of spreading mass over alternatives (the modal world carries only
   ~9% joint mass, so a real difference is expected).
2. A **retrained no-build ensemble** (builds collapsed to a single label,
   3 seeds, same recipe) — the information-equivalent comparator, since the
   prior is deterministic given (champ, role). If marginalisation cannot beat
   it, serve the no-build model for passive prediction and keep the
   build-conditioned model solely as the RL/search counterfactual engine.
3. The recorded production observed-build test metrics
   (0.58260 / 0.67105) — an *oracle-conditioned* reference, expected to be
   better; the gap is the price of removing the leak, reported but not gated.

## Experiment Plan

### Oracle Ceiling (diagnostics only)

How much predictive power exists if the model knew the true completed-game
build shape. Phase A variants (runnable against the existing checkpoint):

1. observed primary label (this is exactly the current cache path — the
   recorded production test metrics already are this number)
2. observed primary with the marginal calibration applied (isolates
   calibration from information)

The aggregate-level Phase B-deciding numbers are already measured (signal
inventory: secondary +6.99pp², shape +7.21pp², 15-min label 3.06pp² vs final
5.36pp² on identical games). The remaining Phase B-deciding work is
model-level: a train-side check that conditioning on the 15-min time-capped
label retains its separation through the HGNN, not just in aggregates.
Report all oracle results separately, labelled `oracle_observed_build`.

### Pregame-Safe Ablations (accepted)

1. modal-build baseline (`W=1`)
2. retrained no-build ensemble (baseline 2 above)
3. primary-prior marginalisation, default pruning/smoothing
4. support-gated variant (stricter `profile_min_*`)
5. calibrated marginalisation (step 6 above)
6. sensitivity sweep over `K_slot ∈ {2,3,5}`, `W ∈ {128,512,2048}` on
   train-only scoring before touching test (measured retained-mass means:
   ~0.66 / ~0.75 / ~0.78 at `K=3`; `K=2` plateaus at ~0.5)

Promotion priority: NLL first, then accuracy, Brier, corrected reliability/ECE,
and per-patch stratified transfer. Every accepted result reports support tier
distribution, fallback-source counts, retained mass, and the source label.

## RL/Search Surface

The outer RL action surface remains champion drafting. Build control is a
legal inner planning surface over the same train-supported catalog.

What already works and must be preserved: `resolve_rewards` builds the full
config win matrix and aggregates with probability weights in
`expected_value`/`risk_adjusted` modes, and `worst_case` takes min/max over
joint role+build configs.

Changes:

- **Pool generation moves onto the catalog.** `build_pool_from_priors`
  currently uses raw matchup counts gated by `min_matchups`. Replace its
  weight source with smoothed `BuildPriorVector` probabilities and the
  `rl_core_min_*` gates, and stamp the pool file with `catalog_version`. Keep
  the pool file format (role, build_id, weight) so `make_pool_sampler` and the
  reward path are untouched structurally.
- **Stop hiding pruned mass.** The sampler's top-K renormalisation should also
  surface retained mass on `OptimizationResult` (or its configs) so reward
  diagnostics can flag low-coverage terminals.
- **Support penalties.** Rare profiles surviving the gates still need either
  masking from `worst_case`/argmax-style selection or a support-scaled
  penalty, so search cannot exploit noisy high-win-rate tails.
- Uniform averaging over candidate worlds is allowed only as an explicitly
  labelled ablation.

Supported modes stay `expected_value`, `risk_adjusted`, `worst_case` with
their current semantics.

## Sub-Agent Workstreams

Use isolated agents to keep context small and avoid cross-file churn. Each
agent gets the shared contract, its owned files, the relevant tests, and this
document. Agents report files changed, public interfaces changed, tests run,
model-metric risk, leakage risk, and runtime risk.

| Agent | Primary responsibility | Acceptance gate |
| --- | --- | --- |
| Orchestrator | Context packets, disjoint ownership, integration. | No overlapping edits without an explicit handoff. |
| Build Contract Agent | `BuildProfile`, `BuildPriorVector`, source labels, serialization, validation. | Canonical ids stable; JSON round trip; `oracle_observed_build` rejected by the accepted-mode validator; vocab identity asserted against a checkpoint config. |
| Train-Only Catalog Agent | Catalog + smoothed priors from `synergy_1vx` train rows. | Synthetic split tests prove non-train rows never contribute; fallback chain covered; no new CH tables in Phase A; catalog stats match the signal-inventory concentration numbers. |
| Dataset And Cache Safety Agent | Split-scoped guards; delete the dead unknown-build arm and id-0 default. | Accepted eval cannot read test-side `build_id`; train path byte-identical (no cache rebuild); removal verified by tests, not comments. |
| Passive Predictor Agent | Batched marginalisation: heap enumeration, per-slot precompute, single forward, marginal calibration. | Probability-space averaging; retained mass reported; `W=1` reduces exactly to modal scoring; runtime predictor path and cache-side harness share the assembly code. |
| Oracle Ablation Agent | Oracle matrix. | Outputs labelled `oracle_observed_build`, kept out of accepted reports. |
| Safe Ablation And Calibration Agent | Accepted ablations, sensitivity sweep, calibration fit. | NLL/acc/Brier/corrected ECE plus support/fallback/coverage in every report; sweep run train-only before test. |
| RL/Search Agent | Catalog-backed pool, retained-mass surfacing, support penalties. | Pool stamped with catalog version; reward weighting preserved; uniform averaging only as labelled ablation. |
| Performance Engineering Agent | Batching, lookup caching, enumeration efficiency. | Test-split marginal eval within the stated cost envelope; RL terminal scoring latency measured and reported. |
| Implementation Evaluator Agent | Independent review. | Blocks leakage, metric regressions, bad calibration, slow search, or avoidable complexity. |

The Implementation Evaluator Agent is separate from implementers; its job is
to protect model performance and code performance, not to add features.

## Implementation Order (Phase A)

1. Land contracts, source labels, and validation (incl. vocab assertion).
2. Catalog + smoothed priors from `synergy_1vx`.
3. Split-scoped guards; delete the dead unknown-build runtime arm and the
   id-0 default.
4. Cache-side marginalised eval harness; batched marginal path in the
   predictor sharing its assembly code.
5. Marginal calibration fit on train; safe ablations (incl. the retrained
   no-build ensemble); oracle references.
6. RL pool regeneration from the catalog; retained-mass surfacing; support
   penalties.
7. Performance pass (enumeration, per-slot precompute reuse, lookup caching).
8. Evaluator audit; record results in `EXPERIMENTS.md`; only then decide
   Phase B.

## Test And Metric Gates

Minimum local tests:

```bash
uv run pytest tests/ml/test_build_dataset.py tests/ml/test_dataset.py tests/ml/test_predictor.py
uv run pytest tests/ml/test_train_defaults.py tests/ml/test_train_calibration.py tests/ml/test_encoder_sidecar.py
uv run pytest tests/rl/test_reward.py
```

Final verification:

```bash
uv run pytest tests/core tests/ml tests/classification tests/rl
uv run ruff check .
uv run pyright
```

Metric promotion gates:

- Oracle metrics are reported only as ceiling estimates.
- Accepted marginalisation improves or preserves test NLL versus the modal
  (`W=1`) baseline, without reliability collapse; the gap to the
  oracle-conditioned production reference is reported, not gated.
- Serving decision: if marginalisation does not beat the retrained no-build
  ensemble on test NLL, the no-build model becomes the passive serving path
  and the conditional model is RL/search-only. Either outcome is a valid
  Phase A result.
- Brier and corrected ECE do not materially regress.
- Retained mass, support tier, fallback source, and build source are emitted
  in every accepted report.
- RL/search reward aggregation stays weighted by profile probabilities, with
  retained mass surfaced.
- Training path is regression-free: train-side cache arrays untouched in
  Phase A, and `verify_equivalence.py` passes after any model-code refactor.

## Non-Goals

- Do not treat observed held-out final build labels as accepted inputs.
- Do not remove observed build labels from the *train* path — the conditional
  model requires them.
- Do not create a normal production `"unknown build"` arm or route anything
  to the untrained reserve embedding row.
- Do not use latent-vector or logit averaging as a substitute for probability
  marginalisation.
- Do not allow RL/search to choose build shapes outside the train catalog.
- Do not promote an accuracy gain that comes with material NLL or calibration
  regression.
- Do not enrich the *final*-inventory label (secondary set, margin, shape) —
  measured as the most outcome-inflated facets; Phase B enrichment is
  time-capped purchase labels only.
- Do not add deterministic per-(champ, role) build features (averaged
  historical labels, group shares, mean shape vectors): they carry zero
  draft-time information beyond the champion identity the model already has.
- Do not start Phase B (label redefinition, cache v33, retrain) before the
  Phase A accepted metrics and the model-level time-capped check are
  recorded in `EXPERIMENTS.md`.
