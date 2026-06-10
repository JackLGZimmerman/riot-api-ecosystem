# HGNN Build Intent Plan

Last updated: 2026-06-10.

This document is the implementation plan for adding build intent to the HGNN and
RL/search stack without leaking completed-game item information. It reconciles
the current model surface, the central-band build-profile diagnostics, and the
sub-agent plan for isolated implementation.

## Goal

Maximise the useful effect of build information while keeping every accepted
training, validation, test, serving, and RL/search path constrained to
draft-phase information.

The production signal is not an observed held-out build label and not a single
synthetic "unknown" build. The only normal pregame build signal is a train-only
historical distribution:

```text
P(build_profile | champion_id, teamposition)
```

The HGNN should score concrete train-supported build profiles, then average the
resulting probabilities:

```text
P(win | draft) =
  sum_b P_HGNN(win | draft, concrete_build_profile=b)
        * P(build_profile=b | champion_id, teamposition)
```

Do not average build ids, build embeddings, sidecar tensors, semantic features,
logits, or hidden states as a shortcut. The HGNN is nonlinear, and the current
sidecar and semantic lookup surfaces are keyed by concrete `(champion, role,
build)` identities. Production marginalisation must happen over output
probabilities.

## Current Risks

The existing code still contains oracle-friendly paths that are useful for
diagnostics but dangerous as defaults:

- `DatasetConfig.use_final_build_labels` currently allows final build labels in
  cache construction.
- The build pivot SQL joins `participant_item_value_totals` and emits
  `highest_value_label` as the build label.
- Runtime prediction can default a missing build id to `0`, which may be a real
  build label rather than a safe fallback.
- The RL reward path computes candidate probabilities, but the expected reward
  currently averages the win matrix uniformly.
- Existing final-build residual experiments are oracle diagnostics because they
  read held-out completed-game build labels, secondary labels, or margins.

The implementation must make accepted paths hard to run with those oracle
signals by accident.

## Core Data Contracts

### BuildProfile

`BuildProfile` is a train-supported build atom. It describes a concrete build
shape that can be scored by the HGNN.

Required fields:

- `profile_id`
- `champion_id`
- `teamposition`
- `primary_label`
- `secondary_label_set`
- `shape_bucket`
- optional `margin_bucket`
- `hgnn_build_id`
- `support_count`
- `support_share`
- `support_tier`
- `catalog_version`

Secondary labels only make sense relative to a primary profile. There should be
no standalone "observed secondary only" feature or ablation.

### BuildPriorVector

`BuildPriorVector` is the normal pregame representation. It is a probability
distribution over train-supported `BuildProfile` rows for one
`(champion_id, teamposition)`.

Required fields:

- `champion_id`
- `teamposition`
- `profile_ids`
- `hgnn_build_ids`
- `probabilities`
- `purchase_counts`
- `retained_mass`
- `pruned_mass`
- `fallback_source`
- `smoothing_strength`
- `catalog_version`

No normal production mass should be assigned to an `"unknown"` build. If a
champion-role catalog has no support, fall back to role-level then global
historical profile priors. Treat an unmappable build only as a hard failure or
emergency diagnostics case, not a modelled option.

Default support and smoothing policy:

- `profile_min_count=20`
- `profile_min_share=0.01`
- `rl_core_min_count=50`
- `rl_core_min_share=0.02`
- `tau=20` for empirical-Bayes smoothing
- start RL/search with top `K=3-5` legal profiles per champion-role

The smoothed prior should use the form:

```text
p_i = (n_i + tau * q_i) / (sum_j n_j + tau)
```

where `q_i` is a role/global fallback prior over retained profiles.

### Source Labels

Use explicit source labels in cache metadata, runtime diagnostics, experiment
payloads, and RL/search output:

- `pregame_marginal_build`: accepted passive prior over train-supported profiles.
- `rl_candidate`: accepted build profile chosen by RL/search from the train
  catalog.
- `oracle_observed_build`: unsafe observed held-out final build/profile signal.

Accepted validation/test/runtime modes must reject `oracle_observed_build`.

## Experiment Plan

### Oracle Ceiling

Oracle experiments answer how much predictive power exists if the model knew the
true completed-game build shape. These are not production-valid.

Run these variants:

1. observed primary
2. observed primary plus secondary label set
3. observed primary plus margin
4. observed full profile
5. observed full profile plus margin

Report them separately from accepted metrics. Their role is to decide whether
primary direction is enough, whether secondary/full profile shape matters, and
whether profile confidence or margin adds useful signal.

### Pregame-Safe Ablations

Accepted experiments must not read the held-out game's final build label.

Run these variants:

1. no-build safe baseline
2. train historical primary-prior marginalisation
3. train historical full-profile marginalisation
4. support-gated full-profile marginalisation
5. calibrated support-gated marginalisation

Promotion should prioritize NLL, then accuracy, Brier, corrected reliability/ECE,
and stratified transfer to test. Report support tier, fallback source, retained
mass, and uncertainty band for every accepted result.

## Passive Prediction

For one draft:

1. Build one `BuildPriorVector` per drafted champion-role slot.
2. Retain top profile assignments by product mass with a deterministic top-K or
   beam search.
3. Evaluate every retained concrete world through the normal HGNN input path.
4. Average sigmoid probabilities with the original retained weights.
5. Report retained mass and tail mass instead of hiding pruned probability by
   blind renormalisation.
6. Calibrate the final marginal probability on validation using the same
   marginalisation procedure.

If retained joint mass is too low, surface a low-confidence diagnostic rather
than pretending the top-K world set is complete.

## RL/Search Surface

The outer RL action surface should remain champion drafting. Build control is a
legal inner planning surface over the same train-supported build catalog.

At terminal scoring or late search:

```text
outer search: choose champions and bans
inner search: choose role assignment and one legal build profile per champion
opponent: marginalise, risk-adjust, or solve worst-case over legal profiles
HGNN: score each concrete profile world
reward: aggregate with profile probabilities and support penalties
```

Expected reward must use build-profile probabilities. It must not use uniform
averaging over candidate worlds unless that is an explicitly labelled ablation.
Rare profiles need support penalties or masking so RL cannot exploit noisy high
win-rate tails.

Supported modes:

- `expected_value`: own plan optimised against opponent historical priors.
- `risk_adjusted`: expected value penalised by profile uncertainty or variance.
- `worst_case`: maximin against opponent legal best response.

## Sub-Agent Workstreams

Use isolated agents to keep context small and avoid cross-file churn. Each agent
gets only the shared contract, its owned files, the relevant tests, and this
document. Agents must report files changed, public interfaces changed, tests
run, model-metric risk, leakage risk, and runtime risk.

| Agent | Primary responsibility | Acceptance gate |
| --- | --- | --- |
| Orchestrator | Create context packets, assign disjoint ownership, integrate reports. | No overlapping edits without an explicit handoff. |
| Build Contract Agent | Implement `BuildProfile`, `BuildPriorVector`, source labels, serialization, validation. | Stable canonical ids, JSON round trip, oracle source rejected in accepted mode. |
| Train-Only Catalog Agent | Build train-split-only profile catalog and priors. | Synthetic split tests prove val/test rows never contribute. |
| Dataset And Cache Safety Agent | Wire safe config/cache surfaces and rebuild errors. | Accepted cache/eval fails when final build labels are enabled. |
| Passive Predictor Agent | Implement batched concrete-profile marginalisation. | Probability-space averaging, no id-0 default, retained mass reported. |
| Oracle Ablation Agent | Run the unsafe diagnostic matrix. | Outputs labelled oracle and kept separate from accepted metrics. |
| Safe Ablation And Calibration Agent | Run accepted ablations and calibration. | NLL, accuracy, Brier, corrected ECE, support/fallback/coverage reported. |
| RL/Search Agent | Add legal build-profile planning to RL/search. | Reward uses profile weights, not uniform matrix means. |
| Performance Engineering Agent | Optimise batching, caching, top-K/beam pruning. | Inference remains feasible for RL/search with latency/memory diagnostics. |
| Implementation Evaluator Agent | Independently review the implementation. | Blocks leakage, metric regressions, bad calibration, slow search, or weak code. |

The Implementation Evaluator Agent should be separate from implementers. Its
primary job is to protect model performance and code performance, not to add
features.

## Implementation Order

1. Land shared contracts and source validation.
2. Build the train-only profile catalog and smoothed priors.
3. Wire dataset/cache metadata and unsafe-mode guards.
4. Add passive batched marginalisation in the predictor.
5. Add accepted safe ablations and oracle-only diagnostics.
6. Add RL/search build-profile planning and weighted reward aggregation.
7. Optimise batching and lookup caching.
8. Run evaluator audit before promotion.

## Test And Metric Gates

Minimum local tests:

```bash
uv run pytest tests/ml/test_build_profiles.py
uv run pytest tests/ml/test_dataset.py tests/ml/test_predictor.py
uv run pytest tests/rl/test_pool.py tests/rl/test_reward.py tests/rl/test_env.py
```

Final verification:

```bash
uv run pytest tests/core tests/ml tests/classification tests/rl
uv run ruff check .
uv run pyright
```

Metric promotion gates:

- Oracle metrics are reported only as ceiling estimates.
- Accepted build marginalisation improves or preserves validation NLL versus the
  safe no-build baseline.
- Accuracy gain transfers from validation to test without reliability collapse.
- Brier and corrected ECE do not materially regress.
- Retained mass, support tier, fallback source, and build source are emitted in
  every accepted report.
- RL/search reward aggregation is weighted by profile probabilities.

## Non-Goals

- Do not treat observed held-out final build labels as accepted inputs.
- Do not create a normal production `"unknown build"` arm.
- Do not use latent-vector or logit averaging as a substitute for probability
  marginalisation.
- Do not allow RL/search to choose arbitrary build shapes outside the train
  catalog.
- Do not promote an accuracy gain that comes with material NLL or calibration
  regression.
