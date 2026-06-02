# Metric Catalogue Plan

Declarative registry for `app/classification` full-game + static-champion metrics.
Temporal/timeline metrics are a separate workstream: see
[TEMPORAL_BRANCH_PLAN.md](TEMPORAL_BRANCH_PLAN.md).

## Goal

Replace the implicit catalogue (membership in `RATE_METRICS` / `PER_MINUTE_METRICS`
/ `FINAL_SNAPSHOT_AVG_METRICS` tuples + the `DERIVED_METRIC_FUNCS` dict in
`config.py`) with one declarative registry that:

1. Drives a single evidence-keyed prior-derivation loop, replacing the two
   hardcoded loops in `load.derive_prior` ([load.py:497-514]).
2. Carries a catalogue hash so adding/changing a metric invalidates stale `_raw`
   caches (the baseline `_raw` cache currently does not self-invalidate on
   metric-set change — [load.py:146-151]).

It is **not** sold as a line-count reduction: the 81 derived calculations still
exist as code; the registry adds metadata around them.

## MetricSpec (lean)

```python
@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: Source                 # how the value is produced (see below)
    evidence_kind: Evidence        # which weight column smoothing uses
    dependencies: tuple[str, ...]  # source columns this metric reads
    calculation: Callable | None   # None for raw source columns
```

Dropped vs the original proposal: `family` (descriptive only, nothing keys off
it), per-metric `version` (one catalogue-level hash replaces it), `prior_policy`
(folds into `evidence_kind` + the existing `isolated_roles` rule in
`EmbeddingConfig`).

A `branch: Branch` field joins the spec in Phase 2, when `STATIC_CHAMPION`
becomes a second branch; Phase 1 is full-game only, so a one-value enum is
deferred.

### Source

- `PARTICIPANT_STATS_RATE` — `sum(x)/count()` over `participant_stats`.
- `PARTICIPANT_STATS_PER_MINUTE` — `60*sum(x)/sum(timeplayed)`.
- `FINAL_SNAPSHOT` — final `tl_participant_stats` snapshot, matchups-averaged.
- `DERIVED` — computed from smoothed source columns at matrix-build time.
- `TEAM_SHARE` / `MATCHUP_DIFF` — participant-grain, then averaged to identity
  (Phase 3).
- `STATIC_CHAMPION` — per-champion lookup, no aggregation (Phase 2).

### Evidence

- `MATCHUPS` — rate-like sources; smoothed with `matchups` weight.
- `SUM_W_TIMEPLAYED` — per-minute sources; `sum_w_timeplayed` weight, with the
  existing reliability cap ([smoothing.py:738-744]).
- `STATIC_NONE` — derived + static metrics; not individually smoothed.

## Compatibility contract (hard requirement for Phase 1)

The registry becomes the source of truth; these stay **byte-identical**:

- `ALL_METRICS` order = `RATE_METRICS, LARGEST_AVG, FINAL_SNAPSHOT, PER_MINUTE`.
- `raw_and_derived_metric_names()` order = `(*ALL_METRICS, *derived)`.
- `raw_metric_names()`, `full_game_derived_metric_names()`, `RATE_LIKE_METRICS`,
  `PER_MINUTE_METRICS`, `FINAL_SNAPSHOT_AVG_METRICS` keep current values.
- Every `LevelMatrix.matrix` is bit-for-bit unchanged.

`config.py` keeps these names as thin views over the registry.

## Phases

| # | Scope | Size | Model impact |
| --- | --- | --- | --- |
| 1 | Registry + evidence-loop unification + cache hash; 147 features byte-stable | S | none |
| 2 | Static champion branch (lookup join, standardize, no priors) | S | re-benchmark once |
| 3 | Team-share / matchup features (participant-grain SQL + evaluator) | L | re-benchmark per batch |
| — | Temporal branch + its own autoencoder | XL | separate doc |

### Phase 1 — registry, zero behavior change

1. New `app/classification/embeddings/registry.py` with `MetricSpec`, `Branch`,
   `Source`, `Evidence`, and the FULL_GAME specs (66 raw + 81 derived).
2. `config.py` derives `ALL_METRICS` etc. from the registry; public API unchanged.
3. Collapse the two loops in `derive_prior` ([load.py:497-514]) into one loop
   keyed by `evidence_kind` (weight = `matchups` for `MATCHUPS`,
   `sum_w_timeplayed` for `SUM_W_TIMEPLAYED`). `smooth_hierarchical_baseline`
   stays untouched — it already takes metric lists as args.
4. `catalogue_hash()` (stable hash of registry names + sources + evidence) stored
   in the baseline `_raw` npz; `_load_level_rows` rejects a mismatching hash.
5. **Golden test**: capture current `LevelMatrix` per level offline from the
   populated `_raw` cache, refactor, assert arrays + feature_names identical.

### Phase 2 — static champion

- Load `champion_basic_stats_flat.jsonl`; columns except `_key`/`id`; add
  level-18 derived (`health_l18 = health_flat + 17*health_perLevel`).
- `evidence_kind = STATIC_NONE`; standardize only; join on `championid`.

### Phase 3 — team-share / matchup

- New participant-grain CTE: team aggregate (sum over 5 teammates) + opponent
  join (same `teamposition`, other team), then aggregate to identity.
- Participant-grain evaluator for composite dependencies (`epic_kills`,
  `structure_takedowns`, `objective_damage` are derived, not raw columns).
- `evidence_kind = MATCHUPS`, smoothed directly. Add in small batches, each
  gated on the autoencoder rank/recall benchmark.

## Test plan

- Registry: unique names; dependencies resolve; no `challenge` substring;
  byte-stable `ALL_METRICS` / `raw_and_derived_metric_names()` ordering.
- Evidence loop: each metric routed to the correct weight column; `derive_prior`
  output identical to the pre-refactor two-loop output.
- Cache hash: changing a spec invalidates the baseline `_raw` cache.
- Golden: `LevelMatrix` arrays identical (Phase 1 gate).
- Math (Phase 3): safe-divide, team share, matchup diff/advantage, concentration.
- Static (Phase 2): level-18 formulas.

## Open decisions

- Curated team-share / matchup lists — use the original plan's curated recipes,
  not cross-products. Confirm final list before Phase 3.
- Whether the static branch is concatenated into the full-game matrix or kept as
  a sidecar the autoencoder reads separately.
