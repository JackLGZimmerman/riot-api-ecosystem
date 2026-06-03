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

| # | Scope | Size | Model impact | Status |
| --- | --- | --- | --- | --- |
| 1 | Registry + evidence-loop unification + cache hash; 147 features byte-stable | S | none | done |
| 2 | Static champion branch (lookup join, standardize, no priors) | S | re-benchmark once | done (+47 feat) |
| 3 | Team-share / matchup / concentration features (participant-grain SQL) | L | re-benchmark per batch | done (+55 feat) |
| — | Temporal branch + its own autoencoder | XL | separate doc | done (see temporal doc) |

Full-game matrix widths: 147 default, +47 with `include_static_champion`, +55 with
`include_context_features` (249 with both). All branches are opt-in so the default
matrix stays byte-identical to Phase 0.

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

- `context_features.py`: team aggregate (sum over 5 teammates) + opponent via the
  pair-sum identity (`self - opponent = 2*self - pair_sum`, `HAVING count() = 2`),
  avoiding an 11M x 11M self-join. Composite quantities (`takedowns`,
  `durability_total`, `epic_kills`, `structure_takedowns`, ...) are expressed at
  participant grain.
- 21 team-share + 4 concentration + 30 matchup (11 raw + 4 share, each diff +
  advantage) = 55.
- Concentration: per-team Herfindahl HHI of the 4 share metrics (`gold`, `xp`,
  `total_farm`, `champion_damage`), `HHI = sum(x^2) / sum(x)^2` over the five
  teammates, broadcast to each identity. Emitted by `team_share_query` (mirrors
  the `concentration()` helper). High = one player carries that metric.
- `evidence_kind = MATCHUPS`, smoothed through the existing hierarchy
  (`apply_hierarchical_shrinkage`, extended when present).

## Materialised aggregation (current)

Aggregation and prior derivation run in ClickHouse, not Python. `build_tables.py`
materialises **sufficient-statistic** tables (raw `SUM` / `COUNT` /
`SUM_timeplayed`), one row per `(split, championid, teamposition, build)`:

- `classification_identity_base` — participant-stats sums + `matchups` +
  `sum_w_timeplayed` + `build_group`.
- `classification_final_base` — final-snapshot sums (denominator = `matchups`).
- `classification_context_base` — team-share / matchup sums + their counts.

Heavy scans use `shard -> stage (append) -> GROUP BY combine` to stay under the
server memory limit. `load.py` then issues thin SELECTs: the baseline divides the
sums into rate form, and **every prior level is an exact SQL `GROUP BY` rollup**
(`sum(sum_x) / sum(matchups)` etc.) — replacing the former in-Python
`derive_prior` and context shard-averaging. The shared smoother
(`smoothing.py`) and median/MAD standardisation stay in Python. A
`classification_base_meta` row carries `catalogue_hash()`; loaders raise if the
tables are missing or stale.

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
