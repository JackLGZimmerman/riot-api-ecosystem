# HGNN Asset Separation: Build-Conditional vs Pregame-Native

Implementation-ready design for separating the HGNN win model's assets into two
explicit families with disjoint key spaces and disjoint leakage rules. Lean by
intent: it names contracts, ownership, and guards precisely so a small executor
can land each slice without re-deriving the system.

## 0. Framing (read first)

There is **one trained checkpoint today**: a build-conditional HGNN scoring
`P(win | draft, concrete build world)`. The "pregame-native" prediction
`P(win | draft)` is currently *emergent*, produced two ways:

- **Marginalisation** — average the build-conditional model's *output
  probabilities* over `P(build | champ, role)` from the train-only catalog
  (`predictor.predict_marginal`, `marginal_eval` `marginal`/`modal` modes).
- **Modal comparator** — retrain with every build input replaced by the modal
  build per `(champion, role)` (`TrainConfig.pregame_modal_builds`).

This document does **not** ask for a second trained model. It asks for an
explicit, mechanically-enforced **asset boundary** so that:

- build-conditional assets may legitimately read observed train builds, and
- pregame-native (accepted test/serving) assets are *provably* keyed only on
  draft-available inputs and never touch a held-out observed `build_id` or
  final inventory.

The boundary is the deliverable. Splitting the checkpoint itself is future work
and is explicitly a non-goal here.

## 1. Key-space taxonomy

Two families, two key spaces. Every ML asset belongs to exactly one.

| Family | Key space | Reads observed build? | Reaches accepted scoring? |
| --- | --- | --- | --- |
| `build_conditional` | `(championid, teamposition, build)` | Yes (train rows) | Only as `oracle`/`train_observed` diagnostics |
| `pregame_native` | `(championid, teamposition)` (later `+ draft-safe loadout`) | **Never** | Yes (`pregame_marginal_build`, `rl_candidate`) |

"draft-safe loadout" = runes / summoner spells, chosen before the game starts
(already admissible: see the keystone-conditioned prior). Final item inventory
is **not** draft-safe and is build-conditional only.

## 2. Current shared asset surfaces (built / loaded / served)

Everything below is keyed `(champion, role, build)` today and is therefore
**build-conditional**, even where it feeds an "accepted" path via
marginalisation. The marginal path is leakage-free only because it never reads a
*held-out* build; it enumerates *train-catalog* builds.

| Surface | Module | Built | Loaded | Served |
| --- | --- | --- | --- | --- |
| Solo 1vX prior win_rate | `app/ml/priors.py` `PriorTables.p1: dict[(int,str,str)]` | `build_dataset.py` `_CHUNK_QUERY_TEMPLATE` from `synergy_1vx` at observed build (`split='train'`) | `load_priors()` (`WHERE split='train'`) | `predictor._batch_inputs` → `win_rate` model input |
| `win_rate` / `p1_cnt` cache arrays | `app/ml/cache_layout.py` ARRAY_FILES | `build_dataset.py` per game/slot | `dataset.load_splits()` mmap | train + eval forward |
| `build_id` cache array | `cache_layout.py` ARRAY_FILES | `build_dataset.py` (observed build per slot) | `dataset.load_splits()` | build identity embedding; `apply_modal_build_split` replaces it pregame |
| Identity-encoder sidecars npz | `app/ml/encoder_sidecar.py`, `semantic_identity_sidecar_compact.npz` | offline encoder pipeline, keyed `(championid, teamposition, build)` | `EncoderSidecarLookup.load`; `train._build_sidecar_gatherer` (per-batch gather) | `_SidecarGatherer` → model sidecar inputs |
| Semantic group features | `app/ml/semantic_group_features.py` (`SEMANTIC_GROUP_FEATURE_DIM=25`) | `materialize_semantic_group_feature_cache` from `identity_context_raw` + `build_id` | `dataset.load_splits()` (optional materialise) | model semantic head |
| Semantic context raw lookup | `app/ml/semantic_context_lookup.py` `(int,str,str)` | `load_semantic_context_raw_lookup()` from classification embeddings (`split='train'`) | `predictor` runtime | `build_semantic_group_features` |
| Build identity embedding | `app/ml/hgnn_model.py` (`n_builds`) | trained weights | checkpoint | forward |
| Loadout / patch heads | `cache_layout.py` loadout_features/patch_features | `build_dataset.py` | `dataset.load_splits()` | offline only; production serving dims = 0 |
| **Build catalog** `P(build\|champ,role)` | `app/ml/build_catalog.py` `BuildCatalog.vectors: dict[(int,str)]` | `build_catalog()` from `priors.p1` (train-only) | `marginal_eval` line ~410; `predict_marginal` arg | marginal world weights |
| Modal transform | `app/ml/pregame.py` `apply_modal_build_split` | n/a (deterministic fn of `(champ, slot role)`) | `train()` `pregame_modal_builds`, `marginal_eval` `modal` | replaces build inputs |
| Keystone-conditioned prior | `build_catalog.conditioned_prior_vector`, `app/ml/loadout_condition.py` | train-only keystone counts | `marginal_eval --condition keystone` | reweights catalog vector |

The **catalog**, **modal transform**, and **keystone conditioning** are already
pregame-native by construction (champ-role keyed, train-only, never read a
held-out build). Everything else is build-conditional.

## 3. Target contracts

### 3.1 Build-conditional HGNN — `P(win | draft, build world)`

- **Input key space:** `(championid, teamposition, build)`.
- **Training:** may read observed train builds (`build_id` cache array, the
  `synergy_1vx` prior at the observed build, sidecars keyed at build). This is
  correct and stays.
- **Scoring a held-out row at its own observed build** is the `oracle`
  diagnostic and is rejected for accepted scoring by
  `validate_accepted_build_source` (`oracle_observed_build`,
  `train_observed_build`).
- **Identity:** this is the current production checkpoint
  (`hgnn_production_model.pt`).

### 3.2 Pregame-native HGNN — `P(win | draft)` / `P(win | draft, draft-safe loadout)`

- **Input key space:** `(championid, teamposition)` initially; may extend to
  `(championid, teamposition, draft-safe-loadout)` (runes/summoners) but never
  to final inventory.
- **Two realisations (both kept):**
  - Marginalisation over the catalog — the current accepted path.
  - Modal comparator (`pregame_modal_builds`) — the `W=1` floor.
- **Hard rule:** no accepted pregame-native asset may depend on a held-out
  observed `build_id` or final inventory. The catalog, modal transform, and
  keystone conditioning satisfy this today; the separation makes it *checked*.
- **Marginalisation invariant:** average **output probabilities**, never logits
  or embeddings (`predict_marginal`: `np.dot(probabilities, weights) /
  weights.sum()`).

## 4. Asset ownership matrix

| Asset class | Build-conditional owner | Pregame-native owner |
| --- | --- | --- |
| Sidecars | `encoder_sidecar.py`, `*_sidecar_compact.npz` `(c,r,b)` | none (excluded; sidecars are build-keyed) |
| Semantic features | `semantic_group_features.py` (uses `build_id`) | none yet (would need `(c,r)`-only descriptor) |
| Priors | `priors.py` `p1 (c,r,b)` at observed build | catalog-derived `P(build\|c,r)` only |
| Identity embeddings | build embedding (`n_builds`) | champion + role embeddings only |
| Cache fields | `build_id`, loadout/patch, sidecar arrays | `champion_id`, role (slot index), `blue_win`, patch (draft-safe subset) |
| Config flags | `interaction_loo`, `encoder_sidecar_path` | `pregame_modal_builds`, catalog gates (`CatalogGates`, `ConditionGates`) |
| Artifact paths | `hgnn_production_model.pt`, sidecar npz | `build_catalog` (in-memory, train-derived; versioned by `catalog_version`) |
| Build-source labels | `train_observed_build`, `oracle_observed_build` | `pregame_marginal_build`, `rl_candidate` |

## 5. Leakage rules (mechanical, not by convention)

1. **Training** (build-conditional) may read observed train builds. No change.
2. **Accepted test/serving** assets must be pregame-native: champ-role keyed,
   derived from `split='train'` only, never reading a held-out row's observed
   `build_id` or final inventory.
3. **Observed-build scoring** (`oracle`, `train_observed`) stays diagnostic and
   is rejected by `validate_accepted_build_source` on every accepted payload.
4. **The catalog is the asset boundary.** A pregame-native asset's key space
   must be assertable. Add `assert_pregame_native_key_space()` and call it at
   every accepted catalog entry point so a build-keyed asset cannot silently
   reach accepted scoring.
5. **Draft-safe loadout** extensions (keystone, summoners, styles) are
   admissible *only* because they are selected pregame and aggregated train-only
   (`conditioned_prior_vector`, `loadout_condition.py`). Final inventory never
   qualifies.

## 6. Minimal migration plan (preserves the build-marginal path)

The build-marginal path stays byte-for-byte equivalent. The separation is
*additive enforcement*, not behaviour change.

- **Step 1 (this PR — Phase 2 slice):** make the pregame-native contract
  explicit and enforced in `build_catalog.py` (taxonomy constants + key-space
  guard + `BuildCatalog.assert_pregame_native()`), and call the guard at the two
  accepted catalog entry points (`predict_marginal`, `marginal_eval`). No model,
  cache, or metric change. `verify_equivalence.py` must stay green.
- **Step 2 (later):** annotate the build-conditional surfaces with the
  `build_conditional` family tag where an executor touches them, so the two
  families are self-documenting in code.
- **Step 3 (later, gated):** if a genuinely pregame-native *trained* model is
  wanted, it is the `pregame_modal_builds` comparator promoted to its own
  artifact path — a separate, explicitly gated decision (label/cache work).

No step rewrites the marginal math, the catalog construction, or the cache
format.

## 7. File-by-file implementation plan (Phase 2 slice)

1. `app/ml/build_catalog.py`
   - Add family constants: `ASSET_FAMILY_BUILD_CONDITIONAL = "build_conditional"`,
     `ASSET_FAMILY_PREGAME_NATIVE = "pregame_native"`.
   - Add key-space constants: `KEY_SPACE_CHAMP_ROLE = ("championid",
     "teamposition")`, `KEY_SPACE_CHAMP_ROLE_BUILD = ("championid",
     "teamposition", "build")`.
   - Add `assert_pregame_native_key_space(keys)` — raises `ValueError` if any
     key is not a `(int championid, str teamposition)` 2-tuple (i.e. a
     build-keyed asset has leaked in).
   - Tag `BuildCatalog` with `ClassVar` `asset_family` / `key_space` (no new
     dataclass field) and add `BuildCatalog.assert_pregame_native()` calling the
     guard over `self.vectors.keys()`.
   - Export the new names in `__all__`.
2. `app/ml/predictor.py` `predict_marginal`
   - After `catalog.validate_model_vocab(...)` (line ~341), call
     `catalog.assert_pregame_native()`.
3. `app/ml/marginal_eval.py`
   - After `catalog = build_catalog(...)` (line ~410), call
     `catalog.assert_pregame_native()`.
4. `tests/ml/test_asset_separation.py` (new)
   - Mirror `tests/ml/test_build_catalog.py` conventions.
   - Assert a real catalog passes `assert_pregame_native()`.
   - Assert `assert_pregame_native_key_space` rejects a 3-tuple
     `(champ, role, build)` key with a clear message.
   - Assert the family/key-space constants are wired
     (`BuildCatalog.asset_family == ASSET_FAMILY_PREGAME_NATIVE`,
     `key_space == KEY_SPACE_CHAMP_ROLE`).

## 8. Sub-agent work packets

### Packet A — catalog taxonomy + guard (`build_catalog.py`)

- **Goal:** add the family/key-space taxonomy and pregame-native key-space
  guard; tag `BuildCatalog`; export names.
- **Owned files:** `app/ml/build_catalog.py` only.
- **Non-goals:** no change to catalog construction, smoothing, enumeration, or
  conditioning math; no new dataclass *field* (use `ClassVar`); no edits to
  callers.
- **Interfaces:** constants + `assert_pregame_native_key_space(keys)` +
  `BuildCatalog.assert_pregame_native()`.
- **Acceptance:** existing `tests/ml/test_build_catalog.py` unchanged and green;
  `assert_pregame_native()` is a no-op on a normally-built catalog.
- **Risks/escalation:** if `vectors` ever legitimately needed a 3-tuple key,
  escalate — that would mean the catalog stopped being pregame-native.

### Packet B — wire guard into accepted paths (`predictor.py`, `marginal_eval.py`)

- **Goal:** call `catalog.assert_pregame_native()` at the two accepted entry
  points.
- **Owned files:** `app/ml/predictor.py`, `app/ml/marginal_eval.py`.
- **Non-goals:** no change to scoring, enumeration, calibration, payloads, or
  the `oracle` branch.
- **Acceptance:** `predict_marginal` and `marginal_eval` accepted modes behave
  identically; the guard runs before any forward pass.
- **Depends on:** Packet A.

### Packet C — tests (`tests/ml/test_asset_separation.py`)

- **Goal:** lock the contract.
- **Owned files:** `tests/ml/test_asset_separation.py` (new) only.
- **Non-goals:** no GPU, no cache load, no model load — pure catalog-level unit
  tests (CPU, sub-second).
- **Acceptance:** new tests pass; full `tests/ml/test_build_catalog.py` still
  passes.

## 9. Test plan

- **Cache generation:** unchanged; no cache field added. Verify
  `cache_layout.py` ARRAY_FILES untouched.
- **Model input assembly:** `predict_marginal` / `_batch_inputs` produce
  identical tensors (guard is pre-forward, side-effect-free).
- **Train path:** unaffected — guard is not on the training path; the build-
  conditional cache and `pregame_modal_builds` are unchanged.
- **Eval path:** `marginal_eval` `marginal`/`modal`/`oracle` modes return
  identical metrics; the guard only runs on accepted (non-oracle) modes.
- **Predictor/runtime:** `predict_marginal` returns identical
  `MarginalPrediction`.
- **Leakage guards (new, the point of the slice):**
  - `assert_pregame_native()` passes on a real catalog.
  - `assert_pregame_native_key_space` rejects a build-keyed (3-tuple) asset.
  - `validate_accepted_build_source` still rejects `oracle`/`train_observed`
    (regression).
- **No-regression gate:** `verify_equivalence.py` stays green (production
  artifact contract unchanged).

Run order (CPU-only unit tests, GPU not required):

```
python -m pytest tests/ml/test_build_catalog.py tests/ml/test_asset_separation.py -q
```

Check `nvidia-smi` / `free -h` before any run that imports `app.ml` while
training is active (serialise GPU/RAM-heavy work per project ops rules).
