# app/rl Redesign — Reduction + Adversarial League

Design-implementation spec for optimising `app/rl`. Orchestrated top-level
(Claude); isolated/parallel packets delegated to cheap executors. Every change
below is decision-complete: file, exact action, rationale, and acceptance gate.

## Goals (success criteria, in priority order)

1. **Speed** — fewer predictor forward passes; the terminal reward is the
   per-episode bottleneck.
2. **Lines of code** — collapse duplicated transition/net/worker/episode logic.
3. **Clarity** — one source of truth per concept.
4. **Cost** — fewer training iterations to a strong agent (SPRT-gated league).
5. **Files** — fewer modules, balanced with clarity.

Two capabilities must be preserved/added:
- **Reward signal** = HGNN outputs via `WinRatePredictor` (already wired through
  the `Predictor` protocol → `app/ml/hgnn_model.py` `final_logit` → sigmoid).
- **Adversarial league** — new agents keep facing a pool of strong, diverse
  frozen opponents and only graduate when they provably beat the champion
  (Stockfish fishtest SPRT + AlphaStar PFSP).

Hidden information (positions, runes, summoners, **intended builds**) is already
modelled: the env never exposes enemy roles/builds; `make_pool_sampler`
marginalises terminal reward over plausible (role, build) joint worlds. The
league must keep this boundary — opponents see only public draft state.

## Current state (2470 LOC, 14 files) and the 5 redundancies

| Redundancy | Where | Cost |
|---|---|---|
| R1 Reward double-loop | `reward.py:210-221` calls scalar `predictor()` n_b×n_r times | **speed** — up to 64 forward passes/terminal instead of 1 |
| R2 Duplicate net trunk | `policy.py:51-57` ≈ `alpha_net.py:30-37` | 1 extra file, ~15 LOC |
| R3 Duplicate draft rules | `DraftEnv` (`env.py`) vs `DraftState` (`mcts.py`) two reps of the same transition table | ~50 LOC, drift risk |
| R4 Duplicate episode structs | `EpisodeBatch` (`rollout.py:34`) vs `EpisodeSamples` (`selfplay.py:29`) + two concat helpers | ~30 LOC — **evaluated, NOT merged** (see 1d) |
| R5 Duplicate worker plumbing | `_worker_init`/spawn-pool/`_state_to_bytes`/`set_num_threads` in `rollout.py` **and** `alpha_train.py` | ~40 LOC |

## Target architecture

```
draft.py     sequence + Side/ActionType + DraftState (single transition truth)
reward.py    Predictor protocol (+ optional predict_batch), sampler, resolve_rewards (batched)
pool.py      champion (role,build) eligibility pool            [unchanged]
net.py       encode_obs/obs_dim + trunk + MaskedPolicy + AlphaZeroNet + auto_device  (was policy.py + alpha_net.py)
env.py       DraftEnv: thin gym wrapper over DraftState
mcts.py      PUCT search over DraftState
selfplay.py  play_episode → EpisodeSamples (AlphaZero); REINFORCE keeps EpisodeBatch
worker.py    spawn-pool + worker globals + state<->bytes  (shared by both trainers)
league.py    opponent pool + PFSP sampling + SPRT gating   [NEW]
train.py / alpha_train.py   thin loops over worker.py + league.py
example.py / alpha_smoke.py  dummy fixtures + end-to-end smoke   [unchanged API]
```

Net file delta: −2 (`alpha_net.py` folds into `net.py`; `policy.py`→`net.py`
rename) +2 (`worker.py`, `league.py`) ≈ flat file count, but ~+league capability
and ~165 fewer duplicated LOC.

---

## Phase 1 — Reductions (no behaviour change; do first, top-level)

Low-risk, high-ROI on criteria 1–3/5. Each gated by `python -m app.rl.alpha_smoke`
and `pytest tests/rl/` (both torch-light; safe per current idle GPU/RAM).

**Status:** 1a, 1b, 1c, 1e **landed and gated** (net −125 LOC; `policy.py` +
`alpha_net.py` → `net.py`; shared `worker.py`; one `DraftState`; batched reward).
1d **dropped** (rationale below). `gymnasium` stays isolated to `env.py` so the
AlphaZero/`DraftState` path runs without it.

### 1a. Batch the reward path (R1 — the speed win) — `reward.py`

- Add an **optional** batched hook. In `resolve_rewards`, after building
  `blue_configs`/`red_configs`, detect `fn = getattr(predictor, "predict_batch", None)`.
  - If present: build the full list of `(blue_team, red_team, bc.roles, rc.roles,
    bc.builds, rc.builds)` for every (i, j) pair, call `fn(games)` **once**, and
    reshape the returned `np.ndarray` into `win_matrix` (n_b, n_r).
  - Else: keep the existing scalar double-loop (dummy/test predictors).
- Add `predict_batch` to `WinRatePredictor` (`app/ml/predictor.py`): validate +
  convert each game to `blue_tuples + red_tuples` (reuse `_team_tuples`,
  `_validate_team_assignment`) and call the existing `_forward_probabilities`
  **once**. Returns `np.ndarray` of P(blue wins), one per game.
- Add `predict_batch` to the `Predictor` Protocol docstring as optional (a
  `BatchPredictor` note), not a required method — keeps closures valid.
- **Acceptance**: `tests/rl/test_reward.py` unchanged and green (hits scalar
  fallback); a new test asserts a stub exposing `predict_batch` yields the same
  `win_matrix` as the scalar path. Bench: terminal eval forward-pass count drops
  from n_b·n_r to 1.

### 1b. Merge nets (R2) — `alpha_net.py` → `net.py`

- Rename `policy.py` → `net.py`. Move `AlphaNetConfig`, `AlphaZeroNet`,
  `auto_device` into it; delete `alpha_net.py`.
- Factor one private `_trunk(d, hidden) -> nn.Sequential` used by both
  `MaskedPolicy` and `AlphaZeroNet`.
- Update imports `app.rl.alpha_net` → `app.rl.net` and `app.rl.policy` →
  `app.rl.net` in: `mcts.py`, `selfplay.py`, `alpha_train.py`, `train.py`,
  `rollout.py`, `__init__.py`, `alpha_smoke.py`.
- **Acceptance**: smoke + tests green; `__init__` re-exports unchanged names.

### 1c. One DraftState (R3) — move into `draft.py`, env wraps it

- Move `DraftState` from `mcts.py` to `draft.py` (it is foundational state, not
  search-specific). `mcts.py` imports it from `draft`.
- Reduce `DraftEnv` to a wrapper: hold a `DraftState`; `reset` builds
  `DraftState.initial` + applies `random_start_steps` via `np_random`; `step`
  calls `state.apply` then `resolve_rewards`/optimizer on terminal; `_obs`/
  mask/`current_step` delegate to the state. Remove `env.py`'s `_apply`,
  `_pad_indices`, `_obs`, `get_action_mask` duplicates (keep `get_action_mask`
  as a one-line delegate — rollout.py uses `obs["available_mask"]`, MCTS uses
  the state, so the public obs dict is unchanged).
- Keep the int8 obs dtypes exactly (gym space contract). `DraftState.available`
  is int8 {0,1}; `available_mask` stays MultiBinary-compatible.
- **Acceptance**: smoke + tests green; an episode vs `example.dummy_*` produces a
  legal 20-step draft and terminal reward in [−0.5, 0.5]. **Highest-risk item —
  land 1a/1b first, run gates, then 1c alone.**

### 1d. One Episode struct (R4) — **DROPPED**

Examined both structs in full: `EpisodeBatch` (REINFORCE) carries `actions`,
`returns`, and **per-episode** `blue_rewards`/`red_rewards`/`p_blue_win` arrays;
`EpisodeSamples` (AlphaZero) carries `policy_targets`, `value_targets`, `sides`,
and **scalar** rewards + `info`. Only `features`/`masks` overlap, and even the
reward fields differ in shape (array vs scalar). A unified superset dataclass
would leave ~half its fields `None`/unused in each path and force conditional
concat — a **clarity regression** that fails success criteria 3. The remaining
duplication is two ~10-line field-wise concat helpers; merging the containers
costs more clarity than it saves LOC. The real R4-adjacent lever is Phase 3
(whether the REINFORCE stack should exist at all), which removes `EpisodeBatch`
wholesale rather than fusing it.

### 1e. Shared worker plumbing (R5) — new `worker.py`

- Extract `state_to_bytes`/`bytes_to_state`, `make_spawn_pool(initializer,
  initargs)`, and the `torch.set_num_threads(1)` worker preamble. `rollout.py`
  and `alpha_train.py` call these instead of redefining them. Remove the
  cross-module private import `from app.rl.rollout import _state_to_bytes` in
  `alpha_train.py:39`.
- **Acceptance**: smoke + tests green; no `_`-prefixed cross-module imports.

---

## Phase 2 — Adversarial league (the headline; additive, after Phase 1)

New `league.py` + minimal `alpha_train.py` hooks. Reuses the Phase-1 unified
net + worker + DraftState. Stockfish/AlphaStar practices, kept lean.

### Concepts

- **Pool**: list of frozen checkpoints `{path, rating, games, wins}` persisted as
  `data/policies/league/index.json` + `*.pt` snapshots. The first entry is the
  current champion.
- **PFSP opponent sampling** (AlphaStar): per league episode, sample opponent
  with weight `∝ (1 − winrate_vs)^p` (default p=2), bootstrapped uniform when a
  pairing has no games. Concentrates training on hard-but-beatable adversaries —
  the user's "strong opponents that introduce new ideas". A configurable
  fraction `self_play_frac` still plays vs the current learner for stability.
- **Asymmetric episode**: learner plays one side with full MCTS; the sampled
  frozen opponent plays the other side with its own net (cheap: policy-head
  greedy or low-sim MCTS). Only learner-side steps become training samples
  (clean signal, mirrors the existing `vs_random` path). Opponent sees only
  public draft state — hidden-info boundary preserved.
- **SPRT promotion** (Stockfish fishtest): periodically play N eval games
  learner-vs-champion; maintain the log-likelihood ratio for H0: Elo≤elo0 vs
  H1: Elo≥elo1. On crossing the upper bound, **snapshot the learner into the
  pool and promote it to champion**; on the lower bound, reject and keep
  training. SPRT plays only as many games as needed → directly serves the
  "fewer iterations / cost" criterion.
- **Elo** updates after each recorded result; tracks frontier progress and feeds
  PFSP weights.

### `league.py` surface (lean)

```
@dataclass LeagueEntry(path, rating, games, wins)
class League:
    entries: list[LeagueEntry]; champion_idx: int
    @classmethod load(dir)->League        save(dir)
    sample_opponent(rng, p=2.0)->LeagueEntry      # PFSP
    record(entry, agent_won: bool)                 # Elo + H2H counts
    admit(net_state, rating)->None                 # snapshot into pool
def sprt(wins, losses, draws, elo0, elo1, alpha, beta)->Literal["accept","reject","continue"]
def elo_update(r_a, r_b, score_a, k=16)->tuple[float,float]
```

### `alpha_train.py` hooks (minimal)

- Config: `league: bool=False`, `league_dir`, `self_play_frac=0.5`,
  `eval_games=64`, `elo0=0.0`, `elo1=15.0`, `sprt_alpha=0.05`, `sprt_beta=0.05`,
  `promote_every=5`.
- In generation: when `league` and `rng()>self_play_frac`, sample an opponent
  and run an asymmetric episode (opponent side fixed net); else current
  self-play. Wire opponent weights through the worker like learner weights.
- Every `promote_every` iters: run `eval_games` learner-vs-champion, feed results
  to `sprt`; on "accept" call `league.admit` + reset the SPRT counters.
- **Acceptance**: extend `alpha_smoke.py` with a league smoke (2 dummy entries,
  ≤4 eval games) asserting: PFSP sampling returns a valid entry, an asymmetric
  episode yields legal drafts, `sprt` returns one of the three states, `admit`
  grows the pool. No ClickHouse, dummy predictor only.

---

## Phase 3 — Stack-consolidation decision (deferred, flagged)

Once the league makes MCTS self-play the canonical strong-agent path, the
REINFORCE stack (`train.py` + `vs_random` half of `rollout.py` + `MaskedPolicy`)
is largely redundant and would be the single biggest LOC/file cut (~400 LOC).
**Do not auto-delete** — it is the cheap `vs_random` baseline/sanity loop and a
one-way door. Surface as an explicit recommendation; execute only on user
confirm. If retained, it still benefits from Phases 1a/1b/1e and is the only
remaining consumer of `EpisodeBatch`/`MaskedPolicy`/`RolloutPool` (~485 LOC);
removing it is the single biggest LOC/file cut left.

---

## Verification & resource discipline

- Gates (both torch-light, no ClickHouse, no training cache):
  `python -m app.rl.alpha_smoke` and `pytest tests/rl/ -q`.
- Before any run: `nvidia-smi` + `free -h`; serialize — one torch job at a time.
  No `app.ml` training, ClickHouse aggregations, or cache loads while another
  ML job is active.
- README/EXPERIMENTS docs updated per phase (delete stale `MaskedPolicy`/two-net
  references; document the league + SPRT gating).

## Orchestration plan (who does what)

- **Top-level (Claude)**: Phase 1a/1c (correctness-critical, tightly coupled,
  warm-context → most token-efficient here), integration, and final gates.
- **Cheap executor(s)** where isolation/parallelism justifies the spawn:
  the mechanical net rename+import sweep (1b), the docs rewrite, and a
  fresh-context Level-3 review of the Phase-1 diff. Each gets a decision-complete
  packet (files, exact edits, acceptance) and the cheapest adequate model.
- **Loop**: after each phase lands and gates pass, re-review for the next
  highest-ROI reduction. Stop when no change improves the criteria without
  hurting clarity (definition of "optimised").
