# Draft RL

Gymnasium environment + AlphaZero self-play learner that drafts champions
and is scored by the win-probability model in `app/ml`.

## Overview

`DraftEnv` plays one tournament draft per episode. Intermediate rewards
are zero; at the terminal step the environment resolves hidden
role/build assignments and queries a `Predictor` to compute the reward.

```text
DraftEnv(predictor, DraftEnvConfig(...), sampler=..., optimizer=...)
```

One of `sampler` or `optimizer` is **required** — there is no default.
Roles and builds are unknown until the end of the draft; picks must be
made against the full set of plausible role/build assignments available
to each champion (see [Champion Pool](#champion-pool) below).

Internal action space is `Discrete(len(champion_ids))` — positional
indices into the `champion_ids` tuple, not raw champion IDs. Real
champion IDs (sparse, 1..950) are only resolved at the predictor
boundary.

Internal state is a `DraftState` (defined in `draft.py`), which holds four
Python lists: `blue_picks`, `red_picks`, `blue_bans`, `red_bans` (positional
champion indices), plus an `int8` `available` vector of shape `(n_champions,)`
that is the legal mask (`1` = selectable, `0` = taken), and `step_idx`.
`DraftState.to_obs()` produces the observation dict. `DraftEnv` is a thin
gym wrapper over `DraftState`.

---

## Draft Sequence

20 actions per episode, defined in `draft.py`:

```text
BB1, RB1, BB2, RB2, BB3, RB3,
B1,  R1,  R2,  B2,  B3,  R3,
RB4, BB4, RB5, BB5,
R4,  B4,  B5,  R5
```

`BB` = blue ban, `RB` = red ban, `B` = blue pick, `R` = red pick.

---

## Roles

Role labels match `app/ml/config.py::POSITIONS` exactly:

```text
TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
```

---

## Inputs and Observability

Each player-facing decision during champion select maps to a different
visibility regime and a different model-input path.

| Decision | Visibility | Who controls | Model-input path |
| --- | --- | --- | --- |
| Bans | Public | Each side | Shapes the legal pool (observation + mask); not scored |
| Picks (champion) | Public | Each side | `champion_id` per slot |
| Roles (position) | Own-known; enemy hidden | Own team assigns | `teamposition` key for priors / sidecar / semantic |
| Builds (items) | Never visible until in-game | Own team intends | `build_id` per slot for train-supported candidate worlds — **implemented** |
| Patch | Public match metadata | Queue/deployment context | Optional game-level provider for patch-head HGNN artifacts |

Summoner spells and runes are **not** model inputs. Patch metadata is allowed
only as public match/deployment context when the loaded HGNN checkpoint has a
patch residual head; it is never inferred by zero-filling a missing feature.

### Per-team observability asymmetry

During the draft the observation is **public-only**: picks, bans, mask,
step, acting side, and action type. Roles and builds are not public
observations and are not committed during champion select.

At the terminal step the env must complete each side's five champions into a
full hidden assignment before querying the model. The intended treatment is
asymmetric: the agent may choose or resolve **its own** role/build
strategy at the terminal boundary, but it **never sees the enemy's** hidden
strategy. Enemy configs must be handled as latent worlds under a marginal,
robust, or opponent-policy distribution. This is the build-intent principle
applied generally.

How that maps to the code today:

- **Default path — `make_pool_sampler` + `resolve_rewards`.** Both sides are
  enumerated symmetrically over their plausible role+build assignments from
  catalog priors. This is valid for passive population-marginal scoring and
  for robust bounds over hidden worlds. It is not an explicit own-build choice
  policy.
- **`RoleBuildOptimizer` hook (extension point, no default ships).** `env`,
  `mcts`, and `selfplay` accept an optional optimizer that replaces the
  sampler + reward-mode aggregation, which is where explicit own-side
  best-response against an enemy distribution would live. Concrete optimizers
  should use semantics such as `max_own E_enemy` or `max_own min_enemy`, not a
  global min/max that lets the environment also choose the acting side's bad
  build. It is a Protocol only today — supply a concrete optimizer to use it.

Role + build is the complete hidden-assignment surface; there is no loadout
dimension to resolve.

---

## Observation

`Dict` space, public draft state only:

| Key | Type | Meaning |
| --- | --- | --- |
| `blue_picks`, `red_picks` | `int32[5]` | Index of picked champion; `-1` empty |
| `blue_bans`, `red_bans` | `int32[5]` | Index of banned champion; `-1` empty |
| `available_mask` | `MultiBinary(n_champions)` | `1` = still draftable |
| `step` | `Discrete(21)` | Index of the next draft step |
| `acting_side` | `Discrete(2)` | `0` = blue, `1` = red |
| `action_type` | `Discrete(2)` | `0` = ban, `1` = pick |

Enemy roles and builds are never in the observation. `net.encode_obs` flattens
this to four multi-hot champion vectors (blue/red picks, blue/red bans) plus
three scalars (`acting_side`, `action_type`, `step / len(DRAFT_SEQUENCE)`).

---

## Action

`Discrete(n_champions)`. Legal actions are indices whose `available_mask` bit
is `1`. `env.get_action_mask()` returns a bool vector that the policy must
respect. `DraftState.apply()` validates legality and raises `ValueError` on a
masked action.

---

## Build-Label Boundary

The build-conditioned HGNN is useful in several regimes, but the source label
must travel with every score. Accepted RL rewards must never read the held-out
match's final build labels.

| Source label | Build source | Accepted use |
| --- | --- | --- |
| `oracle_observed_build` | Cached final `build_id` from the completed match | Diagnostic ceiling only. It measures how good the conditional model can be when the answer is already known; it must not feed served prediction, RL rewards, search targets, or policy training. |
| `pregame_marginal_build` | Train-only `P(build | champion, role)` catalog or train-only modal build | Leakage-free passive pregame prediction. This evaluates normal-player uncertainty when nobody has committed a hidden build strategy in the draft. |
| `rl_candidate` | A train-supported candidate build world chosen or weighted by RL/search/catalog logic | Safe counterfactual scoring. The agent can ask "what if my side used this supported build plan?" while enemy configs remain latent. |

Never use held-out final builds, player-specific priors, or other post-game
data in accepted RL rewards. It is fine for
the supervised model to learn a conditional response from train-split observed
builds; the boundary is that evaluation and RL/search scoring choose build
worlds from pregame-available train artifacts, not from the evaluated match's
future.

### Model interaction

The observed-build oracle answers a conditional-capacity question: "If the
final builds were magically known, how much signal can the model use?" It is
not a deployable draft policy input.

Leakage-free marginal/modal paths answer a passive pregame question: "Given the
public draft and train-only population behavior, what is the expected outcome
under normal hidden build uncertainty?"

RL/search uses the same conditional HGNN as a counterfactual engine: it scores
concrete, train-supported `rl_candidate` worlds so a policy or optimizer can
compare possible own strategies against latent enemy worlds. This is not
passive accuracy proof; it is decision analysis under a leakage-free catalog.

If a future no-build model wins passive NLL, serve that model for passive
pregame prediction while keeping the build-conditioned model available for
RL/search counterfactual scoring.

---

## Hybrid Draft + Build Strategy

The recommended architecture is hybrid:

1. **Stage 1 — public draft policy.** The RL action space remains champion
   picks and bans only. The observation stays public-only: picks, bans, mask,
   step, acting side, and action type.
2. **Stage 2 — private terminal strategy.** After the public draft is complete,
   each side resolves private role/build intent. Own configs are
   controllable strategy decisions. Enemy configs are hidden latent worlds
   handled by marginal population distributions, robust distributions, or
   explicit opponent-policy distributions.

The current symmetric sampler is appropriate when the objective is population
marginal scoring over hidden role/build worlds. Explicit build choice belongs
behind `RoleBuildOptimizer`, where the optimizer separates controllable own
configs from latent enemy configs.

Future full-agent league entries should freeze both the public draft policy
and the private strategy metadata: catalog version, source-label policy,
role/build resolver, retained-mass thresholds, and enemy distribution policy.
Per-iteration capture artifacts should record bans, picks with resolved
roles/builds, source labels, catalog version, retained mass, selected own
configs, and an enemy-distribution summary.

---

## Adapter: Env ↔ Model

The `Predictor` protocol (`app/rl/reward.py`) is the boundary between the
draft environment and the win-probability model.

### Current contract

```python
class Predictor(Protocol):
    def __call__(
        self,
        blue_team: list[int],
        red_team: list[int],
        blue_roles: dict[int, str],
        red_roles: dict[int, str],
        blue_builds: dict[int, int],
        red_builds: dict[int, int],
    ) -> float: ...
```

Optional `predict_batch(games) -> np.ndarray` scores a list of games in one
forward pass; `resolve_rewards` uses it when available to evaluate the whole
`n_blue_cfg × n_red_cfg` config matrix in a single call.

The adapter carries `(champion, role, build)` per slot. `WinRatePredictor`
(`app/ml/predictor.py`) then derives the model's actual inputs from those
three fields per slot:

- `champion_id` — vocab index into the HGNN identity embeddings
- `build_id` — vocab index from `model.config.build_vocab`
- `win_rate` / `p1_cnt` — smoothed 1vX prior keyed by `(champ, role, build)`,
  from the train split only
- encoder-sidecar blocks — keyed by `(champ, role, build)` identity
- semantic-group features — derived from the same identity triple

The adapter carries only `(champion, role, build)` per slot. Patch, when needed,
is a game-level provider bound when `load_predictor()` constructs the HGNN
adapter, not a per-slot field. Summoner spells and runes are not model inputs.

### Serving boundary

The HGNN adapter must fail fast for runtime-unsupported heads. Loadout heads are
not part of the maintained serving surface. Patch-head checkpoints are valid
only when `load_predictor(serving_patch=(season, patch))` or an explicit
`PatchFeatureProvider` is supplied; missing patch features are never silently
zeroed. Non-patch checkpoints load directly from `(champion, role, build)`.

---

## Champion Pool

Each champion has a per-role, per-build eligibility set derived from the
train-only `BuildCatalog`. The pool is generated local data at:

```text
app/rl/data/champion_pool.json
```

Format: `{ "__meta__": {"catalog_version": "..."}, "champion_id": [["ROLE", build_id, weight], ...] }`.

The generated pool is ignored by git; regenerate it after catalog changes.
Entry weight approximates `P(role, build | champion)` = the catalog's smoothed
`P(build | champion, role)` scaled by the cell's share of that champion's total
observed games, so the sampler's role-permutation ranking sees role
plausibility. With `core_only` (the default), profiles that miss the
`rl_core_min_count` / `rl_core_min_share` gates are dropped, keeping the RL
surface to well-supported `(role, build)` atoms. The pool file is stamped with
`catalog_version`.

Generate or regenerate:

```bash
python -m app.rl.pool generate              # core-only (default)
python -m app.rl.pool generate --include-supported  # include thinly supported profiles
```

---

## Sampler

`make_pool_sampler(pool, top_k_build_configs)` (`app/rl/reward.py`) returns the
`RoleBuildSampler` the env (or MCTS) calls at the terminal step:

1. Enumerate every permutation of the team's 5 champions across the 5 roles.
   Skip any permutation where a `(champion, role)` pair is absent from the
   pool.
2. For each valid role-assignment, enumerate the Cartesian product of
   per-champion build candidates from the pool.
3. Score each `(role-assignment, build-assignment)` by the product of
   per-champion weights. Keep the top `top_k_build_configs` overall and
   normalise their `probability` to sum to 1. `retained_mass` is computed from
   raw weights and stamped on every returned `RoleBuildConfig` so
   low-coverage terminals are visible.

`top_k_build_configs` directly caps predictor calls per terminal:
`top_k_build_configs²` cells per team-pair, since `resolve_rewards` evaluates
the full `n_blue × n_red` config matrix.

Two guards keep search from exploiting thinly-supported high-win-rate tails:
the `rl_core_*` gates drop weak `(role, build)` atoms at pool-generation time,
and `worst_case_min_probability` (a `resolve_rewards` arg, default `0.0` = off)
drops low-weight configs before the `worst_case` min/max. `retained_mass` is
surfaced for diagnostics, not enforcement.

---

## Reward

Zero at every intermediate step. At the terminal step, `resolve_rewards`
(`app/rl/reward.py`) builds the full `P(blue|cfg_blue, cfg_red)` matrix via
one batched predictor call and aggregates per `reward_mode`.

The built-in reward modes are passive or robust summaries over sampled hidden
worlds. `expected_value` marginalises over the joint distribution: both sides'
config probability vectors form an outer product and the matrix is summed
against that joint weight. `worst_case` is a global robust bound over plausible
config pairs (min for blue, max for red), with a
`worst_case_min_probability` tail guard that drops implausible configs before
taking min/max. It should not be interpreted as an active own-build optimizer.

When build choice becomes an explicit private strategy, use
`RoleBuildOptimizer` semantics that optimize only the acting side's
controllable configs against latent enemy worlds, such as `max_own E_enemy` or
`max_own min_enemy`.

| Mode | `p_blue_win` for blue | `p_blue_win` for red |
| --- | --- | --- |
| `expected_value` | joint-weighted mean | joint-weighted mean |
| `risk_adjusted` | weighted mean − λ·weighted std | weighted mean + λ·weighted std |
| `worst_case` | `min(win_matrix)` over plausible configs | `max(win_matrix)` over plausible configs |

Final reward per side:

```text
blue_reward = p_blue_win_for_blue − 0.5
red_reward  = (1 − p_blue_win_for_red) − 0.5
```

The scalar returned by `step()` is selected by `agent_side` (`blue`, `red`, or
`self_play`). `info` always carries `blue_reward`, `red_reward`,
`p_blue_win_for_blue`, `p_blue_win_for_red`, `win_matrix`, `blue_configs`, and
`red_configs`.

---

## Config

```python
DraftEnvConfig(
    top_k_build_configs=8,                       # required, no default
    champion_ids=tuple(predictor.champion_ids),  # required
    agent_side="self_play",
    reward_mode="expected_value",
    risk_lambda=0.5,
    random_start_steps=0,            # K random legal actions before agent acts
)
```

All fields are keyword-only (`@dataclass(kw_only=True)`).

`random_start_steps` pre-plays `K` uniform-random legal actions during
`reset()`, so the agent starts every episode in a different mid-draft state.
Diversifies training data without changing the action space.

---

## Plugging in the Real ML Model

```python
from pathlib import Path

from app.ml.config import TrainConfig
from app.ml.predictor import load_predictor
from app.rl.env import DraftEnv, DraftEnvConfig
from app.rl.pool import load_pool
from app.rl.reward import make_pool_sampler

predictor = load_predictor(
    cfg=TrainConfig(model_path=Path("<serving-compatible-hgnn.pt>"))
)
pool = load_pool()                   # reads app/rl/data/champion_pool.json
sampler = make_pool_sampler(pool, top_k_build_configs=8)
env = DraftEnv(
    predictor,
    DraftEnvConfig(
        top_k_build_configs=8,
        champion_ids=predictor.champion_ids,
    ),
    sampler=sampler,
)
```

`WinRatePredictor` (in `app/ml/predictor.py`) loads `(champ, role, build)`
priors from `synergy_1vx` (train split only) at init, then maps every reward
query to the full HGNN input tensor — champion embeddings, build embeddings,
smoothed 1vX win-rate priors, encoder-sidecar blocks, semantic-group features,
and optional provider-backed patch features.

The promoted production ensemble loads directly for RL serving:
`load_predictor()` rejects any checkpoint-required patch feature unless a
serving patch/provider is configured. See the [Serving boundary](#serving-boundary)
above.

---

## Training

The learner is search-based: self-play + MCTS + a policy-value net
(AlphaZero). It reuses everything documented above: same `DraftEnv`
rules (mirrored in `DraftState`), same `Predictor` boundary, same
`resolve_rewards` for terminal rewards, same `reward_mode` semantics, same
`encode_obs` features.

### Method

- Shared policy-value net (`AlphaZeroNet`): identical trunk for both
  sides; the value head is interpreted from the perspective of the side
  about to act at the encoded state.
- PUCT MCTS (`MCTS`) with legal-action masking. Per-node memory is
  capped to a `beam_width` of the top-K priors (beam/MCTS hybrid) so
  ~950-action nodes stay light.
- Leaf evaluation: net forward for non-terminal leaves; predictor via
  `resolve_rewards` for terminal leaves. Both are cached per episode
  (net cache keyed by `available_mask + step`; predictor cache keyed by
  the sorted blue/red pick sets, which is order-invariant).
- Self-play (`play_episode`): one full draft per episode; per-step
  sample = `(features, visit-distribution policy target, legal mask,
  acting side)`. Terminal predictor reward is broadcast back to each
  step as the value target — `blue_reward` for blue steps,
  `red_reward` for red steps. Both sides train against the same network.
- Training: policy = cross-entropy vs MCTS visit distribution; value =
  MSE vs the predictor-derived terminal reward.

### Setup

No new dependencies. Same `torch`, `numpy`, `gymnasium` already pinned
in `pyproject.toml`.

### Training command

```bash
HGNN_SERVING_PATCH_SEASON=16 HGNN_SERVING_PATCH=11 python -m app.rl.alpha_train
```

Use the season/patch known for the draft being served. The bare module command
is valid only for non-patch HGNN artifacts or callers that inject an explicit
`PatchFeatureProvider`; patch-head artifacts fail fast without patch metadata.

Checkpoints land in `app/rl/data/policies/{run_name}.pt` and metrics
in `app/rl/data/logs/{run_name}.jsonl`; both paths are generated local data
and ignored by git.

### `AlphaTrainConfig`

```python
AlphaTrainConfig(
    top_k_build_configs=8,        # required
    iterations=50,
    episodes_per_iter=16,
    n_workers=1,                  # >1 spawns a multiprocessing pool
    device="auto",                # "auto" | "cuda" | "mps" | "cpu"
    reward_mode="expected_value", # | "risk_adjusted" | "worst_case"
    risk_lambda=0.5,
    pool_path=DEFAULT_POOL_PATH,
    serving_patch_season=None,     # set with serving_patch for patch-head HGNNs
    serving_patch=None,            # both can also come from HGNN_SERVING_PATCH_*
    # MCTS
    simulations=64,
    c_puct=1.5,
    beam_width=32,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    temperature=1.0,
    temperature_drop_step=10,
    # Net + optimisation
    hidden=256,
    lr=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    batch_size=256,
    epochs_per_iter=2,
    value_loss_coef=1.0,
    # Adversarial league (off by default)
    league=False,
    league_dir=None,
    self_play_frac=0.5,
    eval_games=64,
    promote_every=5,
    elo0=0.0,
    elo1=15.0,
    sprt_alpha=0.05,
    sprt_beta=0.05,
    # Persistence
    run_name=None,
    save_every=5,
)
```

### CPU/GPU notes

`auto_device("auto")` picks CUDA > MPS > CPU. The net is small enough
that CPU works for small `simulations`/`beam_width`; CUDA helps mostly
when you scale `simulations` or batch large replay updates. Predictor
calls are pure-numpy and run on the host process regardless. With
`n_workers > 1`, each worker process loads its own predictor and net
copy and runs self-play independently; the master broadcasts fresh
weights once per iteration.

### Expected outputs

Per iteration, one JSONL row with `policy_loss`, `value_loss`,
`grad_norm`, `blue_reward_mean`, `red_reward_mean`, `play_sec`,
`update_sec`. Checkpoints every `save_every` iterations. These outputs are
local run artifacts, not maintained repository fixtures.

Planned draft-capture artifacts should be per-iteration and source-labeled:
public bans and picks; terminal own picks with roles/builds; catalog version;
retained mass; selected own configs; and an enemy-distribution summary. These
captures are diagnostics for replay and
TensorBoard inspection, not additional public observations for the policy.

### Predictor reward usage

The terminal reward returned by `resolve_rewards` is the only learning
signal. The value head is trained against the side's own reward
(`blue_reward` for blue-acting steps, `red_reward` for red-acting
steps), so both sides regress to their own perspective of the same
underlying `P(blue wins)` matrix produced by the predictor. The MCTS
backup applies the same per-side convention, so search and learner agree.

### Smoke test

End-to-end check that does not touch ClickHouse — uses `dummy_predictor`
so it runs in a fresh repo on CPU or GPU:

```bash
python -m app.rl.alpha_smoke
```

Validates: device auto-selection, full 20-step legal draft sequence,
legal-action masking on every search policy, terminal reward in range,
one full self-play episode, one learner update with finite losses that
does not increase total loss, and the adversarial league round-trip.

---

## Adversarial League

Optional self-improvement loop (`league=True`) so the learner keeps facing a
pool of strong, diverse frozen opponents instead of only itself — referencing
AlphaStar PFSP and Stockfish-fishtest SPRT practices, kept lean. The
hidden-information boundary is preserved: opponents see only public draft state.

- **Pool** (`league.py`): frozen checkpoints `{path, rating, games, wins}`
  persisted locally under `data/policies/league/` (`index.json` + `entry_k.pt`).
  The current best is the champion.
- **PFSP sampling**: each league episode samples an opponent with weight
  proportional to `(1 - learner_winrate_vs)^p` (default p=2), concentrating
  training on hard-but-beatable adversaries. A `self_play_frac` of episodes
  still play vanilla self-play for stability.
- **Asymmetric episode**: the learner plays one side with full MCTS; the frozen
  opponent plays the other side greedily from its own policy head (no search).
  Only learner-side steps become training samples — a clean single-agent signal.
- **SPRT promotion**: every `promote_every` iterations the learner plays
  `eval_games` side-balanced games vs the champion; a Wald SPRT (H0: Elo<=elo0
  vs H1: Elo>=elo1) decides accept/reject/continue. On accept the learner is
  snapshotted into the pool and promoted to champion. SPRT plays only as many
  games as needed, directly serving the "fewer iterations" goal.
- **Elo** tracks the learner's live rating and feeds PFSP weights.

`league=True` forces inline generation (the parallel worker pool is used only
for plain self-play). The `alpha_smoke` test covers the full round-trip: admit,
PFSP sample, asymmetric episode, league generation/eval glue, and SPRT.

If explicit private build strategy is added, a league entry should
freeze the complete agent, not just the draft network: policy checkpoint,
catalog version, source-label policy, role/build optimizer or sampler settings,
retained-mass thresholds, and enemy-distribution policy.

---

## Test Plan

This README pass is documentation-only; validate it with markdown/static diff
review. Later implementation should add tests for public-only observations,
`RoleBuildOptimizer` own-vs-enemy semantics, leakage source-label rejection,
and per-iteration source-labeled capture contents.

---

## Files

| File | Contents |
| --- | --- |
| `draft.py` | `Side`, `ActionType`, `DraftStep`, `DRAFT_SEQUENCE`, `DraftState` |
| `pool.py` | `ChampionPool`, `PoolEntry`, `load_pool`, `build_pool_from_catalog` |
| `reward.py` | `Predictor` (protocol), `RoleBuildConfig`, `RoleBuildOptimizer`, `make_pool_sampler`, `resolve_rewards` |
| `net.py` | `AlphaZeroNet`, `encode_obs`, `auto_device` |
| `env.py` | `DraftEnv`, `DraftEnvConfig` (thin gym wrapper over `DraftState`) |
| `mcts.py` | `MCTS` (PUCT + beam), `visit_policy` |
| `worker.py` | shared spawn-pool + `state_to_bytes`/`bytes_to_state` |
| `selfplay.py` | `play_episode`, `EpisodeSamples` |
| `league.py` | adversarial opponent pool: PFSP sampling + SPRT promotion + Elo |
| `alpha_train.py` | AlphaZero training loop + `AlphaTrainConfig` |
| `alpha_smoke.py` | end-to-end smoke test (no ClickHouse) |
| `example.py` | random episode with a dummy predictor and sampler |

---

## Dependencies

`gymnasium`, `numpy`, `torch`. All pinned in `pyproject.toml`.
