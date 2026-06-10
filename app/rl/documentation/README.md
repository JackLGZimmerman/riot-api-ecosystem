# Draft RL

Gymnasium environment + REINFORCE trainer that drafts champions and is
scored by the win-probability model in `app/ml`.

## Overview

`DraftEnv` plays one tournament draft per episode. Intermediate rewards
are zero; at the terminal step the environment resolves hidden
role/build assignments and queries a `Predictor` to compute the reward.

```text
DraftEnv(predictor, DraftEnvConfig(...), sampler=..., optimizer=...)
```

One of `sampler` or `optimizer` is **required** — there is no default.
The previous pick-order role assignment was unsound and has been
removed: roles and builds are unknown until the end of the draft, and
picks must be made against the full set of plausible role/build
assignments available to each champion (see "Champion Pool" below).

Internal action space is `Discrete(len(champion_ids))` — positional
indices into the `champion_ids` tuple, not raw champion IDs. Real
champion IDs (sparse, 1..950) are only resolved at the predictor
boundary.

Internal state is a single `int8` ownership vector of shape
`(n_champions,)` with codes `{0: AVAILABLE, 1: BLUE_BAN, 2: RED_BAN, 3:
BLUE_PICK, 4: RED_PICK}`. The four pick/ban index arrays and the
available mask in the observation are derived from it on each `_obs`
call, so all four are always consistent with each other.

## Draft Sequence

20 actions per episode, defined in `draft.py`:

```text
BB1, RB1, BB2, RB2, BB3, RB3,
B1,  R1,  R2,  B2,  B3,  R3,
RB4, BB4, RB5, BB5,
R4,  B4,  B5,  R5
```

`BB` = blue ban, `RB` = red ban, `B` = blue pick, `R` = red pick.

## Roles

Role labels match `app/ml/config.py::POSITIONS` exactly:

```text
TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
```

No translation table — the RL side now produces the canonical names the
DB and ML model already use.

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

Enemy roles and builds are never exposed during the draft.

## Action

`Discrete(n_champions)`. Legal actions are indices whose
`available_mask` bit is `1`. `env.get_action_mask()` returns a bool
vector that the policy must respect.

If an illegal action is passed to `step()`:

- Reward = `illegal_action_penalty` (default `-1.0`).
- Episode terminates if `terminate_on_illegal=True` (default).

## Champion Pool

Each champion has a per-role, per-build eligibility set — most
champions only see play in 1-2 roles, and each role has a small set of
realistic builds. The pool is a JSON file checked into the repo:

```text
app/rl/data/champion_pool.json
```

Format: `{ "champion_id": [["ROLE", build_id, weight], ...] }`. Weights
are relative likelihoods within a champion; `make_pool_sampler` uses
them to rank (role-assignment, build-assignment) combinations.

Generate from the priors:

```bash
python -m app.rl.pool generate --min-matchups 50
```

This scans `priors.p1` (the `(champion, role, build) -> (win_rate,
matchups)` table) and keeps every entry with at least `min_matchups`
observations. `weight = matchups`, so combinations with the most
evidence dominate the top-K sample.

## Sampler

`make_pool_sampler(pool, top_k_build_configs)` returns the
`RoleBuildSampler` the env (or MCTS) calls at the terminal step:

1. Enumerate every permutation of the team's 5 champions across the 5
   roles. Skip any permutation where a `(champion, role)` is missing
   from the pool.
2. For each valid role-assignment, enumerate the Cartesian product of
   per-champion build candidates.
3. Score each `(role-assignment, build-assignment)` by the product of
   per-champion weights. Keep the top `top_k_build_configs` overall and
   normalise their `probability` to sum to 1.

`top_k_build_configs` directly caps predictor calls per terminal:
`top_k_build_configs²` per team-pair, since `resolve_rewards` evaluates
the full `n_b × n_r` matrix.

## Reward

Zero at every intermediate step. At the terminal step,
`resolve_rewards` builds the full `P(blue|cfg_blue, cfg_red)` matrix
and aggregates per `reward_mode`:

| Mode | `p_blue_win` for blue | `p_blue_win` for red |
| --- | --- | --- |
| `expected_value` | `mean(win_matrix)` | `mean(win_matrix)` |
| `risk_adjusted` | `mean − λ·std` | `mean + λ·std` |
| `worst_case` | `min(win_matrix)` | `max(win_matrix)` |

Final reward per side:

```text
blue_reward = p_blue_win_for_blue − 0.5
red_reward  = (1 − p_blue_win_for_red) − 0.5
```

The scalar returned by `step()` is selected by `agent_side` (`blue`,
`red`, or `self_play`). `info` always contains both rewards.

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
`reset()`, so the agent starts every episode in a different mid-draft
state. Diversifies training data without changing the action space.

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

`WinRatePredictor` (in `app/ml/predictor.py`) loads `(champ_id, position,
build) -> win_rate` from `synergy_1vx` (train split only) at init, then maps
every reward query to the 10-slot win-rate vector plus supported identity
sidecar/group features. The current promoted validation checkpoint is not a
serving-compatible RL artifact because it requires Loadout and patch tensors;
`load_predictor()` rejects it until the predictor protocol carries those
runtime features or a no-feature-head serving checkpoint is promoted.

## Training

```bash
python -m app.rl.train
```

Architecture:

| Piece | File | Purpose |
| --- | --- | --- |
| Policy | `policy.py` | Torch MLP, masked categorical sampling |
| Workers | `rollout.py` | Persistent `multiprocessing.Pool`, one env + policy per worker |
| Trainer | `train.py` | REINFORCE + baseline + entropy, TensorBoard + JSONL logging |

### Throughput

The predictor is pure numpy after init, so the bottleneck is rollouts.
Workers are spawned once; on each epoch the master broadcasts fresh
weights and each worker runs a batch of episodes independently.
`torch.set_num_threads(1)` inside workers prevents intra-op contention.

Observed (7 workers, 8-core box): ~1500-1800 episodes/sec.

### Modes

| `train_mode` | Behaviour |
| --- | --- |
| `vs_random` (default) | Policy plays blue, uniform random plays red. Only blue transitions trained. Clean learning signal. |
| `self_play` | Policy plays both sides. Per-step return = reward of the side that took the step. |

### TrainConfig

```python
TrainConfig(
    top_k_build_configs=8,      # required
    epochs=200,
    episodes_per_worker=8,
    n_workers=None,             # default: cpu_count - 1
    lr=3e-4,
    entropy_coef=0.01,
    grad_clip=1.0,
    random_start_steps=0,
    eval_every=5,
    eval_episodes=64,
    hidden=256,
    train_mode="vs_random",
    run_name=None,              # default: draft_YYYYMMDD_HHMMSS
    pool_path=DEFAULT_POOL_PATH,
)
```

All fields are keyword-only.

## Live Performance Graph

Run TensorBoard against the run directory:

```bash
tensorboard --logdir app/rl/data/runs
```

Scalars per epoch: `policy_loss`, `entropy`, `grad_norm`,
`blue_reward_mean`, `p_blue_win_mean`, `ep_per_sec`. Every
`eval_every` epochs: `eval_blue_reward_mean`, `eval_p_blue_win_mean`,
`eval_win_rate` (policy-vs-random).

## JSONL Logging

Every epoch appends one JSON object to
`app/rl/data/logs/{run_name}.jsonl` for offline analysis.

## Verified Improvement

A 60-epoch `vs_random` run (~20 s wallclock) takes a fresh policy from
`eval_win_rate ≈ 0.55` to `≈ 0.85`, entropy from `5.08` to `3.3`. The
learning curve is visible live in TensorBoard.

## AlphaZero Learner

Stronger search-based learner that replaces vanilla REINFORCE with
self-play + MCTS + a policy-value net. Reuses everything documented
above: same `DraftEnv` rules (mirrored in `DraftState`), same
`Predictor` boundary, same `resolve_rewards` for terminal rewards, same
`reward_mode` semantics, same `encode_obs` features.

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
  `red_reward` for red steps. Both sides train against the same
  network.
- Training: policy = cross-entropy vs MCTS visit distribution; value =
  MSE vs the predictor-derived terminal reward.

### Setup

No new dependencies. Same `torch`, `numpy`, `gymnasium` already pinned
in `pyproject.toml`.

### Training command

```bash
python -m app.rl.alpha_train
```

Checkpoints land in `app/rl/data/policies/{run_name}.pt` and metrics
in `app/rl/data/logs/{run_name}.jsonl`.

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
`update_sec`. Checkpoints every `save_every` iterations.

### Predictor reward usage

The terminal reward returned by `resolve_rewards` is the only learning
signal. The value head is trained against the side's own reward
(`blue_reward` for blue-acting steps, `red_reward` for red-acting
steps), so both sides regress to their own perspective of the same
underlying `P(blue wins)` matrix produced by the predictor. The MCTS
backup applies the same per-side convention, so search and learner
agree.

### Smoke test

End-to-end check that does not touch ClickHouse — uses
`dummy_predictor` so it runs in a fresh repo on CPU or GPU:

```bash
python -m app.rl.alpha_smoke
```

It validates: device auto-selection, full 20-step legal draft sequence,
legal-action masking on every search policy, terminal reward in range,
one full self-play episode, and one learner update with finite losses
that does not increase total loss.

### Files

| File | Contents |
| --- | --- |
| `draft.py` | `Side`, `ActionType`, `DraftStep`, `DRAFT_SEQUENCE` |
| `pool.py` | `ChampionPool`, `PoolEntry`, `load_pool`, `build_pool_from_priors` |
| `reward.py` | `Predictor`, `RoleBuildConfig`, `make_pool_sampler`, `resolve_rewards` |
| `env.py` | `DraftEnv`, `DraftEnvConfig` (int8 ownership vector internally) |
| `policy.py` | `MaskedPolicy`, `encode_obs` (REINFORCE policy) |
| `rollout.py` | `RolloutPool`, persistent multiprocessing workers (REINFORCE) |
| `train.py` | REINFORCE training loop with TensorBoard + JSONL |
| `alpha_net.py` | `AlphaZeroNet`, `auto_device` (CUDA/MPS/CPU) |
| `mcts.py` | `DraftState`, `MCTS` (PUCT + beam), `visit_policy` |
| `selfplay.py` | `play_episode`, `EpisodeSamples` |
| `alpha_train.py` | AlphaZero training loop + `AlphaTrainConfig` |
| `alpha_smoke.py` | End-to-end smoke test (no ClickHouse) |
| `example.py` | Random episode with a dummy predictor and sampler |

## Dependencies

`gymnasium`, `numpy`, `torch`, `tensorboard`. All pinned in
`pyproject.toml`.
