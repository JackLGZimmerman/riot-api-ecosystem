"""RL tools for League of Legends tournament drafting."""

from importlib import import_module

from app.rl.alpha_net import AlphaNetConfig, AlphaZeroNet, auto_device
from app.rl.draft import (
    DRAFT_SEQUENCE,
    ActionType,
    DraftStep,
    Side,
)
from app.rl.mcts import MCTS, DraftState, MCTSConfig, visit_policy
from app.rl.policy import MaskedPolicy, PolicyConfig, encode_obs, obs_dim
from app.rl.pool import (
    DEFAULT_POOL_PATH,
    ChampionPool,
    PoolEntry,
    build_pool_from_catalog,
    load_pool,
    save_pool,
)
from app.rl.reward import (
    OptimizationResult,
    Predictor,
    RewardMode,
    RoleBuildConfig,
    RoleBuildOptimizer,
    RoleBuildSampler,
    make_pool_sampler,
    resolve_rewards,
)
from app.rl.selfplay import EpisodeSamples, play_episode


def __getattr__(name: str):
    if name in {"DraftEnv", "DraftEnvConfig"}:
        module = import_module("app.rl.env")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionType",
    "AlphaNetConfig",
    "AlphaZeroNet",
    "ChampionPool",
    "DEFAULT_POOL_PATH",
    "DRAFT_SEQUENCE",
    "DraftEnv",
    "DraftEnvConfig",
    "DraftState",
    "DraftStep",
    "EpisodeSamples",
    "MCTS",
    "MCTSConfig",
    "MaskedPolicy",
    "OptimizationResult",
    "PolicyConfig",
    "PoolEntry",
    "Predictor",
    "RewardMode",
    "RoleBuildConfig",
    "RoleBuildOptimizer",
    "RoleBuildSampler",
    "Side",
    "auto_device",
    "build_pool_from_catalog",
    "encode_obs",
    "load_pool",
    "make_pool_sampler",
    "obs_dim",
    "play_episode",
    "resolve_rewards",
    "save_pool",
    "visit_policy",
]
