"""Gymnasium RL environment for League of Legends tournament drafting."""

from app.rl.alpha_net import AlphaNetConfig, AlphaZeroNet, auto_device
from app.rl.draft import (
    DRAFT_SEQUENCE,
    ActionType,
    DraftStep,
    Side,
)
from app.rl.env import DraftEnv, DraftEnvConfig
from app.rl.mcts import MCTS, DraftState, MCTSConfig, visit_policy
from app.rl.policy import MaskedPolicy, PolicyConfig, encode_obs, obs_dim
from app.rl.pool import (
    DEFAULT_POOL_PATH,
    ChampionPool,
    PoolEntry,
    build_pool_from_priors,
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
    "build_pool_from_priors",
    "encode_obs",
    "load_pool",
    "make_pool_sampler",
    "obs_dim",
    "play_episode",
    "resolve_rewards",
    "save_pool",
    "visit_policy",
]
