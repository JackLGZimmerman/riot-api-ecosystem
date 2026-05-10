from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.ml.build_dataset import (
    CACHE_META_FILE,
    NORM_FILE,
    VOCAB_FILE,
    _array_paths,
)
from app.ml.config import (
    INTERACTION_ROLES,
    INTERACTION_SIDES,
    INTERACTION_TYPES,
    N_INTERACTION_TOKENS,
    DatasetConfig,
)


@dataclass
class CachedTensors:
    interaction_score: torch.Tensor
    interaction_reliability: torch.Tensor
    champion_idx: torch.Tensor
    role_idx: torch.Tensor
    build_idx: torch.Tensor
    blue_win: torch.Tensor


@dataclass
class InteractionLayout:
    """Static per-token metadata shared across every game."""

    types: torch.Tensor  # (N_INTERACTION_TOKENS,)
    sides: torch.Tensor  # (N_INTERACTION_TOKENS,)
    roles: torch.Tensor  # (N_INTERACTION_TOKENS, N_ROLE_SLOTS)


@dataclass
class Vocab:
    n_champions: int
    n_builds: int
    n_roles: int
    n_sides: int


def _cached_split(
    n: int,
    meta: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_meta = meta.get("splits")
    if not isinstance(split_meta, dict):
        raise ValueError(
            "Dataset cache does not contain leakage-safe split metadata. "
            "Run `python -m app.ml.build_dataset` to rebuild it."
        )

    n_train = int(split_meta.get("train", 0))
    n_val = int(split_meta.get("val", 0))
    n_test = int(split_meta.get("test", 0))
    if n_train <= 0:
        raise ValueError("Training split is empty; increase the dataset size.")
    if n_train + n_val + n_test != n:
        raise ValueError(
            "Dataset cache split counts do not match n_games. "
            "Run `python -m app.ml.build_dataset` to rebuild it."
        )

    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n, dtype=np.int64)
    return train_idx, val_idx, test_idx


def _validate_cache(
    vocab_meta: dict[str, object],
    interaction_score: np.ndarray,
) -> None:
    cached_n = int(vocab_meta.get("n_interaction_tokens", 0))
    if cached_n and cached_n != N_INTERACTION_TOKENS:
        raise ValueError(
            "Dataset cache was built with a different interaction token count. "
            "Run `python -m app.ml.build_dataset` to rebuild it."
        )
    if interaction_score.shape[-1] != N_INTERACTION_TOKENS:
        raise ValueError(
            "Dataset cache interaction token count does not match the configured "
            "model layout. Run `python -m app.ml.build_dataset` to rebuild it."
        )
    cached_types = tuple(vocab_meta.get("interaction_types", ()))
    if cached_types and cached_types != INTERACTION_TYPES:
        raise ValueError(
            "Cached interaction token type ordering does not match config. "
            "Run `python -m app.ml.build_dataset` to rebuild it."
        )


class GameDataset(Dataset):
    """One game per index. Tensors stay shared across __getitem__ calls."""

    def __init__(self, tensors: CachedTensors, indices: np.ndarray):
        self.tensors = tensors
        self.indices = torch.from_numpy(indices.astype(np.int64))

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = self.indices[i]
        return {
            "interaction_score": self.tensors.interaction_score[idx],
            "interaction_reliability": self.tensors.interaction_reliability[idx],
            "champion_idx": self.tensors.champion_idx[idx],
            "role_idx": self.tensors.role_idx[idx],
            "build_idx": self.tensors.build_idx[idx],
            "blue_win": self.tensors.blue_win[idx],
        }


def interaction_layout() -> InteractionLayout:
    return InteractionLayout(
        types=torch.tensor(INTERACTION_TYPES, dtype=torch.long),
        sides=torch.tensor(INTERACTION_SIDES, dtype=torch.long),
        roles=torch.tensor(INTERACTION_ROLES, dtype=torch.long),
    )


def load_cache(cfg: DatasetConfig) -> tuple[CachedTensors, Vocab, dict[str, object]]:
    meta_path: Path = cfg.cache_dir / CACHE_META_FILE
    norm_path: Path = cfg.cache_dir / NORM_FILE
    vocab_path: Path = cfg.cache_dir / VOCAB_FILE
    paths = _array_paths(cfg.cache_dir)
    missing_paths = [
        p for p in (meta_path, norm_path, vocab_path, *paths.values()) if not p.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Dataset cache is incomplete: "
            f"{', '.join(str(p) for p in missing_paths)}. "
            "Run `python -m app.ml.build_dataset` first."
        )

    meta = json.loads(meta_path.read_text())
    n_games = int(meta["n_games"])
    arrays = {
        name: np.load(path, mmap_mode="r+")[:n_games] for name, path in paths.items()
    }
    vocab_meta = json.loads(vocab_path.read_text())
    _validate_cache(vocab_meta, arrays["interaction_score"])

    tensors = CachedTensors(
        interaction_score=torch.from_numpy(arrays["interaction_score"]),
        interaction_reliability=torch.from_numpy(arrays["interaction_reliability"]),
        champion_idx=torch.from_numpy(arrays["champion_idx"]),
        role_idx=torch.from_numpy(arrays["role_idx"]),
        build_idx=torch.from_numpy(arrays["build_idx"]),
        blue_win=torch.from_numpy(arrays["blue_win"]),
    )
    vocab = Vocab(
        n_champions=int(vocab_meta["n_champions"]),
        n_builds=int(vocab_meta["n_builds"]),
        n_roles=int(vocab_meta["n_roles"]),
        n_sides=int(vocab_meta["n_sides"]),
    )
    return tensors, vocab, meta


def build_loaders(
    cfg: DatasetConfig,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocab, InteractionLayout]:
    tensors, vocab, meta = load_cache(cfg)
    n_games = tensors.blue_win.shape[0]
    train_idx, val_idx, test_idx = _cached_split(n_games, meta)

    train_ds = GameDataset(tensors, train_idx)
    val_ds = GameDataset(tensors, val_idx)
    test_ds = GameDataset(tensors, test_idx)
    if len(train_ds) == 0:
        raise ValueError("Training split is empty; increase the dataset size.")

    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_ds) >= batch_size,
        **common,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, vocab, interaction_layout()
