from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.ml.cache_layout import (
    CACHE_FORMAT,
    CACHE_META_FILE,
    LOAD_ARRAY_DTYPES,
    N_PROFILE_BINS,
    N_PROFILE_FEATURES,
    VOCAB_FILE,
    array_paths,
)
from app.ml.config import (
    POSITIONS,
    DatasetConfig,
)

# Token layout fixed by 6900_ml_game_player_pivot_build.sql:
# slots 0-4 = blue (teamid=100), slots 5-9 = red (teamid=200), both in POSITIONS order.
# Side is implied by slot index, not stored; blue_win is the label for slots 0-4.
_PLAYER_ROLE_IDX = np.array(
    list(range(len(POSITIONS))) * 2,
    dtype=LOAD_ARRAY_DTYPES["role_idx"],
)


def _in_memory_tensor(name: str, array: np.ndarray) -> torch.Tensor:
    dtype = LOAD_ARRAY_DTYPES[name]
    return torch.from_numpy(np.array(array, dtype=dtype, copy=True))


def _decode_player_champion_build(
    packed: np.ndarray,
    n_builds: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.asarray(packed, dtype=np.uint32)
    champion_idx = encoded // np.uint32(n_builds)
    build_idx = encoded % np.uint32(n_builds)
    return (
        champion_idx.astype(LOAD_ARRAY_DTYPES["champion_idx"], copy=False),
        build_idx.astype(LOAD_ARRAY_DTYPES["build_idx"], copy=False),
    )


def _implied_role_idx(n_games: int) -> np.ndarray:
    return np.broadcast_to(_PLAYER_ROLE_IDX, (n_games, _PLAYER_ROLE_IDX.shape[0]))


@dataclass
class CachedTensors:
    champion_idx: torch.Tensor
    role_idx: torch.Tensor
    build_idx: torch.Tensor
    player_profile: torch.Tensor
    blue_win: torch.Tensor


@dataclass
class Vocab:
    n_champions: int
    n_builds: int
    n_roles: int
    n_profile_bins: int = N_PROFILE_BINS
    n_profile_features: int = N_PROFILE_FEATURES


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
    meta: dict[str, object],
) -> None:
    if meta.get("format") != CACHE_FORMAT:
        raise ValueError(
            "Dataset cache format is stale. "
            "Run `python -m app.ml.build_dataset` to rebuild it."
        )


def _profile_shape(meta: dict[str, object]) -> tuple[int, int]:
    profile_meta = meta.get("player_profile")
    if not isinstance(profile_meta, dict):
        return N_PROFILE_BINS, N_PROFILE_FEATURES
    shape = profile_meta.get("shape")
    if not isinstance(shape, list) or len(shape) != 4:
        return N_PROFILE_BINS, N_PROFILE_FEATURES
    return int(shape[2]), int(shape[3])


class _SplitView:
    """Minimal proxy exposing split size for `len(loader.dataset)`."""

    def __init__(self, n: int):
        self._n = int(n)

    def __len__(self) -> int:
        return self._n


class InMemoryBatchLoader:
    """Vectorised batch loader for cached tensor datasets.

    Reads from tensors that already live on the training device, so each batch
    is a single `index_select`. Eliminates DataLoader IPC, pickling, and per-item
    collation — the previous bottleneck once GPU warmup completed.
    """

    def __init__(
        self,
        tensors: CachedTensors,
        indices: torch.Tensor,
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
    ):
        self._tensors: dict[str, torch.Tensor] = {
            "champion_idx": tensors.champion_idx,
            "role_idx": tensors.role_idx,
            "build_idx": tensors.build_idx,
            "player_profile": tensors.player_profile,
            "blue_win": tensors.blue_win,
        }
        self._indices = indices
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._device = indices.device
        self.dataset = _SplitView(int(indices.shape[0]))

    def __len__(self) -> int:
        n = int(self._indices.shape[0])
        if self._drop_last and n >= self._batch_size:
            return n // self._batch_size
        return (n + self._batch_size - 1) // self._batch_size

    def iter_batches(self) -> Iterator[dict[str, torch.Tensor]]:
        n = int(self._indices.shape[0])
        if self._shuffle:
            order = self._indices.index_select(
                0, torch.randperm(n, device=self._device)
            )
        else:
            order = self._indices

        for start in range(0, n, self._batch_size):
            end = start + self._batch_size
            if end > n:
                if self._drop_last and n >= self._batch_size:
                    break
                end = n
            batch_idx = order[start:end]
            batch = {
                name: tensor.index_select(0, batch_idx)
                for name, tensor in self._tensors.items()
            }
            yield batch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return self.iter_batches()


def load_cache(cfg: DatasetConfig) -> tuple[CachedTensors, Vocab, dict[str, object]]:
    meta_path: Path = cfg.cache_dir / CACHE_META_FILE
    vocab_path: Path = cfg.cache_dir / VOCAB_FILE

    meta = json.loads(meta_path.read_text())
    _validate_cache(meta)
    n_games = int(meta["n_games"])
    vocab_meta = json.loads(vocab_path.read_text())
    arrays = {
        name: np.load(path, mmap_mode="r")[:n_games]
        for name, path in array_paths(cfg.cache_dir).items()
    }

    n_builds = int(vocab_meta["n_builds"])
    champion_idx, build_idx = _decode_player_champion_build(
        arrays["player_champion_build_idx"],
        n_builds,
    )
    role_idx = _implied_role_idx(n_games)
    n_profile_bins, n_profile_features = _profile_shape(meta)

    tensors = CachedTensors(
        champion_idx=_in_memory_tensor("champion_idx", champion_idx),
        role_idx=_in_memory_tensor("role_idx", role_idx),
        build_idx=_in_memory_tensor("build_idx", build_idx),
        player_profile=_in_memory_tensor(
            "player_profile",
            arrays["player_profile"],
        ),
        blue_win=_in_memory_tensor("blue_win", arrays["blue_win"]),
    )
    vocab = Vocab(
        n_champions=int(vocab_meta["n_champions"]),
        n_builds=n_builds,
        n_roles=int(vocab_meta["n_roles"]),
        n_profile_bins=n_profile_bins,
        n_profile_features=n_profile_features,
    )
    return tensors, vocab, meta


def _to_device(tensors: CachedTensors, device: torch.device) -> CachedTensors:
    return CachedTensors(
        champion_idx=tensors.champion_idx.to(device, non_blocking=True),
        role_idx=tensors.role_idx.to(device, non_blocking=True),
        build_idx=tensors.build_idx.to(device, non_blocking=True),
        player_profile=tensors.player_profile.to(device, non_blocking=True),
        blue_win=tensors.blue_win.to(device, non_blocking=True),
    )


def build_loaders(
    cfg: DatasetConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[
    InMemoryBatchLoader,
    InMemoryBatchLoader,
    InMemoryBatchLoader,
    Vocab,
]:
    tensors, vocab, meta = load_cache(cfg)
    n_games = tensors.blue_win.shape[0]
    train_idx_np, val_idx_np, test_idx_np = _cached_split(n_games, meta)

    # Hosting the full dataset on the training device removes per-batch H2D
    # transfer and pinning entirely. Compact disk dtypes are promoted before
    # tensors move to the device.
    tensors = _to_device(tensors, device)
    train_idx = torch.from_numpy(train_idx_np).to(device)
    val_idx = torch.from_numpy(val_idx_np).to(device)
    test_idx = torch.from_numpy(test_idx_np).to(device)

    train_loader = InMemoryBatchLoader(
        tensors,
        train_idx,
        batch_size,
        shuffle=True,
        drop_last=train_idx.shape[0] >= batch_size,
    )
    val_loader = InMemoryBatchLoader(
        tensors,
        val_idx,
        batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = InMemoryBatchLoader(
        tensors,
        test_idx,
        batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, vocab
