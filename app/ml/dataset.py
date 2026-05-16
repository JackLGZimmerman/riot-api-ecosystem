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
    LEGACY_ARRAY_FILES,
    LOAD_ARRAY_DTYPES,
    VOCAB_FILE,
    array_paths,
)
from app.ml.config import (
    POSITIONS,
    DatasetConfig,
)

_COMPATIBLE_CACHE_FORMATS = {CACHE_FORMAT, "npy-memmap-v5", "npy-memmap-v4", "npy-memmap-v3"}
_PLAYER_ROLE_IDX = np.array(
    [i + 1 for i in range(len(POSITIONS))] * 2,
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
    roles = np.broadcast_to(_PLAYER_ROLE_IDX, (n_games, _PLAYER_ROLE_IDX.shape[0]))
    return np.array(roles, dtype=LOAD_ARRAY_DTYPES["role_idx"], copy=True)


@dataclass
class CachedTensors:
    champion_idx: torch.Tensor
    role_idx: torch.Tensor
    build_idx: torch.Tensor
    blue_win: torch.Tensor


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
    if meta.get("format") not in _COMPATIBLE_CACHE_FORMATS:
        raise ValueError(
            "Dataset cache format is stale. "
            "Run `python -m app.ml.build_dataset` to rebuild it."
        )


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

    def iter_batches(
        self,
        *,
        shuffle: bool | None = None,
        drop_last: bool | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        n = int(self._indices.shape[0])
        use_shuffle = self._shuffle if shuffle is None else bool(shuffle)
        use_drop_last = self._drop_last if drop_last is None else bool(drop_last)

        if use_shuffle:
            order = self._indices.index_select(
                0, torch.randperm(n, device=self._device)
            )
        else:
            order = self._indices

        for start in range(0, n, self._batch_size):
            end = start + self._batch_size
            if end > n:
                if use_drop_last and n >= self._batch_size:
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
    n_games = int(meta["n_games"])
    vocab_meta = json.loads(vocab_path.read_text())
    cache_format = str(meta.get("format"))
    paths = (
        array_paths(cfg.cache_dir)
        if cache_format == CACHE_FORMAT
        else {
            name: cfg.cache_dir / filename
            for name, filename in LEGACY_ARRAY_FILES.items()
        }
    )
    arrays = {
        name: np.load(path, mmap_mode="r")[:n_games] for name, path in paths.items()
    }
    _validate_cache(meta)

    if "player_champion_build_idx" in arrays:
        n_builds = int(vocab_meta["n_builds"])
        champion_idx, build_idx = _decode_player_champion_build(
            arrays["player_champion_build_idx"],
            n_builds,
        )
        role_idx = _implied_role_idx(n_games)
    else:
        champion_idx = arrays["champion_idx"]
        role_idx = arrays["role_idx"]
        build_idx = arrays["build_idx"]

    tensors = CachedTensors(
        champion_idx=_in_memory_tensor("champion_idx", champion_idx),
        role_idx=_in_memory_tensor("role_idx", role_idx),
        build_idx=_in_memory_tensor("build_idx", build_idx),
        blue_win=_in_memory_tensor("blue_win", arrays["blue_win"]),
    )
    vocab = Vocab(
        n_champions=int(vocab_meta["n_champions"]),
        n_builds=int(vocab_meta["n_builds"]),
        n_roles=int(vocab_meta["n_roles"]),
        n_sides=int(vocab_meta["n_sides"]),
    )
    return tensors, vocab, meta


def _to_device(tensors: CachedTensors, device: torch.device) -> CachedTensors:
    return CachedTensors(
        champion_idx=tensors.champion_idx.to(device, non_blocking=True),
        role_idx=tensors.role_idx.to(device, non_blocking=True),
        build_idx=tensors.build_idx.to(device, non_blocking=True),
        blue_win=tensors.blue_win.to(device, non_blocking=True),
    )


def build_loaders(
    cfg: DatasetConfig,
    batch_size: int,
    device: torch.device,
    train_monitor_samples: int = 0,
) -> tuple[
    InMemoryBatchLoader,
    InMemoryBatchLoader,
    InMemoryBatchLoader,
    InMemoryBatchLoader | None,
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
    # Held-in train slice: a fixed, unshuffled subset the model trains on,
    # evaluated through the validation path so train/val gaps are comparable.
    train_monitor_loader: InMemoryBatchLoader | None = None
    if train_monitor_samples > 0 and train_idx.shape[0] > 0:
        monitor_n = min(int(train_monitor_samples), int(train_idx.shape[0]))
        train_monitor_loader = InMemoryBatchLoader(
            tensors,
            train_idx[:monitor_n],
            batch_size,
            shuffle=False,
            drop_last=False,
        )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_monitor_loader,
        vocab,
    )
