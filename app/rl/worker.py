"""Shared multiprocessing helpers for the self-play trainer.

Deliberately free of the gymnasium env import: the AlphaZero path searches
over ``DraftState`` (not ``DraftEnv``), so importing these helpers must not
pull ``gymnasium`` in. ``alpha_train.py`` uses them instead of redefining
state<->bytes and spawn-pool plumbing.
"""

from __future__ import annotations

import io
import os
from multiprocessing import get_context
from multiprocessing.pool import Pool
from typing import Any, Callable

import torch


def state_to_bytes(state: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def bytes_to_state(data: bytes, *, map_location: Any = None) -> dict:
    return torch.load(
        io.BytesIO(data), map_location=map_location, weights_only=True
    )


def make_spawn_pool(
    n_workers: int, initializer: Callable[..., None], initargs: tuple
) -> Pool:
    """Persistent spawn-context pool (spawn avoids CUDA/fork issues)."""
    return get_context("spawn").Pool(
        n_workers, initializer=initializer, initargs=initargs
    )


def default_workers(n_workers: int | None) -> int:
    return n_workers or max(1, (os.cpu_count() or 2) - 1)
