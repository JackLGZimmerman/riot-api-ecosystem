# file_storage.py
from __future__ import annotations

import asyncio
from functools import wraps
from pathlib import Path
from typing import Any, AsyncIterable, Callable, Dict, Final, ParamSpec, TypeVar

import zstandard as zstd
from pydantic import BaseModel

from app.core.config import settings

P = ParamSpec("P")
R = TypeVar("R")

_ZSTD_COMPRESSION: Final[int] = 15
DATA_PATH: Path = settings.data_path

_FILE_OPERATIONS: Final[Dict[str, Callable[..., Any]]] = {}


def get_file_operations() -> dict[str, Callable[..., Any]]:
    return _FILE_OPERATIONS.copy()


def storage_registry(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator factory to register an exporter function under `name`.

    Usage:
        @storage_registry("csv")
        def export_csv(path: str, rows: list[dict[str, Any]]) -> None: ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if name in _FILE_OPERATIONS:
            raise ValueError(f"Exporter '{name}' is already registered")

        _FILE_OPERATIONS[name] = func

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ========================== Zstandard readers & writers ==========================


@storage_registry(name="zstandard_streamed_export")
async def zstandard_streamed_export_async(
    rows: AsyncIterable[BaseModel],
    path: Path,
) -> None:
    cctx = zstd.ZstdCompressor(level=_ZSTD_COMPRESSION)  # default settings
    with open(path, "wb") as f:
        with cctx.stream_writer(f) as writer:  # single zstd frame
            async for item in rows:
                data = item.model_dump_json().encode("utf-8") + b"\n"
                await asyncio.to_thread(writer.write, data)


# ==================================================================================


# ========================== puuid index readers & writers ==========================
@storage_registry(name="load_puuid_index")
def load_puuid_index() -> set[str]:
    path = DATA_PATH / "puuid_seen_index.csv"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


@storage_registry(name="append_puuid_index")
def append_puuid_index(puuids: list[str]) -> None:
    path = DATA_PATH / "puuid_seen_index.csv"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for p in puuids:
            f.write(p + "\n")
