# file_storage.py
from __future__ import annotations

import asyncio
import aiofiles
import csv
import io
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar, Final, Dict
from functools import wraps
import zstandard as zstd
from pydantic import BaseModel

P = ParamSpec("P")
R = TypeVar("R")

_ZSTD_COMPRESSION: Final[int] = 15


# Registry can hold heterogeneous callables, so value type must be Callable[..., Any]
_EXPORT_REGISTRY: Final[Dict[str, Callable[..., Any]]] = {}


def register_exporter(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator factory to register an exporter function under `name`.

    Usage:
        @register_exporter("csv")
        def export_csv(path: str, rows: list[dict[str, Any]]) -> None: ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if name in _EXPORT_REGISTRY:
            raise ValueError(f"Exporter '{name}' is already registered")

        _EXPORT_REGISTRY[name] = func

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        return wrapper

    return decorator


@register_exporter(name="zstandard_streamed_export")
def zstandard_streamed_export(data: type[BaseModel]) -> None:
    pass
