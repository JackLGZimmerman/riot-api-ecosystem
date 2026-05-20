from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from app.core.config.settings import PROJECT_ROOT

METRIC_FLOAT_DECIMALS = 4
# LR can be sub-1e-5 in the tail; keep significant-digit formatting for these.
HIGH_PRECISION_KEYS = frozenset({"lr", "base_lr", "initial_lr"})
MetricScalar = float | int


def metric_float(value: float, *, high_precision: bool = False) -> float:
    if high_precision:
        return float(f"{value:.6g}")
    return round(float(value), METRIC_FLOAT_DECIMALS)


def metric_scalar(value: object) -> MetricScalar | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        return str(path)


def metric_value(value: object, key: str | None = None) -> object:
    scalar = metric_scalar(value)
    if scalar is not None:
        if isinstance(scalar, float):
            return metric_float(
                scalar, high_precision=key in HIGH_PRECISION_KEYS
            )
        return scalar
    if isinstance(value, Path):
        return _relative_path(value)
    if isinstance(value, dict):
        return {str(k): metric_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [metric_value(v, key) for v in value]
    return value
