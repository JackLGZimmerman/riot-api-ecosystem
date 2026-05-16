from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

METRIC_FLOAT_SIGNIFICANT_DIGITS = 6
MetricScalar = float | int


def metric_float(value: float) -> float:
    return float(f"{value:.{METRIC_FLOAT_SIGNIFICANT_DIGITS}g}")


def metric_scalar(value: object) -> MetricScalar | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def metric_value(value: object) -> object:
    scalar = metric_scalar(value)
    if scalar is not None:
        return metric_float(float(scalar)) if isinstance(scalar, float) else scalar
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [metric_value(v) for v in value]
    return value


def prefixed_fields(prefix: str, fields: Mapping[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in fields.items()}
