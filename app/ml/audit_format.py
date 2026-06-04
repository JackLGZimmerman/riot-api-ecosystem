"""Shared markdown-formatting helpers for the ML audit renderers.

Single source of truth for the NaN-safe statistics and percentage /
percentage-point formatters that both `context_examples_audit` and
`semantic_context_audit` render with.
"""

from __future__ import annotations

import numpy as np


def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _max_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def _format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{100.0 * value:.2f}%"


def _format_pp(value: float | int, *, signed: bool = True) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return "N/A"
    sign = "+" if signed and numeric >= 0.0 else ""
    return f"{sign}{100.0 * numeric:.2f} pp"


def _format_pp_mse(value: float | int) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return "N/A"
    return f"{10000.0 * numeric:.2f} pp^2"
