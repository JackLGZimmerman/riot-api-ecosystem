from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_PLAYER_FEATURES = 10
POSITIONS = ("top", "jungle", "middle", "bottom", "utility")
STAT_NAMES = ("mean", "min", "max", "variance")
FEATURE_NAMES = (
    *(f"{position}_diff" for position in POSITIONS),
    *(f"blue_{name}" for name in STAT_NAMES),
    *(f"red_{name}" for name in STAT_NAMES),
    *(f"{name}_diff" for name in STAT_NAMES),
)
N_MODEL_FEATURES = len(FEATURE_NAMES)


def _validate_win_rate(win_rate: np.ndarray) -> None:
    if win_rate.ndim != 2 or win_rate.shape[1] != N_PLAYER_FEATURES:
        raise ValueError(
            f"Expected win_rate with shape [n, {N_PLAYER_FEATURES}], "
            f"got {win_rate.shape}"
        )


def _team_statistics(team: np.ndarray) -> np.ndarray:
    # Skew and kurtosis can be added here later, but each team only has five
    # values, so those higher moments are noisy as first-pass active features.
    return np.column_stack(
        [
            np.mean(team, axis=1),
            np.min(team, axis=1),
            np.max(team, axis=1),
            np.var(team, axis=1),
        ]
    )


def engineer_features(win_rate: np.ndarray) -> np.ndarray:
    _validate_win_rate(win_rate)

    values = win_rate.astype(np.float64, copy=False)
    blue = values[:, :5]
    red = values[:, 5:]
    blue_stats = _team_statistics(blue)
    red_stats = _team_statistics(red)
    return np.column_stack(
        [
            blue - red,
            blue_stats,
            red_stats,
            blue_stats - red_stats,
        ]
    )


@dataclass(frozen=True)
class WinRateLinearModel:
    intercept: float
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.weights.shape != (N_MODEL_FEATURES,):
            raise ValueError(
                f"Expected {N_MODEL_FEATURES} weights, got {self.weights.shape}"
            )

    def predict(self, win_rate: np.ndarray) -> np.ndarray:
        features = engineer_features(win_rate)
        return np.clip(features @ self.weights + self.intercept, 0.0, 1.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(
                handle,
                intercept=np.asarray(self.intercept, dtype=np.float64),
                weights=self.weights.astype(np.float64, copy=False),
            )

    @classmethod
    def load(cls, path: Path) -> WinRateLinearModel:
        with np.load(path) as data:
            return cls(
                intercept=float(data["intercept"]),
                weights=np.asarray(data["weights"], dtype=np.float64),
            )


def fit_linear_regression(
    win_rate: np.ndarray,
    blue_win: np.ndarray,
) -> WinRateLinearModel:
    features = engineer_features(win_rate)

    x = np.column_stack(
        [
            np.ones(features.shape[0], dtype=np.float64),
            features,
        ]
    )
    coefficients, *_ = np.linalg.lstsq(
        x,
        blue_win.astype(np.float64, copy=False),
        rcond=None,
    )
    return WinRateLinearModel(
        intercept=float(coefficients[0]),
        weights=np.asarray(coefficients[1:], dtype=np.float64),
    )


__all__ = [
    "FEATURE_NAMES",
    "N_MODEL_FEATURES",
    "N_PLAYER_FEATURES",
    "WinRateLinearModel",
    "engineer_features",
    "fit_linear_regression",
]
