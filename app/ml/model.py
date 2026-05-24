from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

N_PLAYER_FEATURES = 10
N_MATCHUP_1V1 = 25
N_SYNERGY_2VX = 20  # 10 blue + 10 red

POSITIONS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("blue", "red")

# Raw per-game priors emitted by the cache.
RAW_FEATURE_NAMES: tuple[str, ...] = (
    *(
        f"win_rate_1vx_{side}_{pos}"
        for side in SIDES
        for pos in POSITIONS
    ),
    *(
        f"matchup_1v1_blue_{blue_pos}_vs_red_{red_pos}"
        for blue_pos in POSITIONS
        for red_pos in POSITIONS
    ),
    *(
        f"synergy_2vx_{side}_{i}_{j}"
        for side in SIDES
        for i, j in (
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4),
            (3, 4),
        )
    ),
)
assert len(RAW_FEATURE_NAMES) == N_PLAYER_FEATURES + N_MATCHUP_1V1 + N_SYNERGY_2VX

FEATURE_NAMES: tuple[str, ...] = RAW_FEATURE_NAMES
N_MODEL_FEATURES = len(FEATURE_NAMES)


def _validate(
    win_rate: np.ndarray, matchup_1v1: np.ndarray, synergy_2vx: np.ndarray
) -> None:
    if win_rate.ndim != 2 or win_rate.shape[1] != N_PLAYER_FEATURES:
        raise ValueError(f"win_rate shape {win_rate.shape}")
    if matchup_1v1.ndim != 2 or matchup_1v1.shape[1] != N_MATCHUP_1V1:
        raise ValueError(f"matchup_1v1 shape {matchup_1v1.shape}")
    if synergy_2vx.ndim != 2 or synergy_2vx.shape[1] != N_SYNERGY_2VX:
        raise ValueError(f"synergy_2vx shape {synergy_2vx.shape}")


def engineer_features(
    win_rate: np.ndarray,
    matchup_1v1: np.ndarray,
    synergy_2vx: np.ndarray,
) -> np.ndarray:
    _validate(win_rate, matchup_1v1, synergy_2vx)
    return np.column_stack(
        [
            win_rate.astype(np.float64, copy=False),
            matchup_1v1.astype(np.float64, copy=False),
            synergy_2vx.astype(np.float64, copy=False),
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

    def predict(
        self,
        win_rate: np.ndarray,
        matchup_1v1: np.ndarray,
        synergy_2vx: np.ndarray,
    ) -> np.ndarray:
        features = engineer_features(win_rate, matchup_1v1, synergy_2vx)
        return expit(features @ self.weights + self.intercept)

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


def fit_logistic_regression(
    win_rate: np.ndarray,
    matchup_1v1: np.ndarray,
    synergy_2vx: np.ndarray,
    blue_win: np.ndarray,
) -> WinRateLinearModel:
    features = engineer_features(win_rate, matchup_1v1, synergy_2vx)
    y = blue_win.astype(np.float64, copy=False)
    x = np.column_stack([np.ones(features.shape[0], dtype=np.float64), features])

    def loss_and_grad(coeffs: np.ndarray) -> tuple[float, np.ndarray]:
        p = expit(x @ coeffs)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        loss = -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        grad = x.T @ (p - y) / y.size
        return loss, grad

    result = minimize(
        loss_and_grad,
        x0=np.zeros(x.shape[1]),
        jac=True,
        method="L-BFGS-B",
    )
    return WinRateLinearModel(
        intercept=float(result.x[0]),
        weights=np.asarray(result.x[1:], dtype=np.float64),
    )


__all__ = [
    "FEATURE_NAMES",
    "N_MATCHUP_1V1",
    "N_MODEL_FEATURES",
    "N_PLAYER_FEATURES",
    "N_SYNERGY_2VX",
    "RAW_FEATURE_NAMES",
    "WinRateLinearModel",
    "engineer_features",
    "fit_logistic_regression",
]
