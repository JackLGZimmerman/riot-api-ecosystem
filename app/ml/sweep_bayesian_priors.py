"""Sweep Bayesian prior settings for build-labeled win-rate priors.

Run with:
    uv run python -m app.ml.sweep_bayesian_priors --tests 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config.settings import PROJECT_ROOT
from app.core.logging.logger import setup_logging_config
from app.ml.config import ML_DATA_DIR, PLAYER_PIVOT_TABLE, SPLIT_TABLE
from app.ml.model import N_PLAYER_FEATURES, WinRateLinearModel, fit_linear_regression
from app.ml.utils.bayesian_smoothing import (
    DEFAULT_PRIOR_STRENGTH,
    bayesian_smoothed_rate,
)
from app.ml.utils.calibration import expected_calibration_error
from database.clickhouse.client import get_client

SPLIT_NAMES = ("train", "val", "test")
RAW_CACHE_FORMAT = "bayesian-prior-sweep-raw-v1"
RAW_CACHE_DIR = ML_DATA_DIR / "bayesian_prior_sweep_raw"
REPORT_PATH = ML_DATA_DIR / "bayesian_prior_sweep_latest.json"
CSV_PATH = ML_DATA_DIR / "bayesian_prior_sweep_latest.csv"
INTERACTION_COUNTS_TABLE = "game_data_filtered.ml_interaction_counts"
SPLITS = {"train": "train", "validation": "val", "test": "test"}
METRICS = ("n", "accuracy", "ece", "mse")
RANKED_METRICS = {
    "val_accuracy": True,
    "val_ece": False,
    "test_accuracy": True,
    "test_ece": False,
}

setup_logging_config()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepConfig:
    tests: int = 1000
    prior_mean_min: float = 0.30
    prior_mean_max: float = 0.50
    strength_min: float = 1.0
    strength_max: float = 1000.0
    ece_bins: int = 15
    max_games: int | None = None
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    raw_cache_dir: Path = RAW_CACHE_DIR
    report_path: Path = REPORT_PATH
    csv_path: Path = CSV_PATH


@dataclass(frozen=True)
class RawSplitData:
    win_rate: np.ndarray
    matchups: np.ndarray
    blue_win: np.ndarray


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _raw_paths(raw_cache_dir: Path) -> dict[str, Path]:
    return {
        "win_rate": raw_cache_dir / "raw_win_rate.npy",
        "matchups": raw_cache_dir / "matchups.npy",
        "blue_win": raw_cache_dir / "blue_win.npy",
        "meta": raw_cache_dir / "raw_cache_meta.json",
    }


def _split_counts(cfg: SweepConfig) -> dict[str, int]:
    rows = get_client().query(
        f"""
        SELECT split, count()
        FROM {PLAYER_PIVOT_TABLE}
        WHERE split IN ('train', 'validation', 'test')
        GROUP BY split
        """
    )
    available = {str(split): int(count) for split, count in rows.result_rows}
    if cfg.max_games is None:
        return {
            meta_split: available.get(sql_split, 0)
            for sql_split, meta_split in SPLITS.items()
        }

    n_test = round(cfg.max_games * cfg.test_fraction)
    n_val = round(cfg.max_games * cfg.val_fraction)
    n_train = cfg.max_games - n_val - n_test
    return {
        "train": min(n_train, available.get("train", 0)),
        "val": min(n_val, available.get("validation", 0)),
        "test": min(n_test, available.get("test", 0)),
    }


def _raw_arrays(n_games: int, raw_cache_dir: Path) -> dict[str, np.ndarray]:
    paths = _raw_paths(raw_cache_dir)
    shapes = {
        "win_rate": (n_games, N_PLAYER_FEATURES),
        "matchups": (n_games, N_PLAYER_FEATURES),
        "blue_win": (n_games,),
    }
    dtypes = {"win_rate": np.float32, "matchups": np.uint32, "blue_win": np.uint8}
    return {
        name: np.lib.format.open_memmap(
            paths[name],
            mode="w+",
            dtype=dtypes[name],
            shape=shapes[name],
        )
        for name in dtypes
    }


def _raw_row_blocks(split: str, limit: int):
    if limit <= 0:
        return

    query = f"""
    SELECT
        any(p.blue_win) AS blue_win,
        arrayMap(
            x -> tupleElement(x, 2),
            arraySort(
                x -> tupleElement(x, 1),
                groupArray((i.token_idx, i.win_rate))
            )
        ) AS win_rates,
        arrayMap(
            x -> tupleElement(x, 2),
            arraySort(
                x -> tupleElement(x, 1),
                groupArray((i.token_idx, i.matchups))
            )
        ) AS matchup_counts
    FROM {SPLIT_TABLE} AS s
    INNER JOIN {PLAYER_PIVOT_TABLE} AS p
        ON s.matchid = p.matchid
        AND s.split = p.split
    INNER JOIN {INTERACTION_COUNTS_TABLE} AS i
        ON i.matchid = s.matchid
    WHERE s.split = '{split}'
    GROUP BY s.split_index, s.matchid
    ORDER BY s.split_index
    LIMIT {int(limit)}
    """
    with get_client().query_column_block_stream(query) as stream:
        for block in stream:
            if not block or len(block[0]) == 0:
                continue
            blue_win = np.asarray(block[0], dtype=np.uint8)
            win_rate = np.asarray(block[1], dtype=np.float32)
            matchups = np.asarray(block[2], dtype=np.uint32)
            if win_rate.ndim != 2 or win_rate.shape[1] != N_PLAYER_FEATURES:
                raise ValueError(
                    f"Expected win_rate block with shape [n, {N_PLAYER_FEATURES}], "
                    f"got {win_rate.shape}"
                )
            if matchups.ndim != 2 or matchups.shape[1] != N_PLAYER_FEATURES:
                raise ValueError(
                    f"Expected matchup block with shape [n, {N_PLAYER_FEATURES}], "
                    f"got {matchups.shape}"
                )
            yield blue_win, win_rate, matchups


def _write_raw_split(
    arrays: dict[str, np.ndarray],
    *,
    split: str,
    limit: int,
    offset: int,
) -> int:
    written = 0
    for blue_win, win_rate, matchups in _raw_row_blocks(split, limit):
        start = offset + written
        end = start + blue_win.shape[0]
        arrays["blue_win"][start:end] = blue_win
        arrays["win_rate"][start:end] = win_rate
        arrays["matchups"][start:end] = matchups
        written += blue_win.shape[0]
    return written


def _raw_meta_matches(cfg: SweepConfig, counts: dict[str, int]) -> bool:
    paths = _raw_paths(cfg.raw_cache_dir)
    if not all(paths[name].exists() for name in paths):
        return False
    try:
        meta = json.loads(paths["meta"].read_text())
    except json.JSONDecodeError:
        return False

    return (
        meta.get("format") == RAW_CACHE_FORMAT
        and meta.get("splits") == counts
        and meta.get("n_games") == sum(counts.values())
        and meta.get("n_player_features") == N_PLAYER_FEATURES
    )


def _build_raw_cache(cfg: SweepConfig, counts: dict[str, int]) -> None:
    cfg.raw_cache_dir.mkdir(parents=True, exist_ok=True)
    n_games = sum(counts.values())
    arrays = _raw_arrays(n_games, cfg.raw_cache_dir)
    logger.info("Building raw prior cache: games=%d splits=%s", n_games, counts)

    offset = 0
    for sql_split, meta_split in SPLITS.items():
        written = _write_raw_split(
            arrays,
            split=sql_split,
            limit=counts[meta_split],
            offset=offset,
        )
        if written != counts[meta_split]:
            raise RuntimeError(
                f"{meta_split} wrote {written}, expected {counts[meta_split]}"
            )
        offset += written

    for array in arrays.values():
        flush = getattr(array, "flush", None)
        if flush is not None:
            flush()

    paths = _raw_paths(cfg.raw_cache_dir)
    paths["meta"].write_text(
        json.dumps(
            {
                "format": RAW_CACHE_FORMAT,
                "source_table": INTERACTION_COUNTS_TABLE,
                "n_games": n_games,
                "n_player_features": N_PLAYER_FEATURES,
                "splits": counts,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _ensure_raw_cache(cfg: SweepConfig, *, rebuild: bool) -> dict[str, int]:
    counts = _split_counts(cfg)
    if rebuild or not _raw_meta_matches(cfg, counts):
        _build_raw_cache(cfg, counts)
    else:
        logger.info("Using raw prior cache: %s", _project_relative(cfg.raw_cache_dir))
    return counts


def _load_raw_splits(
    cfg: SweepConfig, counts: dict[str, int]
) -> dict[str, RawSplitData]:
    n = sum(counts.values())
    paths = _raw_paths(cfg.raw_cache_dir)
    win_rate = np.load(paths["win_rate"], mmap_mode="r")[:n]
    matchups = np.load(paths["matchups"], mmap_mode="r")[:n]
    blue_win = np.load(paths["blue_win"], mmap_mode="r")[:n]

    def split(start: int, end: int) -> RawSplitData:
        return RawSplitData(
            win_rate=np.asarray(win_rate[start:end], dtype=np.float32),
            matchups=np.asarray(matchups[start:end], dtype=np.float32),
            blue_win=np.asarray(blue_win[start:end], dtype=np.float64),
        )

    splits: dict[str, RawSplitData] = {}
    offset = 0
    for name in SPLIT_NAMES:
        end = offset + counts[name]
        splits[name] = split(offset, end)
        offset = end
    return splits


def _grid_shape(n_tests: int) -> tuple[int, int]:
    if n_tests <= 0:
        raise ValueError("tests must be positive")
    n_means = 1
    for candidate in range(1, int(math.sqrt(n_tests)) + 1):
        if n_tests % candidate == 0:
            n_means = candidate
    return n_means, n_tests // n_means


def _parameter_grid(cfg: SweepConfig) -> list[tuple[float, float]]:
    if not 0.0 <= cfg.prior_mean_min <= cfg.prior_mean_max <= 1.0:
        raise ValueError("prior mean bounds must be within [0.0, 1.0]")
    if cfg.strength_min <= 0.0 or cfg.strength_max < cfg.strength_min:
        raise ValueError("strength bounds must be positive and ordered")

    n_means, n_strengths = _grid_shape(cfg.tests)
    prior_means = np.linspace(cfg.prior_mean_min, cfg.prior_mean_max, n_means)
    strengths = np.geomspace(cfg.strength_min, cfg.strength_max, n_strengths)
    if cfg.strength_min <= DEFAULT_PRIOR_STRENGTH <= cfg.strength_max:
        closest = int(np.argmin(np.abs(np.log(strengths / DEFAULT_PRIOR_STRENGTH))))
        strengths[closest] = DEFAULT_PRIOR_STRENGTH
        strengths.sort()

    return [
        (float(prior_mean), float(strength))
        for prior_mean in prior_means
        for strength in strengths
    ]


def _evaluate(
    model: WinRateLinearModel,
    win_rate: np.ndarray,
    blue_win: np.ndarray,
    *,
    ece_bins: int,
) -> dict[str, float | int]:
    if blue_win.size == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "ece": float("nan"),
            "mse": float("nan"),
        }

    predictions = model.predict(win_rate)
    targets = blue_win.astype(np.float64, copy=False)
    mse = float(np.mean(np.square(predictions - targets)))
    return {
        "n": int(targets.size),
        "accuracy": float(np.mean((predictions >= 0.5) == (targets > 0.5))),
        "ece": expected_calibration_error(predictions, targets, n_bins=ece_bins),
        "mse": mse,
    }


def _run_one(
    splits: dict[str, RawSplitData],
    *,
    prior_mean: float,
    prior_strength: float,
    ece_bins: int,
) -> dict[str, Any]:
    train_win_rate = bayesian_smoothed_rate(
        splits["train"].win_rate,
        splits["train"].matchups,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )
    model = fit_linear_regression(train_win_rate, splits["train"].blue_win)
    result: dict[str, Any] = {
        "prior_mean": prior_mean,
        "prior_strength": prior_strength,
        "intercept": model.intercept,
    }
    for split_name, split in splits.items():
        win_rate = (
            train_win_rate
            if split_name == "train"
            else bayesian_smoothed_rate(
                split.win_rate,
                split.matchups,
                prior_mean=prior_mean,
                prior_strength=prior_strength,
            )
        )
        metrics = _evaluate(
            model,
            win_rate,
            split.blue_win,
            ece_bins=ece_bins,
        )
        for metric_name, value in metrics.items():
            result[f"{split_name}_{metric_name}"] = value
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return _project_relative(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _ranked_results(
    results: list[dict[str, Any]],
    metric: str,
    *,
    maximize: bool,
    limit: int = 10,
) -> list[dict[str, Any]]:
    finite = [
        item
        for item in results
        if isinstance(item.get(metric), (int, float)) and math.isfinite(item[metric])
    ]
    return sorted(finite, key=lambda item: item[metric], reverse=maximize)[:limit]


def _summarize(
    cfg: SweepConfig,
    counts: dict[str, int],
    results: list[dict[str, Any]],
    duration_seconds: float,
) -> dict[str, Any]:
    rankings = {
        metric: _ranked_results(results, metric, maximize=maximize)
        for metric, maximize in RANKED_METRICS.items()
    }
    return {
        "config": asdict(cfg),
        "splits": counts,
        "duration_seconds": duration_seconds,
        "n_results": len(results),
        "best": {
            metric: ranked[0] if ranked else None for metric, ranked in rankings.items()
        },
        "top": rankings,
    }


def _write_outputs(
    *,
    cfg: SweepConfig,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_path.write_text(
        json.dumps(
            _json_value(
                {
                    **summary,
                    "results": results,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "prior_mean",
        "prior_strength",
        "intercept",
        *(f"{split}_{metric}" for split in SPLIT_NAMES for metric in METRICS),
    ]
    cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name) for name in fieldnames})


def sweep(
    cfg: SweepConfig | None = None, *, rebuild_raw_cache: bool = False
) -> dict[str, Any]:
    cfg = cfg or SweepConfig()
    counts = _ensure_raw_cache(cfg, rebuild=rebuild_raw_cache)
    splits = _load_raw_splits(cfg, counts)
    grid = _parameter_grid(cfg)
    logger.info("Running Bayesian prior sweep: tests=%d", len(grid))

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for idx, (prior_mean, prior_strength) in enumerate(grid, start=1):
        result = _run_one(
            splits,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            ece_bins=cfg.ece_bins,
        )
        results.append(result)
        if idx == 1 or idx % 25 == 0 or idx == len(grid):
            logger.info(
                "sweep %d/%d prior=%.4f strength=%.4f val_acc=%.6f val_ece=%.6f",
                idx,
                len(grid),
                prior_mean,
                prior_strength,
                result["val_accuracy"],
                result["val_ece"],
            )

    duration_seconds = time.perf_counter() - started
    summary = _summarize(cfg, counts, results, duration_seconds)
    _write_outputs(cfg=cfg, summary=summary, results=results)
    logger.info("Wrote report: %s", _project_relative(cfg.report_path))
    logger.info("Wrote CSV: %s", _project_relative(cfg.csv_path))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Bayesian prior mean and strength for ML win-rate priors."
    )
    parser.add_argument("--tests", type=int, default=1000)
    parser.add_argument("--prior-mean-min", type=float, default=0.30)
    parser.add_argument("--prior-mean-max", type=float, default=0.50)
    parser.add_argument("--strength-min", type=float, default=1.0)
    parser.add_argument("--strength-max", type=float, default=1000.0)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--max-games", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = SweepConfig(
        tests=args.tests,
        prior_mean_min=args.prior_mean_min,
        prior_mean_max=args.prior_mean_max,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
        ece_bins=args.ece_bins,
        max_games=args.max_games,
    )
    sweep(cfg)


if __name__ == "__main__":
    main()
