"""Optimizer × calibration grid sweep on the current token model.

Sweeps three throughput-neutral axes: AdamW β2, label-squishing target range,
and warmup steps. All model and other training settings stay at defaults.

Run the full sweep:
    CLICKHOUSE_HOST=localhost python -m app.ml.run_grid_sweep

Run a single trial in-process (used by the parent for subprocess dispatch):
    python -m app.ml.run_grid_sweep --trial 7 --root <sweep_root>
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.core.logging.logger import setup_logging_config
from app.ml.config import CHECKPOINT_DIR, DatasetConfig, ModelConfig, TrainConfig
from app.ml.train import train

setup_logging_config()
logger = logging.getLogger(__name__)

SWEEP_NAME = "optimizer_sweep"
EPOCHS = 50

# AdamW β2: controls gradient variance tracking. β1=0.9 held fixed.
BETA2: tuple[float, ...] = (0.990, 0.999, 0.9999)

# Symmetric label-squishing range: (target_min, target_max).
TARGET: tuple[tuple[float, float], ...] = (
    (0.05, 0.95),
    (0.10, 0.90),
    (0.15, 0.85),
    (0.20, 0.80),
    (0.25, 0.75),
)

# LR warmup duration in steps. At batch_size=16384 ~90 steps/epoch.
WARMUP: tuple[int, ...] = (50, 125, 250)

SUMMARY_FIELDS: tuple[str, ...] = (
    "epoch",
    "val_loss",
    "val_accuracy",
    "val_brier",
    "val_ece",
    "train_monitor_loss",
    "train_monitor_accuracy",
    "train_monitor_brier",
    "train_monitor_ece",
)

SUMMARY_HEADER = (
    "%-52s %4s %10s %8s %10s %10s %10s %8s %10s %10s",
    "trial",
    "ep",
    "val_loss",
    "val_acc",
    "val_brier",
    "val_ece",
    "test_loss",
    "test_acc",
    "test_brier",
    "test_ece",
)

Trial = tuple[int, float, tuple[float, float], int]
MetricsRow = dict[str, Any]


def _trials() -> list[Trial]:
    return [
        (idx, beta2, target, warmup)
        for idx, (beta2, target, warmup) in enumerate(
            itertools.product(BETA2, TARGET, WARMUP)
        )
    ]


def _trial_dir(idx: int, beta2: float, target: tuple[float, float], warmup: int) -> str:
    return f"t{idx:02d}_b2{beta2:.4f}_tm{target[0]:.2f}_ws{warmup}"


def _make_root() -> Path:
    return CHECKPOINT_DIR / f"{SWEEP_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"


def _trial_cmd(idx: int, root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "app.ml.run_grid_sweep",
        "--trial",
        str(idx),
        "--root",
        str(root),
    ]


def _make_train_config(
    beta2: float,
    target: tuple[float, float],
    warmup: int,
    trial_dir: Path,
) -> TrainConfig:
    return replace(
        TrainConfig(),
        epochs=EPOCHS,
        adamw_betas=(0.9, beta2),
        target_min=target[0],
        target_max=target[1],
        warmup_steps=warmup,
        checkpoint_dir=trial_dir,
        metrics_dir=trial_dir,
        tensorboard_dir="tb",
        attention_diagnostics_interval=0,
    )


def run_trial(
    idx: int,
    beta2: float,
    target: tuple[float, float],
    warmup: int,
    root: Path,
) -> Path:
    trial_dir = root / _trial_dir(idx, beta2, target, warmup)
    trial_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "trial %d -> %s | beta2=%.4f target=(%.2f,%.2f) warmup=%d",
        idx,
        trial_dir,
        beta2,
        target[0],
        target[1],
        warmup,
    )

    train(DatasetConfig(), ModelConfig(), _make_train_config(beta2, target, warmup, trial_dir))
    return trial_dir


def _dispatch_trial(
    idx: int,
    beta2: float,
    target: tuple[float, float],
    warmup: int,
    root: Path,
) -> int:
    del beta2, target, warmup  # Trial metadata is kept in the signature for call-site symmetry.

    cmd = _trial_cmd(idx, root)
    logger.info("dispatch trial %d: %s", idx, " ".join(cmd))

    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)

    logger.info("trial %d exit=%d in %.1fs", idx, result.returncode, time.perf_counter() - t0)
    return result.returncode


def _safe_json_loads(line: str) -> MetricsRow | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _best_epoch(current_best: MetricsRow | None, row: MetricsRow) -> MetricsRow | None:
    val_loss = row.get("val_loss")
    if val_loss is None:
        return current_best

    if current_best is None or float(val_loss) < float(current_best["val_loss"]):
        return {field: row.get(field) for field in SUMMARY_FIELDS}

    return current_best


def _read_summary(trial_dir: Path) -> dict[str, MetricsRow] | None:
    path = trial_dir / "metrics.jsonl"
    if not path.exists():
        return None

    best: MetricsRow | None = None
    test: MetricsRow | None = None

    with path.open() as fh:
        for row in filter(None, map(_safe_json_loads, fh)):
            match row.get("event"):
                case "epoch_end":
                    best = _best_epoch(best, row)
                case "test":
                    test = row

    if best is None:
        return None

    return {"best_val": best, "final_test": test or {}}


def _fmt(value: object, fmt: str = ".6f") -> str:
    if value is None:
        return "nan"

    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "nan"


def _summary_rows(root: Path) -> list[tuple[str, dict[str, MetricsRow]]]:
    rows = [
        (trial_dir.name, summary)
        for trial_dir in sorted(root.iterdir())
        if trial_dir.is_dir() and (summary := _read_summary(trial_dir)) is not None
    ]
    return sorted(rows, key=lambda row: float(row[1]["best_val"]["val_loss"]))


def _log_summary_row(name: str, summary: dict[str, MetricsRow]) -> None:
    best_val = summary["best_val"]
    final_test = summary["final_test"]

    logger.info(
        SUMMARY_HEADER[0],
        name,
        str(best_val.get("epoch", "")),
        _fmt(best_val.get("val_loss")),
        _fmt(best_val.get("val_accuracy"), ".4f"),
        _fmt(best_val.get("val_brier")),
        _fmt(best_val.get("val_ece")),
        _fmt(final_test.get("test_loss")),
        _fmt(final_test.get("test_accuracy"), ".4f"),
        _fmt(final_test.get("test_brier")),
        _fmt(final_test.get("test_ece")),
    )


def print_summary(root: Path) -> None:
    rows = _summary_rows(root)

    logger.info("sweep summary (sorted by best val_loss):")
    logger.info(*SUMMARY_HEADER)

    for name, summary in rows:
        _log_summary_row(name, summary)

    summary_path = root / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(
            [{"trial": name, **summary} for name, summary in rows],
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    logger.info("wrote %s", summary_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=None)
    parser.add_argument("--root", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    all_trials = _trials()

    if args.trial is not None:
        if args.root is None:
            raise SystemExit("--trial requires --root")

        run_trial(*all_trials[args.trial], Path(args.root))
        return

    root = _make_root()
    root.mkdir(parents=True, exist_ok=True)

    logger.info("sweep root: %s | %d trials | epochs=%d", root, len(all_trials), EPOCHS)

    t0 = time.perf_counter()
    for trial in all_trials:
        _dispatch_trial(*trial, root)

    logger.info("sweep wall time: %.1fs", time.perf_counter() - t0)
    print_summary(root)


if __name__ == "__main__":
    main()
