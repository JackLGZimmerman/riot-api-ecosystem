"""Hyper-parameter sweep harness.

Each run is a full `train()` call into its own subdirectory under
`app/ml/data/checkpoints/<phase>/<run_name>/`, with TensorBoard logs under
`app/ml/data/tensorboard/sweep/<phase>/<run_name>/`. Results are appended to
`app/ml/data/checkpoints/<phase>/sweep_summary.jsonl`. Existing run dirs are
skipped so a sweep can resume after interruption.

Invoke phases directly:
    python -m app.ml.sweep baseline
    python -m app.ml.sweep phase_a
    python -m app.ml.sweep phase_b --configs path/to/configs.json
    python -m app.ml.sweep final --name winner --overrides '{"lr": 1e-4, ...}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from app.core.config.settings import PROJECT_ROOT
from app.core.logging.logger import setup_logging_config
from app.ml.config import CHECKPOINT_DIR, ML_DATA_DIR, TrainConfig
from app.ml.train import train

setup_logging_config()
logger = logging.getLogger(__name__)

SWEEP_ROOT = CHECKPOINT_DIR
TB_SWEEP_ROOT = (ML_DATA_DIR / "tensorboard" / "sweep").resolve()

# Sweep defaults: aggressive early stopping, skip per-run test.
SWEEP_DEFAULTS: dict[str, Any] = {
    "epochs": 1000,
    "early_stop_patience": 10,
    "run_final_test": False,
}


def _phase_dir(phase: str) -> Path:
    return SWEEP_ROOT / phase


def _run_dir(phase: str, run_name: str) -> Path:
    return _phase_dir(phase) / run_name


def _parse_metrics(metrics_path: Path) -> dict[str, Any]:
    """Extract best val_loss + last epoch_end + early_stop event."""
    best_val_loss = float("inf")
    best_epoch_fields: dict[str, Any] = {}
    last_epoch = 0
    last_step = 0
    early_stop: dict[str, Any] | None = None
    test_metrics: dict[str, Any] | None = None
    with metrics_path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = row.get("event")
            if event == "epoch_end":
                last_epoch = int(row.get("epoch", last_epoch))
                last_step = int(row.get("step", last_step))
                v = row.get("val_loss")
                if isinstance(v, (int, float)) and v < best_val_loss:
                    best_val_loss = float(v)
                    best_epoch_fields = {
                        "best_epoch": last_epoch,
                        "best_val_loss": float(v),
                        "best_val_accuracy": row.get("val_accuracy"),
                        "best_val_auc": row.get("val_auc"),
                        "best_val_brier": row.get("val_brier"),
                        "best_val_ece": row.get("val_ece"),
                    }
            elif event == "early_stop":
                early_stop = {
                    "early_stop_epoch": row.get("epoch"),
                    "early_stop_patience": row.get("patience"),
                }
            elif event == "test":
                test_metrics = {
                    "test_loss": row.get("test_loss"),
                    "test_accuracy": row.get("test_accuracy"),
                    "test_auc": row.get("test_auc"),
                    "test_brier": row.get("test_brier"),
                    "test_ece": row.get("test_ece"),
                }
    summary = {
        "last_epoch": last_epoch,
        "last_step": last_step,
        **best_epoch_fields,
    }
    if early_stop is not None:
        summary.update(early_stop)
    if test_metrics is not None:
        summary.update(test_metrics)
    return summary


def _project_relative(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _make_train_cfg(
    base: TrainConfig,
    run_dir: Path,
    tb_dir: Path,
    overrides: dict[str, Any],
) -> TrainConfig:
    merged = {**SWEEP_DEFAULTS, **overrides}
    merged.setdefault("checkpoint_dir", run_dir)
    merged.setdefault("metrics_dir", run_dir)
    # `tensorboard_dir` is interpreted as absolute when it starts with `/`; this
    # places the run under the sweep TB root rather than the live training root.
    merged.setdefault("tensorboard_dir", str(tb_dir.parent))
    merged.setdefault("tensorboard_run_name", tb_dir.name)
    return replace(base, **merged)


def run_one(
    phase: str,
    run_name: str,
    overrides: dict[str, Any],
    *,
    base: TrainConfig | None = None,
    summary_path: Path | None = None,
    skip_if_done: bool = True,
) -> dict[str, Any]:
    base = base or TrainConfig()
    run_dir = _run_dir(phase, run_name)
    tb_dir = TB_SWEEP_ROOT / phase / run_name
    metrics_path = run_dir / TrainConfig.metrics_file
    if skip_if_done and metrics_path.exists():
        existing = _parse_metrics(metrics_path)
        if existing.get("best_val_loss") is not None:
            logger.info(
                "Skipping %s/%s (already done, best_val_loss=%.4e)",
                phase,
                run_name,
                existing["best_val_loss"],
            )
            return {
                "phase": phase,
                "run_name": run_name,
                "overrides": overrides,
                "status": "skipped",
                **existing,
            }
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = _make_train_cfg(base, run_dir, tb_dir, overrides)

    logger.info("=" * 80)
    logger.info("SWEEP %s/%s overrides=%s", phase, run_name, overrides)
    logger.info("  checkpoint_dir=%s", _project_relative(run_dir))
    logger.info("  tensorboard=%s", _project_relative(tb_dir))
    logger.info("=" * 80)

    t0 = time.perf_counter()
    status = "ok"
    err = None
    try:
        train(train_cfg=train_cfg)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        err = repr(exc)
        logger.exception("Run %s/%s failed", phase, run_name)
    elapsed = time.perf_counter() - t0

    summary: dict[str, Any]
    if metrics_path.exists():
        summary = _parse_metrics(metrics_path)
    else:
        summary = {"last_epoch": 0, "last_step": 0}

    result = {
        "phase": phase,
        "run_name": run_name,
        "overrides": overrides,
        "status": status,
        "error": err,
        "wall_s": round(elapsed, 1),
        **summary,
    }
    summary_path = summary_path or (_phase_dir(phase) / "sweep_summary.jsonl")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a") as fh:
        fh.write(json.dumps(result, default=str) + "\n")

    bvl = result.get("best_val_loss")
    bvl_str = f"{bvl:.4e}" if isinstance(bvl, (int, float)) else "n/a"
    logger.info(
        "DONE %s/%s in %.1fs | best_val_loss=%s @ ep %s | last_epoch=%s status=%s",
        phase,
        run_name,
        elapsed,
        bvl_str,
        result.get("best_epoch"),
        result.get("last_epoch"),
        status,
    )
    return result


def run_phase(
    phase: str,
    configs: list[tuple[str, dict[str, Any]]],
    base: TrainConfig | None = None,
) -> list[dict[str, Any]]:
    summary_path = _phase_dir(phase) / "sweep_summary.jsonl"
    results: list[dict[str, Any]] = []
    total = len(configs)
    for i, (name, overrides) in enumerate(configs, 1):
        logger.info("---- phase %s run %d/%d: %s ----", phase, i, total, name)
        result = run_one(
            phase, name, overrides, base=base, summary_path=summary_path
        )
        results.append(result)
        _print_leaderboard(phase, results)
    return results


def _print_leaderboard(phase: str, results: list[dict[str, Any]]) -> None:
    ranked = sorted(
        (r for r in results if isinstance(r.get("best_val_loss"), (int, float))),
        key=lambda r: r["best_val_loss"],
    )
    logger.info("---- %s leaderboard (top 10 by val_loss) ----", phase)
    for r in ranked[:10]:
        logger.info(
            "  %-40s val_loss=%.5f acc=%.4f auc=%.4f ep=%s wall=%.0fs %s",
            r["run_name"],
            r["best_val_loss"],
            r.get("best_val_accuracy") or float("nan"),
            r.get("best_val_auc") or float("nan"),
            r.get("best_epoch"),
            r.get("wall_s", 0.0),
            r.get("overrides"),
        )


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------


def baseline_config() -> tuple[str, dict[str, Any]]:
    """Current production defaults, frozen for comparison."""
    return ("baseline", {})


def phase_a_configs() -> list[tuple[str, dict[str, Any]]]:
    """One-axis sweeps around live `TrainConfig` defaults."""
    configs: list[tuple[str, dict[str, Any]]] = []

    for lr in (2e-5, 1e-4, 2e-4):
        configs.append((f"lr_{lr:.0e}", {"lr": lr}))

    for ws in (50, 250, 500):
        configs.append((f"warmup_{ws}", {"warmup_steps": ws}))

    for ce in (10, 40, 80):
        configs.append((f"center_{ce}", {"lr_center_epoch": ce}))

    for sh in (2.0, 8.0):
        configs.append((f"sharp_{sh}", {"lr_sharpness": sh}))

    for ts in (0.1, 1.0, 3.0):
        configs.append((f"tail_{ts}", {"lr_tail_strength": ts}))

    for emr in (0.001, 0.1):
        configs.append((f"eta_min_{emr}", {"lr_eta_min_ratio": emr}))

    for wd in (1e-3, 2e-2):
        configs.append((f"wd_{wd:.0e}", {"weight_decay": wd}))

    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("baseline")
    sub.add_parser("phase_a")

    p_b = sub.add_parser("phase_b")
    p_b.add_argument("--configs", required=True, type=Path)

    p_g = sub.add_parser("sweep")
    p_g.add_argument("--phase", required=True, help="Phase directory name")
    p_g.add_argument("--configs", required=True, type=Path)

    p_f = sub.add_parser("final")
    p_f.add_argument("--name", required=True)
    p_f.add_argument("--overrides", required=True, help="JSON dict of overrides")
    p_f.add_argument(
        "--epochs", type=int, default=500, help="Training cap for final run"
    )
    p_f.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stop patience for final run (0 disables)",
    )

    p_one = sub.add_parser("one")
    p_one.add_argument("--phase", required=True)
    p_one.add_argument("--name", required=True)
    p_one.add_argument("--overrides", required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd
    if cmd == "baseline":
        run_phase("baseline", [baseline_config()])
    elif cmd == "phase_a":
        run_phase("phase_a", phase_a_configs())
    elif cmd == "phase_b":
        configs_data = json.loads(args.configs.read_text())
        configs = [(c["name"], c["overrides"]) for c in configs_data]
        run_phase("phase_b", configs)
    elif cmd == "sweep":
        configs_data = json.loads(args.configs.read_text())
        configs = [(c["name"], c["overrides"]) for c in configs_data]
        run_phase(args.phase, configs)
    elif cmd == "final":
        overrides = json.loads(args.overrides)
        overrides.update(
            {
                "epochs": args.epochs,
                "early_stop_patience": args.patience,
                "run_final_test": True,
            }
        )
        run_one("final", args.name, overrides, skip_if_done=False)
    elif cmd == "one":
        overrides = json.loads(args.overrides)
        run_one(args.phase, args.name, overrides, skip_if_done=False)
    else:
        raise ValueError(f"unknown cmd {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
