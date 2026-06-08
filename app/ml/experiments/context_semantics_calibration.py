"""Contextual semantic calibration experiment runner.

This runner keeps the post-audit experiments reproducible: train on the combined
group+context EB target, checkpoint on support-qualified context tails, then emit
the raw and group audits needed to compare candidates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_EXPERIMENT_ROOT = Path("app/ml/data/experiments/context_semantics_calibration")
DEFAULT_CACHE_DIR = Path("app/ml/data/cache")
DEFAULT_PRODUCTION_MODEL_PATH = Path("app/ml/data/hgnn_production_model.pt")
DEFAULT_ENCODER_SIDECAR_PATH = Path(
    "app/ml/data/experiments/semantic_identity_sidecar_compact.npz"
)
DEFAULT_BOOTSTRAP_SEED = 20260604


@dataclass(frozen=True)
class CalibrationSpec:
    name: str
    description: str
    target: str = "group_context_eb"
    checkpoint_metric: str = "val_group_context_high_support_tail"
    residual_loss: str = "mse"
    loss_weight: float = 3.0
    context_residual_shrink_strength: float | None = None
    context_residual_clip: float | None = None
    holdout_mode: str = "none"
    holdout_fold: int = 0
    diagnostic_only: bool = False


SPECS: tuple[CalibrationSpec, ...] = (
    CalibrationSpec(
        name="group_context_tail",
        description=(
            "Primary combined group+context EB residual run, selected by "
            "group EB MSE plus high-support context max-tail squared."
        ),
    ),
    CalibrationSpec(
        name="group_context_tail_uncert_huber",
        description=(
            "Same target/checkpoint as primary, but ignores residual gaps inside "
            "the EB uncertainty band before Huber penalizing the excess."
        ),
        residual_loss="uncertainty_huber",
    ),
    CalibrationSpec(
        name="group_context_tail_uncert_huber_relaxed_context",
        description=(
            "Huber residual run with a less conservative context residual teacher "
            "to test whether target amplitude, not representation, is limiting "
            "high-support contextual semantics."
        ),
        residual_loss="uncertainty_huber",
        context_residual_shrink_strength=15_000.0,
        context_residual_clip=0.08,
    ),
    CalibrationSpec(
        name="group_context_tail_holdout_even",
        description=(
            "Diagnostic primary run with even-index group specs held out of the "
            "calibration loss."
        ),
        holdout_mode="even_odd",
        holdout_fold=0,
        diagnostic_only=True,
    ),
    CalibrationSpec(
        name="group_context_tail_holdout_odd",
        description=(
            "Diagnostic primary run with odd-index group specs held out of the "
            "calibration loss."
        ),
        holdout_mode="even_odd",
        holdout_fold=1,
        diagnostic_only=True,
    ),
)
SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


def run_dir(
    spec: CalibrationSpec,
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    seed: int,
) -> Path:
    return experiment_root / spec.name / f"seed{seed}"


def model_path(
    spec: CalibrationSpec,
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    seed: int,
) -> Path:
    return run_dir(spec, experiment_root=experiment_root, seed=seed) / "model.pt"


def metrics_path(
    spec: CalibrationSpec,
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    seed: int,
) -> Path:
    return run_dir(spec, experiment_root=experiment_root, seed=seed) / "metrics.json"


def prediction_cache_path(
    spec: CalibrationSpec,
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    seed: int,
) -> Path:
    return run_dir(spec, experiment_root=experiment_root, seed=seed) / "audit_focus.npy"


def context_audit_path(
    spec: CalibrationSpec,
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    seed: int,
    audit_split: str,
    suffix: str,
) -> Path:
    return (
        run_dir(spec, experiment_root=experiment_root, seed=seed)
        / f"{audit_split}_context_examples_audit.{suffix}"
    )


def group_audit_path(
    spec: CalibrationSpec,
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    seed: int,
) -> Path:
    return (
        run_dir(spec, experiment_root=experiment_root, seed=seed)
        / "group_context_audit.json"
    )


def train_command(
    spec: CalibrationSpec,
    *,
    seed: int,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    production_model_path: Path = DEFAULT_PRODUCTION_MODEL_PATH,
    encoder_sidecar_path: Path = DEFAULT_ENCODER_SIDECAR_PATH,
    metric_min_count: int = 2048,
    max_epochs: int = 40,
    patience: int = 5,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "app.ml.train",
        "--cache-dir",
        str(cache_dir),
        "--encoder-sidecar-path",
        str(encoder_sidecar_path),
        "--model-path",
        str(model_path(spec, experiment_root=experiment_root, seed=seed)),
        "--metrics-path",
        str(metrics_path(spec, experiment_root=experiment_root, seed=seed)),
        "--audit-prediction-cache-path",
        str(prediction_cache_path(spec, experiment_root=experiment_root, seed=seed)),
        "--warm-start-model-path",
        str(production_model_path),
        "--batch-size",
        "16384",
        "--train-batch-cap",
        "40960",
        "--max-epochs",
        str(max_epochs),
        "--patience",
        str(patience),
        "--learning-rate",
        "1e-4",
        "--seed",
        str(seed),
        "--checkpoint-metric",
        spec.checkpoint_metric,
        "--use-learned-semantic-moe",
        "--use-semantic-group-features",
        "--semantic-moe-architecture",
        "convex_encoder_mix",
        "--semantic-moe-num-experts",
        "128",
        "--semantic-moe-top-k",
        "32",
        "--semantic-context-calibration-target",
        spec.target,
        "--semantic-context-calibration-objective",
        "residual",
        "--semantic-context-calibration-loss-weight",
        str(spec.loss_weight),
        "--semantic-context-calibration-min-count",
        "8",
        "--semantic-context-calibration-tail-weight",
        "2.0",
        "--semantic-context-calibration-group-surface",
        "train_core",
        "--semantic-context-calibration-bin-weighting",
        "support_family",
        "--semantic-context-calibration-residual-loss",
        spec.residual_loss,
        "--semantic-context-calibration-holdout-mode",
        spec.holdout_mode,
        "--semantic-context-calibration-holdout-fold",
        str(spec.holdout_fold),
        "--semantic-context-metric-min-count",
        str(metric_min_count),
    ]
    if spec.context_residual_shrink_strength is not None:
        command.extend(
            [
                "--semantic-context-calibration-context-residual-shrink-strength",
                str(spec.context_residual_shrink_strength),
            ]
        )
    if spec.context_residual_clip is not None:
        command.extend(
            [
                "--semantic-context-calibration-context-residual-clip",
                str(spec.context_residual_clip),
            ]
        )
    return command


def context_audit_command(
    spec: CalibrationSpec,
    *,
    seed: int,
    audit_split: str,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    encoder_sidecar_path: Path = DEFAULT_ENCODER_SIDECAR_PATH,
    bootstrap_samples: int = 0,
    refresh_predictions: bool = False,
) -> list[str]:
    cli_split = "val" if audit_split == "validation" else audit_split
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "app.ml.context_examples_audit",
        "--context-cache-dir",
        str(cache_dir),
        "--model-cache-dir",
        str(cache_dir),
        "--model-path",
        str(model_path(spec, experiment_root=experiment_root, seed=seed)),
        "--encoder-sidecar-path",
        str(encoder_sidecar_path),
        "--prediction-cache",
        str(prediction_cache_path(spec, experiment_root=experiment_root, seed=seed)),
        "--audit-split",
        cli_split,
        "--output",
        str(
            context_audit_path(
                spec,
                experiment_root=experiment_root,
                seed=seed,
                audit_split=audit_split,
                suffix="md",
            )
        ),
        "--json-output",
        str(
            context_audit_path(
                spec,
                experiment_root=experiment_root,
                seed=seed,
                audit_split=audit_split,
                suffix="json",
            )
        ),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--bootstrap-seed",
        str(DEFAULT_BOOTSTRAP_SEED),
    ]
    if refresh_predictions:
        command.append("--refresh-predictions")
    return command


def group_audit_command(
    spec: CalibrationSpec,
    *,
    seed: int,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "app.ml.group_context_audit",
        "--context-cache-dir",
        str(cache_dir),
        "--prediction-cache",
        str(prediction_cache_path(spec, experiment_root=experiment_root, seed=seed)),
        "--per-row",
        "--json-output",
        str(group_audit_path(spec, experiment_root=experiment_root, seed=seed)),
    ]


def selected_specs(names: Sequence[str]) -> list[CalibrationSpec]:
    if not names or names == ["all"]:
        return list(SPECS)
    out: list[CalibrationSpec] = []
    for name in names:
        try:
            out.append(SPEC_BY_NAME[name])
        except KeyError as exc:
            known = ", ".join(sorted(SPEC_BY_NAME))
            raise ValueError(f"unknown run {name!r}; known: {known}") from exc
    return out


def spec_payload(spec: CalibrationSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "description": spec.description,
        "target": spec.target,
        "checkpoint_metric": spec.checkpoint_metric,
        "residual_loss": spec.residual_loss,
        "loss_weight": spec.loss_weight,
        "context_residual_shrink_strength": spec.context_residual_shrink_strength,
        "context_residual_clip": spec.context_residual_clip,
        "holdout_mode": spec.holdout_mode,
        "holdout_fold": spec.holdout_fold,
        "diagnostic_only": spec.diagnostic_only,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def _dig(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_runs(
    specs: Sequence[CalibrationSpec],
    *,
    seeds: Sequence[int],
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        for seed in seeds:
            metrics = _load_json(
                metrics_path(spec, experiment_root=experiment_root, seed=seed)
            )
            if metrics is None:
                rows.append(
                    {
                        "name": spec.name,
                        "seed": seed,
                        "status": "missing_metrics",
                        "metrics_path": str(
                            metrics_path(
                                spec, experiment_root=experiment_root, seed=seed
                            )
                        ),
                    }
                )
                continue
            metadata = metrics.get("semantic_context_calibration_metadata", {})
            row: dict[str, object] = {
                "name": spec.name,
                "seed": seed,
                "status": "complete",
                "diagnostic_only": spec.diagnostic_only,
                "best_epoch": metrics.get("best_epoch"),
                "best_checkpoint_score": _safe_float(
                    metrics.get("best_checkpoint_score")
                ),
                "target": _dig(metadata, "target"),
                "context_spec_count": _dig(metadata, "context_spec_count"),
                "group_spec_count": _dig(metadata, "group_spec_count"),
            }
            for split_name in ("val", "test"):
                split = metrics.get(split_name, {})
                row.update(
                    {
                        f"{split_name}_accuracy": _safe_float(_dig(split, "accuracy")),
                        f"{split_name}_nll": _safe_float(_dig(split, "nll")),
                        f"{split_name}_context_max_abs_gap": _safe_float(
                            _dig(split, "context_max_abs_gap")
                        ),
                        f"{split_name}_context_high_support_max_abs_gap": _safe_float(
                            _dig(split, "context_high_support_max_abs_gap")
                        ),
                        f"{split_name}_context_support_weighted_gap_mse": _safe_float(
                            _dig(split, "context_support_weighted_gap_mse")
                        ),
                        f"{split_name}_group_eb_gap_mse": _safe_float(
                            _dig(split, "group_eb_gap_mse")
                        ),
                        f"{split_name}_group_eb_max_abs_gap": _safe_float(
                            _dig(split, "group_eb_max_abs_gap")
                        ),
                        f"{split_name}_group_systematic_gap_mse": _safe_float(
                            _dig(split, "group_systematic_gap_mse")
                        ),
                    }
                )
            rows.append(row)
    return {
        "experiment_root": str(experiment_root),
        "rows": rows,
    }


def _add_common_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("runs", nargs="*", default=["all"])
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)


def _run_command(command: Sequence[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    print(printable)
    if not dry_run:
        subprocess.run(command, check=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List calibration specs.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine JSON.")

    commands_parser = subparsers.add_parser("commands", help="Print commands.")
    _add_common_selection_args(commands_parser)
    commands_parser.add_argument(
        "--steps",
        nargs="+",
        choices=("train", "audit-validation", "audit-all", "group-audit"),
        default=("train", "audit-validation", "group-audit"),
    )
    commands_parser.add_argument("--seed", type=int, default=4)
    commands_parser.add_argument("--metric-min-count", type=int, default=2048)
    commands_parser.add_argument("--bootstrap-samples", type=int, default=0)
    commands_parser.add_argument("--refresh-predictions", action="store_true")
    commands_parser.add_argument("--max-epochs", type=int, default=40)
    commands_parser.add_argument("--patience", type=int, default=5)

    run_parser = subparsers.add_parser("run", help="Execute selected commands.")
    _add_common_selection_args(run_parser)
    run_parser.add_argument(
        "--steps",
        nargs="+",
        choices=("train", "audit-validation", "audit-all", "group-audit"),
        default=("train", "audit-validation", "group-audit"),
    )
    run_parser.add_argument("--seed", type=int, default=4)
    run_parser.add_argument("--metric-min-count", type=int, default=2048)
    run_parser.add_argument("--bootstrap-samples", type=int, default=0)
    run_parser.add_argument("--refresh-predictions", action="store_true")
    run_parser.add_argument("--max-epochs", type=int, default=40)
    run_parser.add_argument("--patience", type=int, default=5)
    run_parser.add_argument("--dry-run", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="Compare completed runs.")
    _add_common_selection_args(compare_parser)
    compare_parser.add_argument("--seeds", nargs="+", type=int, default=[4])
    compare_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "comparison.json",
    )

    return parser.parse_args(argv)


def _commands_for_steps(
    spec: CalibrationSpec,
    *,
    steps: Sequence[str],
    seed: int,
    experiment_root: Path,
    metric_min_count: int,
    bootstrap_samples: int,
    refresh_predictions: bool,
    max_epochs: int,
    patience: int,
) -> list[list[str]]:
    commands: list[list[str]] = []
    if "train" in steps:
        commands.append(
            train_command(
                spec,
                seed=seed,
                experiment_root=experiment_root,
                metric_min_count=metric_min_count,
                max_epochs=max_epochs,
                patience=patience,
            )
        )
    if "audit-validation" in steps:
        commands.append(
            context_audit_command(
                spec,
                seed=seed,
                audit_split="validation",
                experiment_root=experiment_root,
                bootstrap_samples=bootstrap_samples,
                refresh_predictions=refresh_predictions,
            )
        )
    if "audit-all" in steps:
        commands.append(
            context_audit_command(
                spec,
                seed=seed,
                audit_split="all",
                experiment_root=experiment_root,
                bootstrap_samples=bootstrap_samples,
                refresh_predictions=refresh_predictions,
            )
        )
    if "group-audit" in steps:
        commands.append(
            group_audit_command(spec, seed=seed, experiment_root=experiment_root)
        )
    return commands


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "list":
        payload = [spec_payload(spec) for spec in SPECS]
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for row in payload:
                marker = "diagnostic" if row["diagnostic_only"] else "candidate"
                print(f"{row['name']}\t{marker}\t{row['description']}")
        return 0

    if args.command in {"commands", "run"}:
        dry_run = args.command == "commands" or bool(args.dry_run)
        for spec in selected_specs(args.runs):
            for command in _commands_for_steps(
                spec,
                steps=args.steps,
                seed=int(args.seed),
                experiment_root=args.experiment_root,
                metric_min_count=int(args.metric_min_count),
                bootstrap_samples=int(args.bootstrap_samples),
                refresh_predictions=bool(args.refresh_predictions),
                max_epochs=int(args.max_epochs),
                patience=int(args.patience),
            ):
                _run_command(command, dry_run=dry_run)
        return 0

    if args.command == "compare":
        payload = compare_runs(
            selected_specs(args.runs),
            seeds=args.seeds,
            experiment_root=args.experiment_root,
        )
        _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
