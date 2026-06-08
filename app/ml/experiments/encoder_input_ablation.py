"""Semantic encoder input ablation runner.

This module keeps the semantic-group calibration experiment matrix reproducible
without baking another one-off shell script into the docs.  It intentionally
does not change the HGNN architecture; each run swaps the encoder sidecar input
recipe, then trains and audits with the current production semantic MoE recipe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_EXPERIMENT_ROOT = Path("app/ml/data/experiments/encoder_input_ablation")
DEFAULT_CACHE_DIR = Path("app/ml/data/cache")
DEFAULT_PRODUCTION_MODEL_PATH = Path("app/ml/data/hgnn_production_model.pt")
DEFAULT_PRODUCTION_METRICS_PATH = Path("app/ml/data/metrics_latest.json")
DEFAULT_PRODUCTION_AUDIT_PATH = Path("app/ml/data/context_examples_audit.json")
DEFAULT_PRODUCTION_SIDECAR_PATH = Path(
    "app/ml/data/experiments/semantic_identity_sidecar_compact.npz"
)
DEFAULT_BOOTSTRAP_SEED = 20260604

PRODUCTION_DIMS = {
    "static": 16,
    "full_game": 64,
    "temporal": 64,
}

PRODUCTION_AUDIT_TARGETS = {
    "all": {
        "mean_abs_gap": 0.013310247786660017,
        "max_abs_gap": 0.06399081970172593,
        "gap_mse": 0.0003160303787673601,
        "accuracy": 0.5814042417319056,
    },
    "val": {
        "mean_abs_gap": 0.017781307418476743,
        "gap_mse": 0.0005965371205160952,
    },
    "test": {
        "mean_abs_gap": 0.017248461569548984,
        "gap_mse": 0.0005396753753561297,
    },
}

PRODUCTION_MODEL_TARGETS = {
    "val_accuracy": 0.5788543362374329,
    "test_accuracy": 0.5737960330047299,
    "val_nll": 0.6729778797395639,
    "test_nll": 0.6759647813049514,
}
PRODUCTION_BASELINE_SOURCES = {
    "audit": str(DEFAULT_PRODUCTION_AUDIT_PATH),
    "model_metrics": str(DEFAULT_PRODUCTION_METRICS_PATH),
}

SCREEN_IMPROVEMENT_FACTOR = 0.95
ACCURACY_REGRESSION_TOLERANCE = 0.0010
NLL_REGRESSION_TOLERANCE = 0.0010
# The ablation protocol reports AUC/ECE, but gates only accuracy and NLL per
# the promotion requirement in the experiment plan.
VALIDATION_MODEL_GATE_KEYS = ("val_accuracy", "val_nll")
PROMOTION_MODEL_GATE_KEYS = ("val_accuracy", "test_accuracy", "val_nll", "test_nll")


@dataclass(frozen=True)
class AblationSpec:
    """One sidecar input recipe in the encoder calibration matrix."""

    name: str
    stage: int
    description: str
    sidecar_flags: tuple[str, ...] = ()
    static_latent_dim: int = PRODUCTION_DIMS["static"]
    full_game_latent_dim: int = PRODUCTION_DIMS["full_game"]
    temporal_latent_dim: int = PRODUCTION_DIMS["temporal"]
    reuse_static_full_game_from_control: bool = False
    needs_sidecar_build: bool = True
    promotion_eligible: bool = True
    diagnostic_only: bool = False

    @property
    def changes_hgnn_shape(self) -> bool:
        return (
            self.static_latent_dim != PRODUCTION_DIMS["static"]
            or self.full_game_latent_dim != PRODUCTION_DIMS["full_game"]
            or self.temporal_latent_dim != PRODUCTION_DIMS["temporal"]
        )


CONTROL_REBUILT = AblationSpec(
    name="control_rebuilt",
    stage=0,
    description="Current-code compact sidecar rebuilt with production dimensions.",
)

PRODUCTION_SIDECAR_CONTROL = AblationSpec(
    name="production_sidecar_control",
    stage=0,
    description="Matched HGNN train/audit using the checked-in production sidecar.",
    needs_sidecar_build=False,
)

STAGE1_ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec(
        name="fg_profile_only",
        stage=1,
        description="Full-game autoencoder uses raw+derived profile metrics only.",
        sidecar_flags=("--full-game-input-surface", "profile_only"),
    ),
    AblationSpec(
        name="fg_raw_context",
        stage=1,
        description="Full-game autoencoder uses raw profile metrics plus context metrics.",
        sidecar_flags=("--full-game-input-surface", "raw_context"),
    ),
    AblationSpec(
        name="fg_context_only",
        stage=1,
        description="Full-game autoencoder uses context metrics only.",
        sidecar_flags=("--full-game-input-surface", "context_only"),
    ),
    AblationSpec(
        name="fg_no_identity",
        stage=1,
        description="Full-game autoencoder disables champion/role/build embeddings.",
        sidecar_flags=("--full-game-identity-mode", "disabled"),
    ),
    AblationSpec(
        name="fg_support_log1p",
        stage=1,
        description="Full-game reconstruction is weighted by normalized log support.",
        sidecar_flags=("--full-game-support-weighting", "log1p"),
    ),
    AblationSpec(
        name="fg_soft_v2_w010",
        stage=1,
        description="Full-game AE with soft-v2 semantic auxiliary targets at weight 0.10.",
        sidecar_flags=(
            "--full-game-semantic-target-mode",
            "soft_v2",
            "--full-game-semantic-target-weight",
            "0.10",
        ),
    ),
    AblationSpec(
        name="fg_soft_v2_w020",
        stage=1,
        description="Full-game AE with soft-v2 semantic auxiliary targets at weight 0.20.",
        sidecar_flags=(
            "--full-game-semantic-target-mode",
            "soft_v2",
            "--full-game-semantic-target-weight",
            "0.20",
        ),
    ),
    AblationSpec(
        name="fg_pca_whitened",
        stage=1,
        description="Full-game latent block is deterministic PCA-whitened metrics.",
        sidecar_flags=("--full-game-latent-export", "pca_whitened"),
    ),
    AblationSpec(
        name="fg_semantic_targets",
        stage=1,
        description="Diagnostic direct soft-v2 semantic targets as full-game latents.",
        sidecar_flags=(
            "--full-game-latent-export",
            "semantic_targets",
            "--full-game-semantic-target-mode",
            "soft_v2",
        ),
        promotion_eligible=False,
        diagnostic_only=True,
    ),
    AblationSpec(
        name="static_latent_32",
        stage=1,
        description="Static champion latent widened from 16 to 32 dimensions.",
        static_latent_dim=32,
    ),
    AblationSpec(
        name="tmp_mask_flat",
        stage=1,
        description="Temporal flat encoder receives the observed-bucket mask channel.",
        sidecar_flags=("--temporal-mask-as-input",),
        reuse_static_full_game_from_control=True,
    ),
    AblationSpec(
        name="tmp_mask_tcn",
        stage=1,
        description="Temporal TCN encoder receives the observed-bucket mask channel.",
        sidecar_flags=("--temporal-mask-as-input", "--temporal-architecture", "tcn"),
        reuse_static_full_game_from_control=True,
    ),
    AblationSpec(
        name="tmp_mask_gru",
        stage=1,
        description="Temporal GRU encoder receives the observed-bucket mask channel.",
        sidecar_flags=("--temporal-mask-as-input", "--temporal-architecture", "gru"),
        reuse_static_full_game_from_control=True,
    ),
    AblationSpec(
        name="tmp_mask_standalone",
        stage=1,
        description="Standalone-width temporal flat encoder with observed-bucket mask.",
        sidecar_flags=(
            "--temporal-mask-as-input",
            "--temporal-width-profile",
            "standalone",
        ),
        reuse_static_full_game_from_control=True,
    ),
    AblationSpec(
        name="tmp_no_zero_unobserved",
        stage=1,
        description="Diagnostic: leave temporal unobserved bucket values unzeroed.",
        sidecar_flags=("--no-temporal-zero-unobserved-input",),
        reuse_static_full_game_from_control=True,
        promotion_eligible=False,
        diagnostic_only=True,
    ),
    AblationSpec(
        name="mv_vicreg_w010",
        stage=1,
        description="Post-encoder VICReg-style multiview alignment at weight 0.10.",
        sidecar_flags=(
            "--multiview-alignment-objective",
            "vicreg",
            "--multiview-alignment-weight",
            "0.10",
        ),
    ),
    AblationSpec(
        name="mv_barlow_w010",
        stage=1,
        description="Post-encoder Barlow-style multiview alignment at weight 0.10.",
        sidecar_flags=(
            "--multiview-alignment-objective",
            "barlow",
            "--multiview-alignment-weight",
            "0.10",
        ),
    ),
)

ABLATIONS: tuple[AblationSpec, ...] = (
    CONTROL_REBUILT,
    PRODUCTION_SIDECAR_CONTROL,
    *STAGE1_ABLATIONS,
)
ABLATION_BY_NAME: Mapping[str, AblationSpec] = {spec.name: spec for spec in ABLATIONS}


def run_dir(spec: AblationSpec, *, experiment_root: Path) -> Path:
    return experiment_root / spec.name


def sidecar_path(
    spec: AblationSpec,
    *,
    experiment_root: Path,
    production_sidecar_path: Path = DEFAULT_PRODUCTION_SIDECAR_PATH,
) -> Path:
    if not spec.needs_sidecar_build:
        return production_sidecar_path
    return run_dir(spec, experiment_root=experiment_root) / "sidecar.npz"


def sidecar_summary_path(spec: AblationSpec, *, experiment_root: Path) -> Path:
    return run_dir(spec, experiment_root=experiment_root) / "sidecar_summary.json"


def model_path(spec: AblationSpec, *, experiment_root: Path, seed: int) -> Path:
    return run_dir(spec, experiment_root=experiment_root) / f"model_seed{seed}.pt"


def metrics_path(spec: AblationSpec, *, experiment_root: Path, seed: int) -> Path:
    return run_dir(spec, experiment_root=experiment_root) / f"metrics_seed{seed}.json"


def prediction_cache_path(spec: AblationSpec, *, experiment_root: Path, seed: int) -> Path:
    return run_dir(spec, experiment_root=experiment_root) / f"audit_focus_seed{seed}.npy"


def audit_markdown_path(
    spec: AblationSpec,
    *,
    experiment_root: Path,
    seed: int,
    audit_split: str,
) -> Path:
    return (
        run_dir(spec, experiment_root=experiment_root)
        / f"HGNN_CONTEXT_EXAMPLES_AUDIT_{audit_split}_seed{seed}.md"
    )


def audit_json_path(
    spec: AblationSpec,
    *,
    experiment_root: Path,
    seed: int,
    audit_split: str,
) -> Path:
    return (
        run_dir(spec, experiment_root=experiment_root)
        / f"context_examples_audit_{audit_split}_seed{seed}.json"
    )


def group_audit_json_path(spec: AblationSpec, *, experiment_root: Path, seed: int) -> Path:
    return run_dir(spec, experiment_root=experiment_root) / f"group_context_audit_seed{seed}.json"


def _base_sidecar_args(spec: AblationSpec) -> tuple[str, ...]:
    return (
        "--device",
        "auto",
        "--seed",
        "7",
        "--batch-size",
        "1024",
        "--static-latent-dim",
        str(spec.static_latent_dim),
        "--full-game-latent-dim",
        str(spec.full_game_latent_dim),
        "--temporal-latent-dim",
        str(spec.temporal_latent_dim),
        "--static-epochs",
        "500",
        "--full-game-epochs",
        "200",
        "--temporal-epochs",
        "200",
    )


def build_sidecar_command(
    spec: AblationSpec,
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    control_sidecar_path: Path | None = None,
) -> list[str]:
    if not spec.needs_sidecar_build:
        raise ValueError(f"{spec.name} reuses an existing sidecar and has no build step")
    output = sidecar_path(spec, experiment_root=experiment_root)
    summary = sidecar_summary_path(spec, experiment_root=experiment_root)
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "app.ml.build_encoder_sidecar",
        "--output",
        str(output),
        "--summary-output",
        str(summary),
        *_base_sidecar_args(spec),
    ]
    if spec.reuse_static_full_game_from_control:
        command.extend(
            [
                "--reuse-static-full-game-from",
                str(control_sidecar_path or sidecar_path(CONTROL_REBUILT, experiment_root=experiment_root)),
            ]
        )
    command.extend(spec.sidecar_flags)
    return command


def should_freeze_warm_start(spec: AblationSpec, freeze_mode: str) -> bool:
    if freeze_mode == "always":
        return True
    if freeze_mode == "never":
        return False
    if freeze_mode != "auto":
        raise ValueError("freeze_mode must be one of: auto, always, never")
    return spec.changes_hgnn_shape


def train_command(
    spec: AblationSpec,
    *,
    seed: int,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    production_model_path: Path = DEFAULT_PRODUCTION_MODEL_PATH,
    production_sidecar_path: Path = DEFAULT_PRODUCTION_SIDECAR_PATH,
    freeze_mode: str = "auto",
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
        str(
            sidecar_path(
                spec,
                experiment_root=experiment_root,
                production_sidecar_path=production_sidecar_path,
            )
        ),
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
        "40",
        "--patience",
        "5",
        "--learning-rate",
        "1e-4",
        "--seed",
        str(seed),
        "--checkpoint-metric",
        "val_accuracy",
        "--use-learned-semantic-moe",
        "--use-semantic-group-features",
        "--semantic-moe-architecture",
        "convex_encoder_mix",
        "--semantic-moe-num-experts",
        "128",
        "--semantic-moe-top-k",
        "32",
        "--semantic-context-calibration-target",
        "group_eb",
        "--semantic-context-calibration-loss-weight",
        "10.0",
    ]
    if should_freeze_warm_start(spec, freeze_mode):
        command.append("--freeze-warm-start-loaded-parameters")
    return command


def audit_command(
    spec: AblationSpec,
    *,
    seed: int,
    audit_split: str,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    production_sidecar_path: Path = DEFAULT_PRODUCTION_SIDECAR_PATH,
    refresh_predictions: bool = False,
    bootstrap_samples: int = 0,
) -> list[str]:
    cli_audit_split = "val" if audit_split == "validation" else audit_split
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
        str(
            sidecar_path(
                spec,
                experiment_root=experiment_root,
                production_sidecar_path=production_sidecar_path,
            )
        ),
        "--prediction-cache",
        str(prediction_cache_path(spec, experiment_root=experiment_root, seed=seed)),
        "--audit-split",
        cli_audit_split,
        "--output",
        str(
            audit_markdown_path(
                spec,
                experiment_root=experiment_root,
                seed=seed,
                audit_split=audit_split,
            )
        ),
        "--json-output",
        str(
            audit_json_path(
                spec,
                experiment_root=experiment_root,
                seed=seed,
                audit_split=audit_split,
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
    spec: AblationSpec,
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
        str(group_audit_json_path(spec, experiment_root=experiment_root, seed=seed)),
    ]


def spec_payload(spec: AblationSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "stage": spec.stage,
        "description": spec.description,
        "sidecar_flags": list(spec.sidecar_flags),
        "static_latent_dim": spec.static_latent_dim,
        "full_game_latent_dim": spec.full_game_latent_dim,
        "temporal_latent_dim": spec.temporal_latent_dim,
        "reuse_static_full_game_from_control": spec.reuse_static_full_game_from_control,
        "needs_sidecar_build": spec.needs_sidecar_build,
        "changes_hgnn_shape": spec.changes_hgnn_shape,
        "promotion_eligible": spec.promotion_eligible,
        "diagnostic_only": spec.diagnostic_only,
    }


def selected_specs(names: Sequence[str]) -> list[AblationSpec]:
    if not names or names == ["all"]:
        return list(ABLATIONS)
    if names == ["stage1"]:
        return list(STAGE1_ABLATIONS)
    out: list[AblationSpec] = []
    for name in names:
        try:
            out.append(ABLATION_BY_NAME[name])
        except KeyError as exc:
            known = ", ".join(sorted(ABLATION_BY_NAME))
            raise ValueError(f"unknown ablation {name!r}; known: {known}") from exc
    return out


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def _summary_by_split(audit: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    summaries = audit.get("split_summaries", [])
    if not isinstance(summaries, list):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for row in summaries:
        if isinstance(row, Mapping) and isinstance(row.get("split"), str):
            out[str(row["split"])] = row
    return out


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(metrics: Mapping[str, Any] | None, key: str) -> Any:
    if metrics is None:
        return None
    direct = metrics.get(key)
    if direct is not None:
        return direct
    if "_" not in key:
        return None
    split, metric_name = key.split("_", 1)
    split_payload = metrics.get(split)
    if isinstance(split_payload, Mapping):
        return split_payload.get(metric_name)
    return None


def _model_gate_values(
    metrics: Mapping[str, Any] | None,
    keys: Sequence[str],
) -> dict[str, float] | None:
    if metrics is None:
        return None
    values: dict[str, float] = {}
    for key in keys:
        value = _safe_float(_metric_value(metrics, key))
        if value is None:
            return None
        values[key] = value
    return values


def _extract_validation_summary(audit: Mapping[str, Any]) -> Mapping[str, Any] | None:
    by_split = _summary_by_split(audit)
    return by_split.get("val") or by_split.get("validation")


def _extract_test_summary(audit: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return _summary_by_split(audit).get("test")


def _extract_all_summary(audit: Mapping[str, Any]) -> Mapping[str, Any] | None:
    split_payload = audit.get("splits")
    if isinstance(split_payload, Mapping):
        all_payload = split_payload.get("all")
        if isinstance(all_payload, Mapping):
            summary = all_payload.get("summary")
            if isinstance(summary, Mapping):
                return summary
    summaries = _summary_by_split(audit)
    return summaries.get("all")


def _summary_deltas(
    current: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    keys: Sequence[str],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in keys:
        current_value = None if current is None else _safe_float(current.get(key))
        baseline_value = None if baseline is None else _safe_float(baseline.get(key))
        out[key] = (
            None
            if current_value is None or baseline_value is None
            else current_value - baseline_value
        )
    return out


def _metric_deltas(
    current: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    keys: Sequence[str],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in keys:
        current_value = _safe_float(_metric_value(current, key))
        baseline_value = _safe_float(_metric_value(baseline, key))
        out[key] = (
            None
            if current_value is None or baseline_value is None
            else current_value - baseline_value
        )
    return out


def _validation_screen_passes(
    *,
    validation_summary: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None,
) -> bool | None:
    if validation_summary is None:
        return None
    gap_mse = _safe_float(validation_summary.get("gap_mse"))
    mean_abs_gap = _safe_float(validation_summary.get("mean_abs_gap"))
    improved = (
        gap_mse is not None
        and gap_mse < PRODUCTION_AUDIT_TARGETS["val"]["gap_mse"] * SCREEN_IMPROVEMENT_FACTOR
    ) or (
        mean_abs_gap is not None
        and mean_abs_gap
        < PRODUCTION_AUDIT_TARGETS["val"]["mean_abs_gap"] * SCREEN_IMPROVEMENT_FACTOR
    )
    if not improved:
        return False
    gate_values = _model_gate_values(metrics, VALIDATION_MODEL_GATE_KEYS)
    if gate_values is None:
        return None
    accuracy_ok = (
        gate_values["val_accuracy"]
        >= PRODUCTION_MODEL_TARGETS["val_accuracy"] - ACCURACY_REGRESSION_TOLERANCE
    )
    nll_ok = (
        gate_values["val_nll"]
        <= PRODUCTION_MODEL_TARGETS["val_nll"] + NLL_REGRESSION_TOLERANCE
    )
    return bool(accuracy_ok and nll_ok)


def _promotion_passes(
    *,
    all_summary: Mapping[str, Any] | None,
    validation_summary: Mapping[str, Any] | None,
    test_summary: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None,
    promotion_eligible: bool,
) -> bool | None:
    if not promotion_eligible:
        return False
    if all_summary is None or validation_summary is None or test_summary is None:
        return None
    all_targets = PRODUCTION_AUDIT_TARGETS["all"]
    val_targets = PRODUCTION_AUDIT_TARGETS["val"]
    test_targets = PRODUCTION_AUDIT_TARGETS["test"]
    audit_ok = (
        _safe_float(all_summary.get("mean_abs_gap")) is not None
        and _safe_float(all_summary.get("mean_abs_gap")) < all_targets["mean_abs_gap"]
        and _safe_float(all_summary.get("max_abs_gap")) is not None
        and _safe_float(all_summary.get("max_abs_gap")) < all_targets["max_abs_gap"]
        and _safe_float(all_summary.get("gap_mse")) is not None
        and _safe_float(all_summary.get("gap_mse")) < all_targets["gap_mse"]
        and _safe_float(all_summary.get("accuracy")) is not None
        and _safe_float(all_summary.get("accuracy")) >= all_targets["accuracy"]
        and _safe_float(validation_summary.get("gap_mse")) is not None
        and _safe_float(validation_summary.get("gap_mse")) < val_targets["gap_mse"]
        and _safe_float(validation_summary.get("mean_abs_gap")) is not None
        and _safe_float(validation_summary.get("mean_abs_gap"))
        < val_targets["mean_abs_gap"]
        and _safe_float(test_summary.get("gap_mse")) is not None
        and _safe_float(test_summary.get("gap_mse")) < test_targets["gap_mse"]
        and _safe_float(test_summary.get("mean_abs_gap")) is not None
        and _safe_float(test_summary.get("mean_abs_gap")) < test_targets["mean_abs_gap"]
    )
    if not audit_ok:
        return False
    gate_values = _model_gate_values(metrics, PROMOTION_MODEL_GATE_KEYS)
    if gate_values is None:
        return None
    model_ok = (
        gate_values["val_accuracy"]
        >= PRODUCTION_MODEL_TARGETS["val_accuracy"] - ACCURACY_REGRESSION_TOLERANCE
        and gate_values["test_accuracy"]
        >= PRODUCTION_MODEL_TARGETS["test_accuracy"] - ACCURACY_REGRESSION_TOLERANCE
        and gate_values["val_nll"]
        <= PRODUCTION_MODEL_TARGETS["val_nll"] + NLL_REGRESSION_TOLERANCE
        and gate_values["test_nll"]
        <= PRODUCTION_MODEL_TARGETS["test_nll"] + NLL_REGRESSION_TOLERANCE
    )
    return bool(model_ok)


def compare_runs(
    specs: Iterable[AblationSpec],
    *,
    seeds: Sequence[int],
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
) -> dict[str, object]:
    control_by_seed: dict[int, dict[str, Mapping[str, Any] | None]] = {}
    for seed in seeds:
        control_metrics = _load_json(
            metrics_path(CONTROL_REBUILT, experiment_root=experiment_root, seed=seed)
        )
        control_validation_audit = _load_json(
            audit_json_path(
                CONTROL_REBUILT,
                experiment_root=experiment_root,
                seed=seed,
                audit_split="validation",
            )
        )
        control_by_seed[seed] = {
            "metrics": control_metrics,
            "validation_summary": (
                _extract_validation_summary(control_validation_audit)
                if control_validation_audit is not None
                else None
            ),
        }
    rows: list[dict[str, object]] = []
    for spec in specs:
        for seed in seeds:
            metrics = _load_json(metrics_path(spec, experiment_root=experiment_root, seed=seed))
            validation_audit = _load_json(
                audit_json_path(
                    spec,
                    experiment_root=experiment_root,
                    seed=seed,
                    audit_split="validation",
                )
            )
            all_audit = _load_json(
                audit_json_path(
                    spec,
                    experiment_root=experiment_root,
                    seed=seed,
                    audit_split="all",
                )
            )
            validation_summary = (
                _extract_validation_summary(validation_audit)
                if validation_audit is not None
                else None
            )
            all_summary = _extract_all_summary(all_audit) if all_audit is not None else None
            test_summary = _extract_test_summary(all_audit) if all_audit is not None else None
            control = control_by_seed.get(seed, {})
            control_metrics = control.get("metrics")
            control_validation_summary = control.get("validation_summary")
            rows.append(
                {
                    "run": spec.name,
                    "seed": seed,
                    "stage": spec.stage,
                    "promotion_eligible": spec.promotion_eligible,
                    "diagnostic_only": spec.diagnostic_only,
                    "paths": {
                        "metrics": str(
                            metrics_path(
                                spec,
                                experiment_root=experiment_root,
                                seed=seed,
                            )
                        ),
                        "validation_audit_json": str(
                            audit_json_path(
                                spec,
                                experiment_root=experiment_root,
                                seed=seed,
                                audit_split="validation",
                            )
                        ),
                        "all_audit_json": str(
                            audit_json_path(
                                spec,
                                experiment_root=experiment_root,
                                seed=seed,
                                audit_split="all",
                            )
                        ),
                    },
                    "metrics": {
                        key: _metric_value(metrics, key)
                        for key in (
                            "val_accuracy",
                            "test_accuracy",
                            "val_nll",
                            "test_nll",
                            "val_auc",
                            "test_auc",
                        )
                    },
                    "validation_audit": {
                        key: None
                        if validation_summary is None
                        else validation_summary.get(key)
                        for key in (
                            "gap_mse",
                            "mean_abs_gap",
                            "max_abs_gap",
                            "accuracy",
                            "calibrated_accuracy",
                            "calibration_lift",
                        )
                    },
                    "all_audit": {
                        key: None if all_summary is None else all_summary.get(key)
                        for key in (
                            "gap_mse",
                            "mean_abs_gap",
                            "max_abs_gap",
                            "accuracy",
                            "calibrated_accuracy",
                            "calibration_lift",
                        )
                    },
                    "test_audit": {
                        key: None if test_summary is None else test_summary.get(key)
                        for key in (
                            "gap_mse",
                            "mean_abs_gap",
                            "max_abs_gap",
                            "accuracy",
                            "calibrated_accuracy",
                            "calibration_lift",
                        )
                    },
                    "validation_vs_control_rebuilt": _summary_deltas(
                        validation_summary,
                        control_validation_summary,
                        (
                            "gap_mse",
                            "mean_abs_gap",
                            "max_abs_gap",
                            "accuracy",
                            "calibrated_accuracy",
                            "calibration_lift",
                        ),
                    ),
                    "metrics_vs_control_rebuilt": _metric_deltas(
                        metrics,
                        control_metrics,
                        (
                            "val_accuracy",
                            "test_accuracy",
                            "val_nll",
                            "test_nll",
                            "val_auc",
                            "test_auc",
                        ),
                    ),
                    "passes_validation_screen": _validation_screen_passes(
                        validation_summary=validation_summary,
                        metrics=metrics,
                    ),
                    "passes_promotion_gate": _promotion_passes(
                        all_summary=all_summary,
                        validation_summary=validation_summary,
                        test_summary=test_summary,
                        metrics=metrics,
                        promotion_eligible=spec.promotion_eligible,
                    ),
                }
            )
    return {
        "schema_version": 1,
        "production_baseline_sources": PRODUCTION_BASELINE_SOURCES,
        "production_audit_targets": PRODUCTION_AUDIT_TARGETS,
        "production_model_targets": PRODUCTION_MODEL_TARGETS,
        "screen_improvement_factor": SCREEN_IMPROVEMENT_FACTOR,
        "accuracy_regression_tolerance": ACCURACY_REGRESSION_TOLERANCE,
        "nll_regression_tolerance": NLL_REGRESSION_TOLERANCE,
        "rows": rows,
    }


def _run_command(command: Sequence[str], *, dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_common_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["all"],
        help="Run names, 'stage1', or 'all'.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List ablation specs.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine JSON.")

    commands_parser = subparsers.add_parser("commands", help="Print reproducible commands.")
    _add_common_selection_args(commands_parser)
    commands_parser.add_argument(
        "--steps",
        nargs="+",
        choices=("sidecar", "train", "audit-validation", "audit-all", "group-audit"),
        default=("sidecar", "train", "audit-validation"),
    )
    commands_parser.add_argument("--seed", type=int, default=4)
    commands_parser.add_argument(
        "--freeze-mode",
        choices=("auto", "always", "never"),
        default="auto",
    )
    commands_parser.add_argument("--refresh-predictions", action="store_true")
    commands_parser.add_argument("--bootstrap-samples", type=int, default=0)

    run_parser = subparsers.add_parser("run", help="Execute selected commands.")
    _add_common_selection_args(run_parser)
    run_parser.add_argument(
        "--steps",
        nargs="+",
        choices=("sidecar", "train", "audit-validation", "audit-all", "group-audit"),
        default=("sidecar", "train", "audit-validation"),
    )
    run_parser.add_argument("--seed", type=int, default=4)
    run_parser.add_argument(
        "--freeze-mode",
        choices=("auto", "always", "never"),
        default="auto",
    )
    run_parser.add_argument("--refresh-predictions", action="store_true")
    run_parser.add_argument("--bootstrap-samples", type=int, default=0)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare completed runs.")
    _add_common_selection_args(compare_parser)
    compare_parser.add_argument("--seeds", nargs="+", type=int, default=[4])
    compare_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "ablation_comparison.json",
    )

    return parser.parse_args(argv)


def _commands_for_steps(
    spec: AblationSpec,
    *,
    steps: Sequence[str],
    seed: int,
    experiment_root: Path,
    freeze_mode: str,
    refresh_predictions: bool,
    bootstrap_samples: int,
    emitted_prerequisites: set[str] | None = None,
) -> list[list[str]]:
    commands: list[list[str]] = []
    if "sidecar" in steps and spec.reuse_static_full_game_from_control:
        control_sidecar = sidecar_path(CONTROL_REBUILT, experiment_root=experiment_root)
        prerequisite_key = str(control_sidecar)
        if not control_sidecar.exists() and (
            emitted_prerequisites is None
            or prerequisite_key not in emitted_prerequisites
        ):
            commands.append(
                build_sidecar_command(CONTROL_REBUILT, experiment_root=experiment_root)
            )
            if emitted_prerequisites is not None:
                emitted_prerequisites.add(prerequisite_key)
    if "sidecar" in steps and spec.needs_sidecar_build:
        commands.append(build_sidecar_command(spec, experiment_root=experiment_root))
    if "train" in steps:
        commands.append(
            train_command(
                spec,
                seed=seed,
                experiment_root=experiment_root,
                freeze_mode=freeze_mode,
            )
        )
    if "audit-validation" in steps:
        commands.append(
            audit_command(
                spec,
                seed=seed,
                audit_split="validation",
                experiment_root=experiment_root,
                refresh_predictions=refresh_predictions,
                bootstrap_samples=bootstrap_samples,
            )
        )
    if "audit-all" in steps:
        commands.append(
            audit_command(
                spec,
                seed=seed,
                audit_split="all",
                experiment_root=experiment_root,
                refresh_predictions=refresh_predictions,
                bootstrap_samples=bootstrap_samples,
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
        payload = [spec_payload(spec) for spec in ABLATIONS]
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for row in payload:
                marker = "diagnostic" if row["diagnostic_only"] else "candidate"
                print(
                    f"{row['name']}\tstage={row['stage']}\t{marker}\t"
                    f"{row['description']}"
                )
        return 0

    if args.command in {"commands", "run"}:
        specs = selected_specs(args.runs)
        dry_run = args.command == "commands" or bool(args.dry_run)
        emitted_prerequisites: set[str] = set()
        for spec in specs:
            for command in _commands_for_steps(
                spec,
                steps=args.steps,
                seed=int(args.seed),
                experiment_root=args.experiment_root,
                freeze_mode=args.freeze_mode,
                refresh_predictions=bool(args.refresh_predictions),
                bootstrap_samples=int(args.bootstrap_samples),
                emitted_prerequisites=emitted_prerequisites,
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
    sys.exit(main())
