"""Launch HGNN context encoder / conditioning ablation variants.

This entrypoint records the research-informed ablation matrix without running
the long GPU study as part of normal tests. Use ``--dry-run`` to inspect the
planned variants and output paths, or run selected variants with ``--variants``.

Example:
    python -m app.ml.experiments.context_ablation --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.ml.config import ML_DATA_DIR, DatasetConfig, TrainConfig
from app.ml.context_examples_audit import write_semantic_summary
from app.ml.train import train

AGGREGATED_NUMERIC_FIELDS: tuple[str, ...] = (
    "best_epoch",
    "best_checkpoint_score",
    "best_checkpoint_val_nll",
    "best_checkpoint_val_ece",
    "checkpoint_min_delta",
    "auc_ranking_loss_weight",
    "auc_ranking_loss_pairs",
    "decision_threshold",
    "val_threshold_accuracy",
    "val_auc",
    "val_nll",
    "val_ece",
    "val_brier",
    "val_temperature_scaled_nll",
    "val_temperature_scaled_ece",
    "test_threshold_accuracy",
    "test_accuracy",
    "test_auc",
    "test_nll",
    "test_brier",
    "test_ece",
    "test_temperature_scaled_nll",
    "test_temperature_scaled_ece",
    "temperature",
    "val_prior_1vx_support_max_abs_gap",
    "val_prior_1vx_support_min_bucket_auc",
    "test_prior_1vx_support_max_abs_gap",
    "test_prior_1vx_support_min_bucket_auc",
    "val_identity_context_support_max_abs_gap",
    "test_identity_context_support_max_abs_gap",
    "val_context_mean_abs_logit",
    "val_context_p95_abs_logit",
    "test_context_mean_abs_logit",
    "test_context_p95_abs_logit",
    "test_context_support_temperature_nll",
    "test_context_support_temperature_ece",
    "val_semantic_n_effects",
    "val_semantic_mean_abs_delta_gap",
    "val_semantic_max_abs_delta_gap",
    "val_semantic_mean_abs_endpoint_gap",
    "val_semantic_max_abs_endpoint_gap",
    "val_semantic_worst_effect_delta_gap",
)


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: dict[str, Any]
    train_overrides: dict[str, Any] = field(default_factory=dict)


TRAIN_OVERRIDE_FIELDS = frozenset(
    {
        "checkpoint_metric",
        "checkpoint_min_delta",
        "context_auxiliary_loss_weight",
        "auc_ranking_loss_weight",
        "auc_ranking_loss_pairs",
    }
)


DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant(
        "current_mean_low_rank",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
        },
    ),
    Variant(
        "shared_current_mean",
        {
            "use_identity_conditioned_context": False,
            "identity_context_conditioning_type": "none",
            "context_set_encoder_type": "mean",
        },
    ),
    Variant(
        "film_current_mean",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "film",
            "context_set_encoder_type": "mean",
        },
    ),
    Variant(
        "low_rank_deepsets",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "deepsets",
        },
    ),
    Variant(
        "low_rank_set_transformer",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "set_transformer",
        },
    ),
    Variant(
        "low_rank_attention_pooling",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "attention",
        },
    ),
    Variant(
        "low_rank_summary_stats",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "summary_stats",
        },
    ),
    Variant(
        "low_rank_context_products",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
            "identity_context_include_products": True,
        },
    ),
    Variant(
        "low_rank_support_features",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
            "identity_context_include_support_features": True,
        },
    ),
    Variant(
        "low_rank_no_1vx_variance",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
            "use_1vx_posterior_variance": False,
        },
    ),
    Variant(
        "low_rank_checkpoint_auc",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
        },
        {"checkpoint_metric": "val_auc", "checkpoint_min_delta": 1.0e-4},
    ),
    Variant(
        "low_rank_checkpoint_nll",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
        },
        {"checkpoint_metric": "val_nll", "checkpoint_min_delta": 1.0e-4},
    ),
    Variant(
        "low_rank_checkpoint_nll_ece",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
        },
        {"checkpoint_metric": "val_nll_ece", "checkpoint_min_delta": 1.0e-4},
    ),
    Variant(
        "low_rank_context_auxiliary",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
        },
        {"context_auxiliary_loss_weight": 0.25},
    ),
    Variant(
        "low_rank_auc_ranking_loss",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
        },
        {
            "checkpoint_metric": "val_auc",
            "checkpoint_min_delta": 1.0e-4,
            "auc_ranking_loss_weight": 0.10,
            "auc_ranking_loss_pairs": 8192,
        },
    ),
    Variant(
        "low_rank_profile_detail_auc",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
            "identity_profile_dim": "auto",
            "profile_include_ally_context": True,
            "profile_include_weighted_enemy_context": True,
            "profile_include_resistance_products": True,
            "profile_head_hidden": (32,),
            "m1v1_detail_dim": "auto",
        },
        {
            "checkpoint_metric": "val_auc",
            "checkpoint_min_delta": 1.0e-4,
            "auc_ranking_loss_weight": 0.10,
            "auc_ranking_loss_pairs": 8192,
        },
    ),
    Variant(
        "low_rank_wide_context_auc",
        {
            "node_dim": 160,
            "edge_hidden": 128,
            "node_init_hidden": (160,),
            "readout_hidden": (512, 256),
            "residual_head_hidden": (384, 128),
            "dropout": 0.05,
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "identity_context_source": "raw_plus_dense",
            "identity_context_rank": 32,
            "identity_context_hidden_dim": 128,
            "identity_context_emb_dim": 32,
            "identity_context_dropout": 0.05,
            "identity_context_use_residual_mlp": True,
            "identity_context_include_products": True,
            "identity_context_include_support_features": True,
            "context_set_encoder_type": "mean",
        },
        {
            "checkpoint_metric": "val_auc",
            "checkpoint_min_delta": 1.0e-4,
            "auc_ranking_loss_weight": 0.10,
            "auc_ranking_loss_pairs": 8192,
        },
    ),
    Variant(
        "low_rank_slot_wide_context_auc",
        {
            "node_dim": 160,
            "edge_hidden": 128,
            "node_init_hidden": (160,),
            "readout_hidden": (512, 256),
            "team_slot_readout_hidden": (512, 256),
            "residual_head_hidden": (384, 128),
            "dropout": 0.05,
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "identity_context_source": "raw_plus_dense",
            "identity_context_rank": 32,
            "identity_context_hidden_dim": 128,
            "identity_context_emb_dim": 32,
            "identity_context_dropout": 0.05,
            "identity_context_use_residual_mlp": True,
            "identity_context_include_products": True,
            "identity_context_include_support_features": True,
            "context_set_encoder_type": "mean",
        },
        {
            "checkpoint_metric": "val_auc",
            "checkpoint_min_delta": 1.0e-4,
            "auc_ranking_loss_weight": 0.10,
            "auc_ranking_loss_pairs": 8192,
        },
    ),
    Variant(
        "low_rank_structural_antisymmetry",
        {
            "use_identity_conditioned_context": True,
            "identity_context_conditioning_type": "low_rank",
            "context_set_encoder_type": "mean",
            "structural_antisymmetry": True,
        },
    ),
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_train_overrides(variant: Variant) -> dict[str, Any]:
    unknown = sorted(set(variant.train_overrides) - TRAIN_OVERRIDE_FIELDS)
    if unknown:
        raise ValueError(
            f"{variant.name} has unsupported train_overrides: {', '.join(unknown)}"
        )
    return dict(variant.train_overrides)


def _normalise_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    seeds = (
        tuple(int(seed) for seed in args.seeds)
        if args.seeds is not None
        else (int(args.seed),)
    )
    if not seeds:
        raise SystemExit("--seeds requires at least one seed")
    if any(seed < 0 for seed in seeds):
        raise SystemExit("seed values must be >= 0")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds values must be unique")
    return seeds


def _variant_run_dir(output_dir: Path, name: str, seed: int, *, repeated: bool) -> Path:
    base = output_dir / name
    return base / f"seed_{seed}" if repeated else base


def _support_max_abs_gap(split_metrics: dict[str, Any], support_key: str) -> float | None:
    risk = (
        split_metrics.get("support_buckets", {})
        .get(support_key, {})
        .get("risk_bucket", {})
    )
    gaps = [
        abs(float(row["calibration_gap"]))
        for row in risk.values()
        if row.get("calibration_gap") is not None and int(row.get("n", 0)) > 0
    ]
    return max(gaps) if gaps else None


def _support_min_bucket_auc(split_metrics: dict[str, Any], support_key: str) -> float | None:
    risk = (
        split_metrics.get("support_buckets", {})
        .get(support_key, {})
        .get("risk_bucket", {})
    )
    aucs = [
        float(row["auc"])
        for row in risk.values()
        if row.get("auc") is not None and int(row.get("n", 0)) > 0
    ]
    finite = [value for value in aucs if math.isfinite(value)]
    return min(finite) if finite else None


def _context_residual_metric(split_metrics: dict[str, Any], key: str) -> float | None:
    value = split_metrics.get("context_residual", {}).get(key)
    return float(value) if value is not None else None


def _semantic_leaderboard_fields(
    summary: dict[str, Any] | None,
    summary_path: Path | None,
) -> dict[str, Any]:
    if summary is None:
        return {
            "semantic_summary_path": None,
            "val_semantic_n_effects": None,
            "val_semantic_mean_abs_delta_gap": None,
            "val_semantic_max_abs_delta_gap": None,
            "val_semantic_mean_abs_endpoint_gap": None,
            "val_semantic_max_abs_endpoint_gap": None,
            "val_semantic_worst_effect": None,
            "val_semantic_worst_effect_delta_gap": None,
        }
    semantic = summary.get("semantic_summary", {})
    aggregate = semantic.get("aggregate", {})
    effects = semantic.get("effects", [])
    worst = None
    if effects:
        worst = max(effects, key=lambda row: abs(float(row.get("delta_gap", 0.0))))
    return {
        "semantic_summary_path": str(summary_path)
        if summary_path is not None
        else None,
        "val_semantic_n_effects": aggregate.get("n_effects"),
        "val_semantic_mean_abs_delta_gap": aggregate.get("mean_abs_delta_gap"),
        "val_semantic_max_abs_delta_gap": aggregate.get("max_abs_delta_gap"),
        "val_semantic_mean_abs_endpoint_gap": aggregate.get("mean_abs_endpoint_gap"),
        "val_semantic_max_abs_endpoint_gap": aggregate.get("max_abs_endpoint_gap"),
        "val_semantic_worst_effect": worst.get("label") if worst is not None else None,
        "val_semantic_worst_effect_delta_gap": (
            worst.get("delta_gap") if worst is not None else None
        ),
    }


def _leaderboard_row(
    name: str,
    metrics: dict[str, Any],
    *,
    semantic_summary: dict[str, Any] | None = None,
    semantic_summary_path: Path | None = None,
) -> dict[str, Any]:
    test = metrics.get("test", {})
    val = metrics.get("val", {})
    val_temp = val.get("temperature_scaled", {})
    test_temp = test.get("temperature_scaled", {})
    test_context_temp = test.get("context_support_temperature_scaled", {})
    row = {
        "variant": name,
        "checkpoint_metric": metrics.get("train_config", {}).get("checkpoint_metric"),
        "checkpoint_min_delta": metrics.get("train_config", {}).get(
            "checkpoint_min_delta"
        ),
        "use_relationship_integrations": metrics.get("model_config", {}).get(
            "use_relationship_integrations"
        ),
        "use_1vx_posterior_variance": metrics.get("model_config", {}).get(
            "use_1vx_posterior_variance"
        ),
        "auc_ranking_loss_weight": metrics.get("train_config", {}).get(
            "auc_ranking_loss_weight"
        ),
        "auc_ranking_loss_pairs": metrics.get("train_config", {}).get(
            "auc_ranking_loss_pairs"
        ),
        "best_epoch": metrics.get("best_epoch"),
        "best_checkpoint_score": metrics.get("best_checkpoint_score"),
        "best_checkpoint_val_nll": metrics.get("best_checkpoint_val_nll"),
        "best_checkpoint_val_ece": metrics.get("best_checkpoint_val_ece"),
        "decision_threshold": metrics.get("decision_threshold"),
        "val_threshold_accuracy": val.get("threshold_accuracy"),
        "val_auc": val.get("auc"),
        "val_nll": val.get("nll"),
        "val_ece": val.get("ece"),
        "val_brier": val.get("brier"),
        "val_temperature_scaled_nll": val_temp.get("nll"),
        "val_temperature_scaled_ece": val_temp.get("ece"),
        "test_threshold_accuracy": test.get("threshold_accuracy"),
        "test_accuracy": test.get("accuracy"),
        "test_auc": test.get("auc"),
        "test_nll": test.get("nll"),
        "test_brier": test.get("brier"),
        "test_ece": test.get("ece"),
        "test_temperature_scaled_nll": test_temp.get("nll"),
        "test_temperature_scaled_ece": test_temp.get("ece"),
        "temperature": metrics.get("temperature_scaling", {}).get("temperature"),
        "context_support_calibration_available": metrics.get(
            "context_support_temperature_scaling", {}
        ).get("available"),
        "val_prior_1vx_support_max_abs_gap": _support_max_abs_gap(
            val,
            "prior_1vx_support",
        ),
        "val_prior_1vx_support_min_bucket_auc": _support_min_bucket_auc(
            val,
            "prior_1vx_support",
        ),
        "test_prior_1vx_support_max_abs_gap": _support_max_abs_gap(
            test,
            "prior_1vx_support",
        ),
        "test_prior_1vx_support_min_bucket_auc": _support_min_bucket_auc(
            test,
            "prior_1vx_support",
        ),
        "val_identity_context_support_max_abs_gap": _support_max_abs_gap(
            val,
            "identity_context_support",
        ),
        "test_identity_context_support_max_abs_gap": _support_max_abs_gap(
            test,
            "identity_context_support",
        ),
        "val_context_mean_abs_logit": _context_residual_metric(val, "mean_abs_logit"),
        "val_context_p95_abs_logit": _context_residual_metric(val, "p95_abs_logit"),
        "test_context_mean_abs_logit": _context_residual_metric(test, "mean_abs_logit"),
        "test_context_p95_abs_logit": _context_residual_metric(test, "p95_abs_logit"),
        "test_context_support_temperature_nll": test_context_temp.get("nll"),
        "test_context_support_temperature_ece": test_context_temp.get("ece"),
    }
    row.update(_semantic_leaderboard_fields(semantic_summary, semantic_summary_path))
    return row


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _common_value(rows: list[dict[str, Any]], key: str) -> Any:
    values = [row.get(key) for row in rows]
    if all(value is None for value in values):
        return None
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique[0] if len(unique) == 1 else unique


def _aggregate_repeated_rows(
    name: str,
    run_rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    if len(run_rows) != len(seeds):
        raise ValueError("run_rows and seeds must have the same length")
    if len(run_rows) == 1:
        row = dict(run_rows[0])
        row["seed"] = seeds[0]
        row["seeds"] = list(seeds)
        row["n_repeats"] = 1
        return row

    aggregate: dict[str, Any] = {
        "variant": name,
        "seeds": list(seeds),
        "n_repeats": len(run_rows),
        "runs": run_rows,
        "semantic_summary_paths": [
            row.get("semantic_summary_path")
            for row in run_rows
            if row.get("semantic_summary_path") is not None
        ],
        "context_support_calibration_available": all(
            bool(row.get("context_support_calibration_available")) for row in run_rows
        ),
        "semantic_summary_path": None,
        "checkpoint_metric": _common_value(run_rows, "checkpoint_metric"),
        "use_relationship_integrations": _common_value(
            run_rows,
            "use_relationship_integrations",
        ),
        "use_1vx_posterior_variance": _common_value(
            run_rows,
            "use_1vx_posterior_variance",
        ),
        "val_semantic_worst_effects": [
            row.get("val_semantic_worst_effect")
            for row in run_rows
            if row.get("val_semantic_worst_effect") is not None
        ],
    }
    for key in AGGREGATED_NUMERIC_FIELDS:
        values = _numeric_values(run_rows, key)
        mean, std = _mean_std(values)
        aggregate[key] = mean
        aggregate[f"{key}_mean"] = mean
        aggregate[f"{key}_std"] = std
        aggregate[f"{key}_min"] = min(values) if values else None
        aggregate[f"{key}_max"] = max(values) if values else None
    return aggregate


def _write_leaderboard(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def metric_value(row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        return float(value) if value is not None else float("-inf")

    def lower_metric_value(row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        return -float(value) if value is not None else float("-inf")

    ranked = sorted(
        rows,
        key=lambda row: (
            metric_value(row, "val_threshold_accuracy"),
            metric_value(row, "val_auc"),
            lower_metric_value(row, "val_nll"),
            lower_metric_value(row, "val_semantic_mean_abs_delta_gap"),
        ),
        reverse=True,
    )
    path.write_text(json.dumps(_json_value(ranked), indent=2), encoding="utf-8")
    md = path.with_suffix(".md")
    headers = (
        "variant",
        "checkpoint_metric",
        "use_relationship_integrations",
        "use_1vx_posterior_variance",
        "n_repeats",
        "auc_ranking_loss_weight",
        "auc_ranking_loss_pairs",
        "val_threshold_accuracy",
        "val_threshold_accuracy_std",
        "val_auc",
        "val_auc_std",
        "val_nll",
        "val_ece",
        "val_temperature_scaled_ece",
        "val_semantic_mean_abs_delta_gap",
        "val_semantic_mean_abs_delta_gap_std",
        "val_semantic_max_abs_delta_gap",
        "val_context_mean_abs_logit",
        "test_auc",
        "test_nll",
        "test_ece",
        "test_temperature_scaled_ece",
        "test_prior_1vx_support_max_abs_gap",
        "test_prior_1vx_support_min_bucket_auc",
        "test_identity_context_support_max_abs_gap",
        "test_context_support_temperature_nll",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in ranked:
        lines.append("| " + " | ".join(str(row.get(h)) for h in headers) + " |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _variant_map() -> dict[str, Variant]:
    return {variant.name: variant for variant in DEFAULT_VARIANTS}


def _parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ML_DATA_DIR / "experiments" / "context_ablation",
    )
    parser.add_argument(
        "--variants", nargs="*", default=None, help="Variant names to run"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DatasetConfig().cache_dir)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run every selected variant once per seed and aggregate validation-first leaderboard fields.",
    )
    parser.add_argument("--report-context-support-calibration", action="store_true")
    parser.add_argument("--context-auxiliary-loss-weight", type=float, default=None)
    parser.add_argument("--auc-ranking-loss-weight", type=float, default=None)
    parser.add_argument("--auc-ranking-loss-pairs", type=int, default=None)
    parser.add_argument(
        "--report-semantic-summary",
        action="store_true",
        help="Write validation-only semantic gap summaries and include them in the leaderboard.",
    )
    parser.add_argument(
        "--context-support-calibration-min-bucket",
        type=int,
        default=defaults.context_support_calibration_min_bucket,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_defaults = TrainConfig()
    variants = _variant_map()
    selected = args.variants or list(variants)
    seeds = _normalise_seeds(args)
    repeated = len(seeds) > 1
    unknown = [name for name in selected if name not in variants]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")
    if (
        args.context_auxiliary_loss_weight is not None
        and "low_rank_context_auxiliary" not in selected
    ):
        raise SystemExit(
            "--context-auxiliary-loss-weight only applies to low_rank_context_auxiliary"
        )
    if (
        args.context_auxiliary_loss_weight is not None
        and args.context_auxiliary_loss_weight < 0.0
    ):
        raise SystemExit("--context-auxiliary-loss-weight must be >= 0")
    if (
        args.auc_ranking_loss_weight is not None
        and not any(
            "auc_ranking_loss_weight" in variants[name].train_overrides
            for name in selected
        )
    ):
        raise SystemExit(
            "--auc-ranking-loss-weight only applies to variants with auc_ranking_loss_weight"
        )
    if (
        args.auc_ranking_loss_pairs is not None
        and not any(
            "auc_ranking_loss_pairs" in variants[name].train_overrides
            for name in selected
        )
    ):
        raise SystemExit(
            "--auc-ranking-loss-pairs only applies to variants with auc_ranking_loss_pairs"
        )
    if (
        args.auc_ranking_loss_weight is not None
        and args.auc_ranking_loss_weight < 0.0
    ):
        raise SystemExit("--auc-ranking-loss-weight must be >= 0")
    if args.auc_ranking_loss_pairs is not None and args.auc_ranking_loss_pairs < 1:
        raise SystemExit("--auc-ranking-loss-pairs must be >= 1")

    rows: list[dict[str, Any]] = []
    for name in selected:
        variant = variants[name]
        dataset_cfg = DatasetConfig(cache_dir=args.cache_dir, max_games=args.max_games)
        train_overrides = _validated_train_overrides(variant)
        if (
            name == "low_rank_context_auxiliary"
            and args.context_auxiliary_loss_weight is not None
        ):
            train_overrides["context_auxiliary_loss_weight"] = (
                args.context_auxiliary_loss_weight
            )
        if "auc_ranking_loss_weight" in train_overrides:
            if args.auc_ranking_loss_weight is not None:
                train_overrides["auc_ranking_loss_weight"] = args.auc_ranking_loss_weight
        if "auc_ranking_loss_pairs" in train_overrides:
            if args.auc_ranking_loss_pairs is not None:
                train_overrides["auc_ranking_loss_pairs"] = args.auc_ranking_loss_pairs
        run_rows: list[dict[str, Any]] = []
        for seed in seeds:
            run_dir = _variant_run_dir(args.output_dir, name, seed, repeated=repeated)
            model_path = run_dir / "model.pt"
            metrics_path = run_dir / "metrics.json"
            semantic_summary_path = run_dir / "semantic_summary_val.json"
            train_cfg = TrainConfig(
                model_path=model_path,
                metrics_path=metrics_path,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                device=args.device,
                seed=seed,
                checkpoint_metric=train_overrides.get(
                    "checkpoint_metric",
                    train_defaults.checkpoint_metric,
                ),
                checkpoint_min_delta=train_overrides.get(
                    "checkpoint_min_delta",
                    train_defaults.checkpoint_min_delta,
                ),
                report_context_support_calibration=args.report_context_support_calibration,
                context_support_calibration_min_bucket=args.context_support_calibration_min_bucket,
                context_auxiliary_loss_weight=train_overrides.get(
                    "context_auxiliary_loss_weight",
                    train_defaults.context_auxiliary_loss_weight,
                ),
                auc_ranking_loss_weight=train_overrides.get(
                    "auc_ranking_loss_weight",
                    train_defaults.auc_ranking_loss_weight,
                ),
                auc_ranking_loss_pairs=train_overrides.get(
                    "auc_ranking_loss_pairs",
                    train_defaults.auc_ranking_loss_pairs,
                ),
            )
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "variant": name,
                            "seed": seed,
                            "seeds": list(seeds),
                            "n_repeats": len(seeds),
                            "dataset_config": _json_value(asdict(dataset_cfg)),
                            "train_config": _json_value(asdict(train_cfg)),
                            "model_overrides": _json_value(variant.overrides),
                            "train_overrides": _json_value(train_overrides),
                            "semantic_summary_path": (
                                str(semantic_summary_path)
                                if args.report_semantic_summary
                                else None
                            ),
                        },
                        indent=2,
                    )
                )
                continue
            train(dataset_cfg, train_cfg, model_overrides=variant.overrides)
            semantic_summary = None
            if args.report_semantic_summary:
                semantic_summary = write_semantic_summary(
                    model_path,
                    semantic_summary_path,
                    dataset_config=dataset_cfg,
                    splits=("val",),
                    device=args.device,
                )
            row = _leaderboard_row(
                name,
                _read_metrics(metrics_path),
                semantic_summary=semantic_summary,
                semantic_summary_path=semantic_summary_path
                if semantic_summary is not None
                else None,
            )
            row["seed"] = seed
            run_rows.append(row)
        if run_rows:
            rows.append(_aggregate_repeated_rows(name, run_rows, seeds))

    if rows:
        _write_leaderboard(args.output_dir / "leaderboard.json", rows)


if __name__ == "__main__":
    main()
