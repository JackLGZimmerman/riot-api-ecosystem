from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from app.ml.config import ML_DATA_DIR
from app.ml.utils.metrics import metric_float, metric_scalar

DEFAULT_METRICS_PATH = ML_DATA_DIR / "metrics.jsonl"
SUMMARY_SCHEMA_VERSION = 2


def _fields(names: str) -> tuple[str, ...]:
    return tuple(names.split())


CORE_PROGRESS_FIELDS = _fields(
    "train_loss train_monitor_loss val_loss train_monitor_accuracy val_accuracy "
    "train_monitor_auc val_auc train_monitor_brier val_brier train_monitor_ece "
    "val_ece train_central_475_525_auc val_central_475_525_auc "
    "train_central_475_525_logloss val_central_475_525_logloss "
    "train_central_475_525_brier val_central_475_525_brier "
    "train_central_475_525_calibration val_central_475_525_calibration "
    "val_central_475_525_pct_data gen_loss_gap gen_accuracy_gap gen_auc_gap "
    "gen_brier_gap gen_ece_gap gen_central_475_525_logloss_gap "
    "gen_central_475_525_auc_gap train_mean_pred val_mean_pred "
    "train_positive_rate val_positive_rate lr epoch_s"
)
TRAIN_STEP_FIELDS = _fields("train_loss batch_loss lr grad_norm samples_per_s")
TIMELINE_FIELDS = _fields(
    "epoch step lr train_loss train_monitor_loss val_loss val_accuracy val_auc "
    "val_brier val_ece val_central_475_525_auc val_central_475_525_logloss "
    "val_central_475_525_pct_data gen_loss_gap gen_auc_gap gen_brier_gap "
    "gen_ece_gap train_mean_pred val_mean_pred"
)
EVAL_FIELDS = _fields("loss accuracy auc brier ece mean_pred positive_rate baseline_logloss n")
CENTRAL_METRICS = _fields("count pct_data auc logloss brier calibration accuracy")
HEADLINE_CENTRAL_BAND = "475_525"
LEGACY_HEADLINE_CENTRAL_BANDS = ("40_60",)
CENTRAL_BANDS = ("475_525", "45_55", "40_60")
BUCKET_ORDER = _fields("lt_35 35_40 40_45 45_50 50_55 55_60 60_65 gt_65")
BUCKET_LABELS = {
    "lt_35": "<0.35",
    "35_40": "0.35-0.40",
    "40_45": "0.40-0.45",
    "45_50": "0.45-0.50",
    "50_55": "0.50-0.55",
    "55_60": "0.55-0.60",
    "60_65": "0.60-0.65",
    "gt_65": ">0.65",
}
BUCKET_METRICS = _fields("count pct_data mean_pred actual_rate gap accuracy logloss")
PREDICTION_DISTRIBUTION_FIELDS = _fields("pred_std pred_p01 pred_p05 pred_p10 pred_p50 pred_p90 pred_p95 pred_p99")
PREDICTION_CONFIDENCE_FIELDS = _fields(
    "pred_gt_55_count pred_gt_55_accuracy pred_gt_60_count pred_gt_60_accuracy "
    "pred_gt_65_count pred_gt_65_accuracy pred_lt_45_count pred_lt_45_accuracy "
    "pred_lt_40_count pred_lt_40_accuracy pred_lt_35_count pred_lt_35_accuracy"
)
ATTENTION_FOCUS_FIELDS = _fields(
    "attention_diagnostic_samples attention_layers_observed attention_entropy_mean "
    "attention_effective_tokens_mean attention_max_prob_mean attention_max_prob_p95 "
    "attention_top5_mass_mean attention_head_similarity_mean "
    "attention_head_diversity_mean attention_token_utilization "
    "attention_ignored_token_frac attention_cls_mass attention_player_mass "
    "attention_interaction_mass attention_first_layer_entropy_mean "
    "attention_last_layer_entropy_mean attention_first_layer_effective_tokens_mean "
    "attention_last_layer_effective_tokens_mean attention_layer_entropy_mean_range "
    "attention_layer_effective_tokens_mean_range "
    "attention_layer_head_diversity_mean_range "
    "attention_layer_head_similarity_mean_range attention_drift_l2 "
    "attention_drift_cosine"
)


MetricsRow = dict[str, object]
Number = float | int
Point = tuple[float, float]


def load_metrics(path: Path = DEFAULT_METRICS_PATH) -> list[MetricsRow]:
    rows: list[MetricsRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            rows.append(row)
    return rows


def summarize_training_session(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    *,
    max_timeline_points: int = 24,
    recent_epochs: int = 10,
    max_movers: int = 12,
) -> dict[str, object]:
    rows = load_metrics(metrics_path)
    epoch_rows = _event_rows(rows, "epoch_end")
    train_step_rows = _event_rows(rows, "train_step")
    checkpoint_rows = _event_rows(rows, "checkpoint")
    test_rows = _event_rows(rows, "test")
    heavy_rows = [row for row in epoch_rows if _is_heavy_diagnostic_row(row)]

    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source": str(metrics_path),
        "run": _run_summary(rows, epoch_rows, heavy_rows),
        "model_evaluations": _model_evaluations(
            epoch_rows,
            checkpoint_rows,
            test_rows,
        ),
        "progress": _field_stats(
            epoch_rows,
            CORE_PROGRESS_FIELDS,
            x_key="epoch",
            recent_window=recent_epochs,
        ),
        "timeline": _timeline(epoch_rows, max_timeline_points),
        "prediction_diagnostics": _prediction_diagnostics(
            heavy_rows,
            test_rows,
            max_movers=max_movers,
        ),
        "attention_diagnostics": _attention_diagnostics(
            heavy_rows,
            test_rows,
            max_movers=max_movers,
        ),
        "signals": _signals(epoch_rows, heavy_rows, test_rows),
    }
    if train_step_rows:
        summary["optimization"] = _field_stats(
            train_step_rows,
            TRAIN_STEP_FIELDS,
            x_key="step",
            recent_window=max(3, min(20, len(train_step_rows))),
        )
    ready = _json_ready(summary)
    if not isinstance(ready, dict):
        raise TypeError("training-session summary did not produce a JSON object")
    return ready


def summary_json(
    summary: dict[str, object],
    *,
    pretty: bool = False,
) -> str:
    if pretty:
        return json.dumps(summary, indent=2, sort_keys=True)
    return json.dumps(summary, separators=(",", ":"), sort_keys=True)


def _event_rows(rows: Sequence[MetricsRow], event: str) -> list[MetricsRow]:
    return [row for row in rows if row.get("event") == event]


def _is_heavy_diagnostic_row(row: MetricsRow) -> bool:
    return any(
        "_pred_bucket_" in key
        or "_pred_central_" in key
        or "_attention_" in key
        or key.endswith("_pred_std")
        for key in row
    )


def _metric_number(value: object) -> Number | None:
    scalar = metric_scalar(value)
    if scalar is None:
        return None
    if isinstance(scalar, int):
        return scalar
    return metric_float(float(scalar))


def _numeric(value: object) -> float | None:
    scalar = metric_scalar(value)
    if scalar is None:
        return None
    return float(scalar)


def _emit_number(value: float) -> Number:
    rounded = metric_float(value)
    return int(rounded) if rounded.is_integer() else rounded


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): json_value
            for key, item in value.items()
            if (json_value := _json_ready(item)) is not None
        }
    if isinstance(value, list):
        return [
            json_value
            for item in value
            if (json_value := _json_ready(item)) is not None
        ]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return metric_float(value) if math.isfinite(value) else None
    return value


def _run_summary(
    rows: Sequence[MetricsRow],
    epoch_rows: Sequence[MetricsRow],
    heavy_rows: Sequence[MetricsRow],
) -> dict[str, object]:
    event_counts = Counter(str(row.get("event", "")) for row in rows)
    run_start = _first_event(rows, "run_start")
    data_ready = _first_event(rows, "data_ready")
    model_ready = _first_event(rows, "model_ready")
    last_row = rows[-1] if rows else {}

    summary: dict[str, object] = {
        "events": dict(sorted(event_counts.items())),
        "rows": len(rows),
        "elapsed_s": _metric_number(last_row.get("elapsed_s")),
        "diagnostic_epochs": _epoch_list(heavy_rows),
    }
    if epoch_rows:
        summary["epochs"] = {
            "count": len(epoch_rows),
            "first": _metric_number(epoch_rows[0].get("epoch")),
            "last": _metric_number(epoch_rows[-1].get("epoch")),
        }
    if run_start:
        summary["train_config"] = _copy_keys(
            run_start,
            (
                "device",
                "amp",
                "amp_dtype",
                "batch_size",
                "gradient_accumulation_steps",
                "effective_batch_size",
                "lr",
                "weight_decay",
                "adamw_betas",
                "compile_mode",
                "grad_clip",
                "target_min",
                "target_max",
                "attention_diagnostics_interval",
                "attention_diagnostics_batch_size",
                "attention_diagnostics_eval_samples",
                "train_monitor_samples",
                "tensorboard_dir",
            ),
        )
    if data_ready:
        summary["data"] = _copy_keys(
            data_ready,
            (
                "train_games",
                "val_games",
                "test_games",
                "train_monitor_games",
                "batches_per_epoch",
                "optimizer_steps_per_epoch",
            ),
        )
    if model_ready:
        summary["model"] = _copy_keys(
            model_ready,
            ("parameters", "model_config", "token_identity_encoding"),
        )
    return summary


def _first_event(rows: Sequence[MetricsRow], event: str) -> MetricsRow | None:
    for row in rows:
        if row.get("event") == event:
            return row
    return None


def _copy_keys(row: MetricsRow, keys: Iterable[str]) -> dict[str, object]:
    return {
        key: number if (number := _metric_number(row.get(key))) is not None else row[key]
        for key in keys
        if key in row
    }


def _field_numbers(
    row: MetricsRow,
    fields: Iterable[str],
    *,
    key_for: Callable[[str], str] = lambda field: field,
    out_key_for: Callable[[str], str] = lambda field: field,
) -> dict[str, object]:
    return {
        out_key_for(field): number
        for field in fields
        if (number := _metric_number(row.get(key_for(field)))) is not None
    }


def _epoch_list(rows: Sequence[MetricsRow]) -> dict[str, object]:
    epochs = [
        int(epoch)
        for row in rows
        if (epoch := _numeric(row.get("epoch"))) is not None
    ]
    if len(epochs) <= 16:
        return {"count": len(epochs), "epochs": epochs}
    return {
        "count": len(epochs),
        "first": epochs[0],
        "last": epochs[-1],
        "sample": _evenly_spaced_values(epochs, 16),
    }


def _evenly_spaced_values(values: Sequence[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return list(values)
    if limit <= 1:
        return [values[-1]]
    return [
        values[round(index * (len(values) - 1) / (limit - 1))]
        for index in range(limit)
    ]


def _field_stats(
    rows: Sequence[MetricsRow],
    fields: Sequence[str],
    *,
    x_key: str,
    recent_window: int,
) -> dict[str, object]:
    return {
        field: _describe_series(
            field,
            points,
            recent_window=recent_window,
            x_label=x_key,
        )
        for field in fields
        if (points := _series(rows, field, x_key))
    }


def _series(rows: Sequence[MetricsRow], key: str, x_key: str) -> list[Point]:
    points: list[Point] = []
    for index, row in enumerate(rows):
        y = _numeric(row.get(key))
        if y is None:
            continue
        x = _numeric(row.get(x_key))
        if x is None:
            x = float(index)
        points.append((x, y))
    return points


def _describe_series(
    key: str,
    points: Sequence[Point],
    *,
    recent_window: int,
    x_label: str,
) -> dict[str, object]:
    first_x, first_y = points[0]
    last_x, last_y = points[-1]
    values = [y for _, y in points]
    out: dict[str, object] = {
        "n": len(points),
        f"first_{x_label}": _emit_number(first_x),
        "first": _emit_number(first_y),
        f"last_{x_label}": _emit_number(last_x),
        "last": _emit_number(last_y),
    }
    if len(points) > 1:
        out["delta"] = _emit_number(last_y - first_y)
        if (slope := _slope(points)) is not None:
            out["slope"] = metric_float(slope)
        recent = list(points[-max(1, recent_window) :])
        recent_values = [y for _, y in recent]
        out["recent_mean"] = _emit_number(sum(recent_values) / len(recent_values))
        if len(recent) > 1:
            if (recent_slope := _slope(recent)) is not None:
                out["recent_slope"] = metric_float(recent_slope)
            out["recent_std"] = metric_float(_std(recent_values))

    direction = _metric_direction(key)
    if direction in {"min", "max"}:
        selector = min if direction == "min" else max
        best_x, best_y = selector(points, key=lambda point: point[1])
        out.update(
            {
                f"best_{x_label}": _emit_number(best_x),
                "best": _emit_number(best_y),
                "last_vs_best": _emit_number(
                    last_y - best_y if direction == "min" else best_y - last_y
                ),
            }
        )
    elif direction == "zero":
        best_x, best_y = min(points, key=lambda point: abs(point[1]))
        out.update(
            {
                f"closest_to_zero_{x_label}": _emit_number(best_x),
                "closest_to_zero": _emit_number(best_y),
                "last_abs": _emit_number(abs(last_y)),
            }
        )
    if values:
        out["min"] = _emit_number(min(values))
        out["max"] = _emit_number(max(values))
    return out


def _metric_direction(key: str) -> str | None:
    if key.startswith("gen_"):
        return "zero"
    lower_tokens = (
        "loss",
        "brier",
        "ece",
        "calibration",
        "ignored_token_frac",
        "head_similarity",
        "drift",
    )
    higher_tokens = (
        "accuracy",
        "auc",
        "samples_per_s",
        "head_diversity",
        "token_utilization",
    )
    if any(token in key for token in lower_tokens):
        return "min"
    if any(token in key for token in higher_tokens):
        return "max"
    return None


def _slope(points: Sequence[Point]) -> float | None:
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    if denom <= 0.0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denom


def _std(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _timeline(rows: Sequence[MetricsRow], max_points: int) -> list[dict[str, object]]:
    if not rows:
        return []
    indices = _timeline_indices(rows, max_points)
    return [_compact_row(rows[index], TIMELINE_FIELDS) for index in indices]


def _timeline_indices(rows: Sequence[MetricsRow], max_points: int) -> list[int]:
    if len(rows) <= max_points:
        return list(range(len(rows)))

    selected = {0, len(rows) - 1}
    for key, selector in (
        ("val_loss", min),
        ("val_auc", max),
        ("val_brier", min),
        ("val_ece", min),
    ):
        candidates = [
            (index, value)
            for index, row in enumerate(rows)
            if (value := _numeric(row.get(key))) is not None
        ]
        if candidates:
            selected.add(selector(candidates, key=lambda item: item[1])[0])

    limit = max(2, max_points)
    selected.update(
        round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)
    )

    while len(selected) > max_points:
        removable = sorted(selected - {0, len(rows) - 1})
        if not removable:
            break
        midpoint = len(rows) // 2
        selected.remove(max(removable, key=lambda index: abs(index - midpoint)))
    return sorted(selected)


def _compact_row(row: MetricsRow, fields: Sequence[str]) -> dict[str, object]:
    return _field_numbers(row, fields)


def _model_evaluations(
    epoch_rows: Sequence[MetricsRow],
    checkpoint_rows: Sequence[MetricsRow],
    test_rows: Sequence[MetricsRow],
) -> dict[str, object]:
    evaluations: dict[str, object] = {}
    if epoch_rows:
        last_epoch = epoch_rows[-1]
        evaluations["validation_last"] = _with_position(
            last_epoch,
            _extract_eval(last_epoch, "val"),
        )
        evaluations["train_monitor_last"] = _with_position(
            last_epoch,
            _extract_train_monitor_eval(last_epoch),
        )
        best_loss = _best_row(epoch_rows, "val_loss", min)
        best_auc = _best_row(epoch_rows, "val_auc", max)
        for key, row in (
            ("validation_best_loss", best_loss),
            ("validation_best_auc", best_auc),
        ):
            if row is not None:
                evaluations[key] = _with_position(row, _extract_eval(row, "val"))
        evaluations["generalization_last"] = _copy_prefix(last_epoch, "gen_")

    if checkpoint_rows:
        best_checkpoint = _best_row(checkpoint_rows, "val_loss", min)
        if best_checkpoint is not None:
            evaluations["checkpoint_best"] = _copy_keys(
                best_checkpoint,
                ("epoch", "step", "val_loss", "path"),
            )

    if test_rows:
        test_row = test_rows[-1]
        evaluations["test_final"] = _with_position(
            test_row,
            _extract_eval(test_row, "test"),
        )
        evaluations["test_generalization"] = _copy_prefix(test_row, "gen_")
    return evaluations


def _best_row(
    rows: Sequence[MetricsRow],
    key: str,
    selector: Any,
) -> MetricsRow | None:
    candidates = [
        (index, value)
        for index, row in enumerate(rows)
        if (value := _numeric(row.get(key))) is not None
    ]
    if not candidates:
        return None
    best_index, _ = selector(candidates, key=lambda item: item[1])
    return rows[best_index]


def _with_position(row: MetricsRow, payload: dict[str, object]) -> dict[str, object]:
    positioned = _copy_keys(row, ("epoch", "step", "time"))
    positioned.update(payload)
    return positioned


def _extract_eval(row: MetricsRow, split: str) -> dict[str, object]:
    fields = _field_numbers(row, EVAL_FIELDS, key_for=lambda metric: f"{split}_{metric}")
    fields.update(_compact_central_eval(row, split))
    return fields


def _extract_train_monitor_eval(row: MetricsRow) -> dict[str, object]:
    fields = _field_numbers(
        row,
        ("loss", "accuracy", "auc", "brier", "ece", "n"),
        key_for=lambda metric: f"train_monitor_{metric}",
    )
    fields.update(_compact_central_eval(row, "train"))
    return fields


def _compact_central_eval(row: MetricsRow, split: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    for band in (HEADLINE_CENTRAL_BAND, *LEGACY_HEADLINE_CENTRAL_BANDS):
        fields.update(
            _field_numbers(
                row,
                CENTRAL_METRICS,
                key_for=lambda metric, band=band: f"{split}_central_{band}_{metric}",
                out_key_for=lambda metric, band=band: f"central_{band}_{metric}",
            )
        )
    return fields


def _copy_prefix(row: MetricsRow, prefix: str) -> dict[str, object]:
    return {
        key: number
        for key in sorted(row)
        if key.startswith(prefix)
        if (number := _metric_number(row.get(key))) is not None
    }


def _prediction_diagnostics(
    heavy_rows: Sequence[MetricsRow],
    test_rows: Sequence[MetricsRow],
    *,
    max_movers: int,
) -> dict[str, object]:
    return _split_diagnostics(
        heavy_rows,
        test_rows,
        split_summary=_prediction_split_summary,
        mover_key="top_prediction_movers",
        mover_prefixes=("train_pred_", "val_pred_"),
        max_movers=max_movers,
    )


def _attention_diagnostics(
    heavy_rows: Sequence[MetricsRow],
    test_rows: Sequence[MetricsRow],
    *,
    max_movers: int,
) -> dict[str, object]:
    return _split_diagnostics(
        heavy_rows,
        test_rows,
        split_summary=_attention_split_fields,
        mover_key="top_attention_movers",
        mover_prefixes=("train_attention_", "val_attention_"),
        max_movers=max_movers,
    )


def _split_diagnostics(
    heavy_rows: Sequence[MetricsRow],
    test_rows: Sequence[MetricsRow],
    *,
    split_summary: Callable[[MetricsRow, str], dict[str, object]],
    mover_key: str,
    mover_prefixes: Sequence[str],
    max_movers: int,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "heavy_epoch_count": len(heavy_rows),
        "heavy_epochs": _epoch_list(heavy_rows),
    }
    if heavy_rows:
        diagnostics["latest_epoch"] = _metric_number(heavy_rows[-1].get("epoch"))
    splits = {
        split: fields
        for split, row in _split_sources(heavy_rows, test_rows)
        if (fields := split_summary(row, split))
    }
    if splits:
        diagnostics["splits"] = splits
    if len(heavy_rows) > 1:
        diagnostics[mover_key] = _top_field_deltas(
            heavy_rows,
            prefixes=mover_prefixes,
            limit=max_movers,
        )
    return diagnostics


def _split_sources(
    heavy_rows: Sequence[MetricsRow],
    test_rows: Sequence[MetricsRow],
) -> list[tuple[str, MetricsRow]]:
    sources: list[tuple[str, MetricsRow]] = []
    if heavy_rows:
        sources.extend((("train", heavy_rows[-1]), ("val", heavy_rows[-1])))
    if test_rows:
        sources.append(("test", test_rows[-1]))
    return sources


def _prediction_split_summary(row: MetricsRow, split: str) -> dict[str, object]:
    distribution = _prefixed_fields(row, split, PREDICTION_DISTRIBUTION_FIELDS)
    confidence = _prefixed_fields(row, split, PREDICTION_CONFIDENCE_FIELDS)
    central = _central_bands(row, split)
    buckets = _bucket_rows(row, split)

    summary: dict[str, object] = {}
    base = _copy_keys(
        row,
        (
            f"{split}_mean_pred",
            f"{split}_positive_rate",
            f"{split}_baseline_logloss",
        ),
    )
    if base:
        summary["base"] = {
            key.removeprefix(f"{split}_"): value for key, value in base.items()
        }
    for key, payload in (
        ("distribution", distribution),
        ("confidence", confidence),
        ("central_bands", central),
    ):
        if payload:
            summary[key] = payload
    if buckets:
        summary["bucket_rollup"] = _bucket_rollup(buckets)
        summary["buckets"] = buckets
    return summary


def _prefixed_fields(
    row: MetricsRow,
    split: str,
    fields: Sequence[str],
) -> dict[str, object]:
    return _field_numbers(
        row,
        fields,
        key_for=lambda field: f"{split}_{field}",
        out_key_for=lambda field: field.removeprefix("pred_"),
    )


def _central_bands(row: MetricsRow, split: str) -> dict[str, object]:
    bands: dict[str, object] = {}
    for band in CENTRAL_BANDS:
        fields = _field_numbers(
            row,
            CENTRAL_METRICS,
            key_for=lambda metric: f"{split}_pred_central_{band}_{metric}",
        )
        if fields:
            bands[band] = fields
    fallback = _compact_central_eval(row, split)
    for band in (HEADLINE_CENTRAL_BAND, *LEGACY_HEADLINE_CENTRAL_BANDS):
        prefix = f"central_{band}_"
        if band not in bands:
            fields = {
                key.removeprefix(prefix): value
                for key, value in fallback.items()
                if key.startswith(prefix)
            }
            if fields:
                bands[band] = fields
    return bands


def _bucket_rows(row: MetricsRow, split: str) -> list[dict[str, object]]:
    buckets: list[dict[str, object]] = []
    for bucket in BUCKET_ORDER:
        metrics = _field_numbers(
            row,
            BUCKET_METRICS,
            key_for=lambda metric: f"{split}_pred_bucket_{bucket}_{metric}",
        )
        if metrics:
            buckets.append({"bucket": BUCKET_LABELS[bucket], **metrics})
    return buckets


def _bucket_rollup(buckets: Sequence[dict[str, object]]) -> dict[str, object]:
    pct_by_bucket = {
        str(row.get("bucket")): _numeric(row.get("pct_data")) or 0.0 for row in buckets
    }

    def pct(*labels: str) -> Number:
        return _emit_number(sum(pct_by_bucket.get(label, 0.0) for label in labels))

    worst = sorted(
        (
            row
            for row in buckets
            if _numeric(row.get("gap")) is not None
            and (_numeric(row.get("pct_data")) or 0.0) > 0.0
        ),
        key=lambda row: abs(_numeric(row.get("gap")) or 0.0),
        reverse=True,
    )
    return {
        "confident_pct": pct("<0.35", ">0.65"),
        "wide_40_60_pct": pct(
            "0.40-0.45",
            "0.45-0.50",
            "0.50-0.55",
            "0.55-0.60",
        ),
        "mid_45_55_pct": pct("0.45-0.50", "0.50-0.55"),
        "worst_gap_buckets": [
            _copy_existing(
                row,
                ("bucket", "pct_data", "mean_pred", "actual_rate", "gap"),
            )
            for row in worst[:3]
        ],
    }


def _copy_existing(row: dict[str, object], keys: Sequence[str]) -> dict[str, object]:
    return {key: row[key] for key in keys if key in row}


def _attention_split_fields(row: MetricsRow, split: str) -> dict[str, object]:
    fields = _field_numbers(
        row,
        ATTENTION_FOCUS_FIELDS,
        key_for=lambda field: f"{split}_{field}",
        out_key_for=lambda field: field.removeprefix("attention_"),
    )
    present_attention_fields = [
        key for key in row if key.startswith(f"{split}_attention_")
    ]
    if present_attention_fields:
        fields["scalar_field_count"] = sum(
            1
            for key in present_attention_fields
            if _metric_number(row.get(key)) is not None
        )
    return fields


def _top_field_deltas(
    rows: Sequence[MetricsRow],
    *,
    prefixes: Sequence[str],
    limit: int,
) -> list[dict[str, object]]:
    if len(rows) < 2:
        return []
    first = rows[0]
    last = rows[-1]
    fields = sorted(
        key
        for key in set(first) | set(last)
        if any(key.startswith(prefix) for prefix in prefixes)
    )
    deltas: list[dict[str, object]] = []
    for field in fields:
        first_value = _numeric(first.get(field))
        last_value = _numeric(last.get(field))
        if first_value is None or last_value is None:
            continue
        delta = last_value - first_value
        if delta == 0.0:
            continue
        deltas.append(
            {
                "field": field,
                "first": _emit_number(first_value),
                "last": _emit_number(last_value),
                "delta": _emit_number(delta),
            }
        )
    return sorted(deltas, key=_abs_delta_field, reverse=True)[:limit]


def _abs_delta_field(row: dict[str, object]) -> float:
    value = row.get("delta")
    return abs(float(value)) if isinstance(value, int | float) else 0.0


def _signals(
    epoch_rows: Sequence[MetricsRow],
    heavy_rows: Sequence[MetricsRow],
    test_rows: Sequence[MetricsRow],
) -> list[dict[str, object]]:
    signals: list[dict[str, object]] = []
    latest_eval = (
        test_rows[-1] if test_rows else (epoch_rows[-1] if epoch_rows else None)
    )
    split = "test" if test_rows else "val"

    if epoch_rows:
        latest = epoch_rows[-1]
        best_val_loss_row = _best_row(epoch_rows, "val_loss", min)
        latest_val_loss = _numeric(latest.get("val_loss"))
        best_val_loss = (
            _numeric(best_val_loss_row.get("val_loss"))
            if best_val_loss_row is not None
            else None
        )
        if latest_val_loss is not None and best_val_loss is not None:
            drift = latest_val_loss - best_val_loss
            if drift > 0.001:
                signals.append(
                    {
                        "name": "validation_loss_above_best",
                        "value": _emit_number(drift),
                        "latest_epoch": _metric_number(latest.get("epoch")),
                        "best_epoch": _metric_number(best_val_loss_row.get("epoch"))
                        if best_val_loss_row is not None
                        else None,
                    }
                )
        for key, threshold in (
            ("gen_loss_gap", 0.005),
            ("gen_auc_gap", 0.02),
            ("gen_brier_gap", 0.002),
        ):
            _append_threshold_signal(
                signals,
                name=f"{key}_elevated",
                value=_numeric(latest.get(key)),
                threshold=threshold,
            )

    if latest_eval is not None:
        mean_pred = _numeric(latest_eval.get(f"{split}_mean_pred"))
        positive_rate = _numeric(latest_eval.get(f"{split}_positive_rate"))
        if mean_pred is not None and positive_rate is not None:
            bias = mean_pred - positive_rate
            if abs(bias) > 0.01:
                signals.append(
                    {
                        "name": "prediction_rate_bias",
                        "split": split,
                        "value": _emit_number(bias),
                        "mean_pred": _emit_number(mean_pred),
                        "positive_rate": _emit_number(positive_rate),
                    }
                )
        ece = _numeric(latest_eval.get(f"{split}_ece"))
        _append_threshold_signal(
            signals,
            name="ece_elevated",
            value=ece,
            threshold=0.02,
            split=split,
        )

    attention_row = (
        test_rows[-1] if test_rows else (heavy_rows[-1] if heavy_rows else None)
    )
    attention_split = "test" if test_rows else "val"
    if attention_row is not None:
        ignored = _numeric(
            attention_row.get(f"{attention_split}_attention_ignored_token_frac")
        )
        similarity = _numeric(
            attention_row.get(f"{attention_split}_attention_head_similarity_mean")
        )
        _append_threshold_signal(
            signals,
            name="attention_ignored_token_frac_elevated",
            value=ignored,
            threshold=0.25,
            split=attention_split,
        )
        _append_threshold_signal(
            signals,
            name="attention_head_similarity_high",
            value=similarity,
            threshold=0.85,
            split=attention_split,
        )
    return signals


def _append_threshold_signal(
    signals: list[dict[str, object]],
    *,
    name: str,
    value: float | None,
    threshold: float,
    split: str | None = None,
) -> None:
    if value is None or value <= threshold:
        return
    signal: dict[str, object] = {
        "name": name,
        "value": _emit_number(value),
        "threshold": threshold,
    }
    if split is not None:
        signal["split"] = split
    signals.append(signal)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise ML metrics JSONL into compact LLM-friendly JSON.",
    )
    parser.add_argument(
        "metrics_path",
        nargs="?",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Metrics JSONL path (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional file path for the summary JSON. Defaults to stdout.",
    )
    parser.add_argument(
        "--max-timeline-points",
        type=int,
        default=24,
        help="Maximum downsampled epoch points to include.",
    )
    parser.add_argument(
        "--recent-epochs",
        type=int,
        default=10,
        help="Recent epoch window used for trend stats.",
    )
    parser.add_argument(
        "--max-movers",
        type=int,
        default=12,
        help="Maximum prediction/attention diagnostic movers to include.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON for humans. Compact JSON is the default.",
    )
    args = parser.parse_args(argv)

    summary = summarize_training_session(
        args.metrics_path,
        max_timeline_points=args.max_timeline_points,
        recent_epochs=args.recent_epochs,
        max_movers=args.max_movers,
    )
    text = summary_json(summary, pretty=args.pretty)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(f"{text}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
