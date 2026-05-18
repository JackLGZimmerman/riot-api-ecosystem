from __future__ import annotations

import logging
import os
import re
import socket
import time
from pathlib import Path
from typing import Any

from app.ml.utils.metrics import metric_scalar

logger = logging.getLogger(__name__)

# Auto-routes relation diagnostics (see app.ml.utils.attention_diagnostics)
# into dedicated TensorBoard namespaces without enumerating each tag:
#   train_attention_relation_self        -> attention_relation/train_self
#   val_attention_head_2_bot_duo_ally    -> attention_head_2/val_bot_duo_ally
# Pattern requires digits after `head_` so it never collides with
# attention_head_diversity_mean or attention_drift_cosine.
_RELATION_TAG_RE = re.compile(
    r"^(train|val|test)_attention_(?:relation_([a-z][a-z0-9_]*)"
    r"|head_(\d+)_([a-z][a-z0-9_]*))$"
)


def _relation_tag(event: str, field: str) -> str | None:
    if event not in ("epoch_end", "test"):
        return None
    match = _RELATION_TAG_RE.match(field)
    if match is None:
        return None
    split = match.group(1)
    global_name = match.group(2)
    if global_name is not None:
        return f"attention_relation/{split}_{global_name}"
    return f"attention_head_{match.group(3)}/{split}_{match.group(4)}"

# Curated TensorBoard charts. TensorBoard is kept to the signals that are most
# decision-relevant for iterating on the set transformer: train-vs-val quality,
# the 0.475-0.525 central band, generalization gaps, calibration, and a small
# attention subset. The full JSONL still carries every field; set
# `tensorboard_raw_mirror=True` to additionally mirror raw/<event>/<field>.
TENSORBOARD_SCALAR_TAGS: dict[tuple[str, str], str] = {
    # Per-step training health.
    ("train_step", "train_loss"): "loss/train_step",
    ("train_step", "batch_loss"): "loss/batch",
    ("train_step", "lr"): "optimization/lr",
    ("train_step", "grad_norm"): "optimization/grad_norm",
    ("train_step", "samples_per_s"): "throughput/samples_per_s",
    # Core loss curves. `loss/train_objective` is the epoch-mean optimization
    # objective (smoothed targets); `loss/train_heldin` and `loss/val` are
    # eval-path hard-label losses, so only those two are directly comparable.
    ("epoch_end", "train_loss"): "loss/train_objective",
    ("epoch_end", "train_monitor_loss"): "loss/train_heldin",
    ("epoch_end", "val_loss"): "loss/val",
    # Quality: validation alone. Train-side accuracy/auc live inside gen_*_gap.
    ("epoch_end", "val_accuracy"): "quality/val_accuracy",
    ("epoch_end", "val_auc"): "quality/val_auc",
    # Central 0.475-0.525 prediction band: the critical decision region.
    ("epoch_end", "val_central_475_525_auc"): "central_475_525/val_auc",
    ("epoch_end", "val_central_475_525_logloss"): "central_475_525/val_logloss",
    ("epoch_end", "val_central_475_525_calibration"): (
        "central_475_525/val_calibration"
    ),
    # Generalization gaps: positive == train beats held-out (overfitting).
    ("epoch_end", "gen_loss_gap"): "generalization/loss_gap",
    ("epoch_end", "gen_accuracy_gap"): "generalization/accuracy_gap",
    ("epoch_end", "gen_auc_gap"): "generalization/auc_gap",
    ("epoch_end", "gen_central_475_525_auc_gap"): (
        "generalization/central_475_525_auc_gap"
    ),
    # Calibration.
    ("epoch_end", "val_brier"): "calibration/val_brier",
    ("epoch_end", "val_ece"): "calibration/val_ece",
    # Attention (heavy cadence): collapse, head degeneration, wasted capacity,
    # right-token-family mass, epoch-over-epoch drift.
    ("epoch_end", "train_attention_entropy_mean"): "attention/train_entropy",
    ("epoch_end", "val_attention_entropy_mean"): "attention/val_entropy",
    ("epoch_end", "train_attention_head_diversity_mean"): (
        "attention/train_head_diversity"
    ),
    ("epoch_end", "val_attention_head_diversity_mean"): (
        "attention/val_head_diversity"
    ),
    ("epoch_end", "train_attention_ignored_token_frac"): (
        "attention/train_ignored_token_frac"
    ),
    ("epoch_end", "val_attention_ignored_token_frac"): (
        "attention/val_ignored_token_frac"
    ),
    ("epoch_end", "train_attention_player_mass"): "attention/train_player_mass",
    ("epoch_end", "val_attention_player_mass"): "attention/val_player_mass",
    ("epoch_end", "train_attention_drift_cosine"): "attention/train_drift_cosine",
    # Best checkpoint + final test.
    ("checkpoint", "val_loss"): "best/val_loss",
    ("test", "test_loss"): "loss/test",
    ("test", "test_accuracy"): "quality/test_accuracy",
    ("test", "test_auc"): "quality/test_auc",
    ("test", "test_brier"): "calibration/test_brier",
    ("test", "test_ece"): "calibration/test_ece",
    ("test", "test_central_475_525_auc"): "central_475_525/test_auc",
    ("test", "test_central_475_525_logloss"): "central_475_525/test_logloss",
    ("test", "test_central_475_525_calibration"): (
        "central_475_525/test_calibration"
    ),
    # Final blue/red swap symmetry on the test split (lower = more symmetric).
    ("test_symmetry", "symmetry_abs_delta_mean"): "symmetry/test_mean",
    ("test_symmetry", "symmetry_abs_delta_p50"): "symmetry/test_p50",
    ("test_symmetry", "symmetry_abs_delta_p95"): "symmetry/test_p95",
    ("test_symmetry", "symmetry_abs_delta_max"): "symmetry/test_max",
}


class TensorBoardMetricWriter:
    def __init__(
        self,
        metrics_dir: Path,
        metrics_file: str,
        tensorboard_dir: str | None,
        raw_mirror: bool = False,
        run_name: str | None = None,
    ) -> None:
        self.path: Path | None = None
        self._writer: Any = None
        self._event_cls: Any = None
        self._summary_cls: Any = None
        self._raw_mirror = bool(raw_mirror)
        if not tensorboard_dir:
            return

        try:
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.summary.writer.record_writer import RecordWriter
        except ImportError as exc:
            logger.warning(
                "TensorBoard is unavailable (%s); continuing with JSONL live metrics only",
                exc,
            )
            return

        resolved_run_name = run_name or (
            f"{Path(metrics_file).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        # `metrics_dir / tensorboard_dir` collapses to `tensorboard_dir` when
        # the latter is absolute, so sweeps can pin a shared TB root.
        self.path = metrics_dir / tensorboard_dir / resolved_run_name
        self.path.mkdir(parents=True, exist_ok=True)
        event_path = self.path / (
            f"events.out.tfevents.{int(time.time())}."
            f"{socket.gethostname()}.{os.getpid()}.{time.time_ns()}"
        )
        event_cls: Any = Event
        summary_cls: Any = Summary
        self._writer = RecordWriter(event_path.open("wb"))
        self._event_cls = event_cls
        self._summary_cls = summary_cls
        self._writer.write(
            event_cls(wall_time=time.time(), file_version="brain.Event:2")
            .SerializeToString()
        )
        self._writer.flush()

    def record(
        self,
        event: str,
        fields: dict[str, object],
        row: dict[str, object],
    ) -> None:
        if self._writer is None:
            return

        step_field = metric_scalar(fields.get("step"))
        epoch_field = metric_scalar(fields.get("epoch"))
        global_step = int(step_field or epoch_field or 0)
        values = []
        for key, value in row.items():
            scalar = metric_scalar(value)
            if scalar is None:
                continue

            tag = TENSORBOARD_SCALAR_TAGS.get((event, key))
            if tag is None:
                tag = _relation_tag(event, key)
            if tag is not None:
                values.append(
                    self._summary_cls.Value(tag=tag, simple_value=float(scalar))
                )
            if self._raw_mirror:
                values.append(
                    self._summary_cls.Value(
                        tag=f"raw/{event}/{key}",
                        simple_value=float(scalar),
                    )
                )
        if values:
            self._writer.write(
                self._event_cls(
                    wall_time=time.time(),
                    step=global_step,
                    summary=self._summary_cls(value=values),
                ).SerializeToString()
            )
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
