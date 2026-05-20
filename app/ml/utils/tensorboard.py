from __future__ import annotations

import logging
import os
import socket
import time
from pathlib import Path
from typing import Any

from app.ml.utils.metrics import metric_scalar

logger = logging.getLogger(__name__)

# Curated TensorBoard charts. The full JSONL still carries every field; set
# `tensorboard_raw_mirror=True` to additionally mirror raw/<event>/<field>.
TENSORBOARD_SCALAR_TAGS: dict[tuple[str, str], str] = {
    # Per-step training health.
    ("train_step", "train_loss"): "loss/train_step",
    ("train_step", "batch_loss"): "loss/batch",
    ("train_step", "lr"): "optimization/lr",
    ("train_step", "grad_norm"): "optimization/grad_norm",
    ("train_step", "samples_per_s"): "throughput/samples_per_s",
    # Core loss curves. `loss/train_objective` is the epoch-mean optimization
    # objective (smoothed targets); `loss/val` is the eval-path hard-label loss.
    ("epoch_end", "train_loss"): "loss/train_objective",
    ("epoch_end", "val_loss"): "loss/val",
    ("epoch_end", "val_accuracy"): "quality/val_accuracy",
    ("epoch_end", "val_auc"): "quality/val_auc",
    ("epoch_end", "val_brier"): "calibration/val_brier",
    ("epoch_end", "val_ece"): "calibration/val_ece",
    # Best checkpoint + final test.
    ("checkpoint", "val_loss"): "best/val_loss",
    ("test", "test_loss"): "loss/test",
    ("test", "test_accuracy"): "quality/test_accuracy",
    ("test", "test_auc"): "quality/test_auc",
    ("test", "test_brier"): "calibration/test_brier",
    ("test", "test_ece"): "calibration/test_ece",
}


def _raw_scalar_items(value: object, prefix: str) -> list[tuple[str, float]]:
    scalar = metric_scalar(value)
    if scalar is not None:
        return [(prefix, float(scalar))]
    if isinstance(value, dict):
        items: list[tuple[str, float]] = []
        for key, nested in value.items():
            items.extend(_raw_scalar_items(nested, f"{prefix}/{key}"))
        return items
    if isinstance(value, (list, tuple)):
        items = []
        for idx, nested in enumerate(value):
            items.extend(_raw_scalar_items(nested, f"{prefix}/{idx}"))
        return items
    return []


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
            if self._raw_mirror:
                values.extend(
                    self._summary_cls.Value(
                        tag=f"raw/{event}/{path}",
                        simple_value=float(nested_scalar),
                    )
                    for path, nested_scalar in _raw_scalar_items(value, key)
                )

            scalar = metric_scalar(value)
            if scalar is None:
                continue

            tag = TENSORBOARD_SCALAR_TAGS.get((event, key))
            if tag is not None:
                values.append(
                    self._summary_cls.Value(tag=tag, simple_value=float(scalar))
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
