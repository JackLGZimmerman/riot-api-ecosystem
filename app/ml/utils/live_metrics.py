from __future__ import annotations

import json
import time
from pathlib import Path

from app.ml.utils.metrics import metric_value
from app.ml.utils.tensorboard import TensorBoardMetricWriter


class LiveMetrics:
    """Append-only JSONL metrics with optional TensorBoard mirroring."""

    def __init__(
        self,
        metrics_dir: Path,
        metrics_file: str,
        latest_file: str,
        tensorboard_dir: str | None,
        tensorboard_raw_mirror: bool = False,
    ) -> None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self.path = metrics_dir / metrics_file
        self.latest_path = metrics_dir / latest_file
        self.tensorboard = TensorBoardMetricWriter(
            metrics_dir,
            metrics_file,
            tensorboard_dir,
            raw_mirror=tensorboard_raw_mirror,
        )
        self.tensorboard_path = self.tensorboard.path
        self._t0 = time.perf_counter()
        self._fh = self.path.open("w", encoding="utf-8")

    def record(self, event: str, **fields: object) -> None:
        row = {
            "event": event,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.perf_counter() - self._t0, 3),
            **fields,
        }
        row = {key: metric_value(value) for key, value in row.items()}
        line = json.dumps(row, sort_keys=True)
        self._fh.write(f"{line}\n")
        self._fh.flush()
        self.latest_path.write_text(
            json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
        )
        self.tensorboard.record(event, fields, row)

    def close(self) -> None:
        self._fh.close()
        self.tensorboard.close()
