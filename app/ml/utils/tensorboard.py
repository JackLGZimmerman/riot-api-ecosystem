from __future__ import annotations

import logging
import time
from pathlib import Path

from app.ml.utils.metrics import metric_scalar

logger = logging.getLogger(__name__)

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
    # objective (smoothed targets + any sample weights); `loss/train_heldin`
    # and `loss/val` are eval-path hard-label losses, so only those two are
    # directly comparable. `baseline_logloss` is a per-run constant and stays
    # in the JSONL only - a flat TensorBoard curve adds nothing.
    ("epoch_end", "train_loss"): "loss/train_objective",
    ("epoch_end", "train_monitor_loss"): "loss/train_heldin",
    ("epoch_end", "val_loss"): "loss/val",
    # Quality: train vs val, the headline overfitting read.
    ("epoch_end", "train_monitor_accuracy"): "quality/train_accuracy",
    ("epoch_end", "val_accuracy"): "quality/val_accuracy",
    ("epoch_end", "train_monitor_auc"): "quality/train_auc",
    ("epoch_end", "val_auc"): "quality/val_auc",
    # Central 0.475-0.525 prediction band: the critical decision region. The
    # `central_475_525/` prefix names the band explicitly; `val_data_pct` is the
    # share of validation predictions that land inside the band.
    ("epoch_end", "train_central_475_525_auc"): "central_475_525/train_auc",
    ("epoch_end", "val_central_475_525_auc"): "central_475_525/val_auc",
    ("epoch_end", "train_central_475_525_logloss"): "central_475_525/train_logloss",
    ("epoch_end", "val_central_475_525_logloss"): "central_475_525/val_logloss",
    ("epoch_end", "train_central_475_525_accuracy"): "central_475_525/train_accuracy",
    ("epoch_end", "val_central_475_525_accuracy"): "central_475_525/val_accuracy",
    ("epoch_end", "val_central_475_525_pct_data"): "central_475_525/val_data_pct",
    # Generalization gaps: positive == train beats held-out (overfitting).
    ("epoch_end", "gen_loss_gap"): "generalization/loss_gap",
    ("epoch_end", "gen_accuracy_gap"): "generalization/accuracy_gap",
    ("epoch_end", "gen_auc_gap"): "generalization/auc_gap",
    ("epoch_end", "gen_brier_gap"): "generalization/brier_gap",
    ("epoch_end", "gen_central_475_525_logloss_gap"): (
        "generalization/central_475_525_logloss_gap"
    ),
    ("epoch_end", "gen_central_475_525_auc_gap"): (
        "generalization/central_475_525_auc_gap"
    ),
    # Calibration.
    ("epoch_end", "train_monitor_brier"): "calibration/train_brier",
    ("epoch_end", "val_brier"): "calibration/val_brier",
    ("epoch_end", "train_monitor_ece"): "calibration/train_ece",
    ("epoch_end", "val_ece"): "calibration/val_ece",
    # Prediction distribution health. The split label base rates
    # (train/val positive_rate) are per-run constants and stay in the JSONL
    # only - a flat TensorBoard curve adds nothing.
    ("epoch_end", "train_mean_pred"): "predictions/train_mean_pred",
    ("epoch_end", "val_mean_pred"): "predictions/val_mean_pred",
    # Attention: the curated subset that drives architecture decisions.
    ("epoch_end", "train_attention_entropy_mean"): "attention/train_entropy",
    ("epoch_end", "val_attention_entropy_mean"): "attention/val_entropy",
    ("epoch_end", "train_attention_effective_tokens_mean"): (
        "attention/train_effective_tokens"
    ),
    ("epoch_end", "val_attention_effective_tokens_mean"): (
        "attention/val_effective_tokens"
    ),
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
    ("epoch_end", "train_attention_layer_entropy_mean_range"): (
        "attention/train_layer_entropy_range"
    ),
    ("epoch_end", "val_attention_layer_entropy_mean_range"): (
        "attention/val_layer_entropy_range"
    ),
    ("epoch_end", "train_attention_drift_l2"): "attention/train_drift_l2",
    ("epoch_end", "val_attention_drift_l2"): "attention/val_drift_l2",
    # Best checkpoint + final test.
    ("checkpoint", "val_loss"): "best/val_loss",
    ("test", "test_loss"): "loss/test",
    ("test", "test_accuracy"): "quality/test_accuracy",
    ("test", "test_auc"): "quality/test_auc",
    ("test", "test_brier"): "calibration/test_brier",
    ("test", "test_ece"): "calibration/test_ece",
    ("test", "test_central_475_525_auc"): "central_475_525/test_auc",
    ("test", "test_central_475_525_logloss"): "central_475_525/test_logloss",
    ("test", "test_mean_pred"): "predictions/test_mean_pred",
    ("test", "test_positive_rate"): "predictions/test_positive_rate",
}


class TensorBoardMetricWriter:
    def __init__(
        self,
        metrics_dir: Path,
        metrics_file: str,
        tensorboard_dir: str | None,
        raw_mirror: bool = False,
    ) -> None:
        self.path: Path | None = None
        self._writer = None
        self._raw_mirror = bool(raw_mirror)
        if not tensorboard_dir:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            logger.warning(
                "TensorBoard is unavailable (%s); continuing with JSONL live metrics only",
                exc,
            )
            return

        run_name = f"{Path(metrics_file).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.path = metrics_dir / tensorboard_dir / run_name
        self._writer = SummaryWriter(log_dir=str(self.path))

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
        for key, value in row.items():
            scalar = metric_scalar(value)
            if scalar is None:
                continue

            tag = TENSORBOARD_SCALAR_TAGS.get((event, key))
            if tag is not None:
                self._writer.add_scalar(tag, float(scalar), global_step)
            if self._raw_mirror:
                self._writer.add_scalar(
                    f"raw/{event}/{key}", float(scalar), global_step
                )
        self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
