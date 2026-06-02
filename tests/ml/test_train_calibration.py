from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from app.ml.config import TrainConfig
from app.ml.dataset import SplitData
from app.ml.hgnn_model import HGNNConfig
from app.ml.train import (
    CHECKPOINT_METRICS,
    CONTEXT_SUPPORT_RISK_BUCKETS,
    PRIOR_1VX_SUPPORT_RISK_BUCKETS,
    _auc_ranking_loss,
    _checkpoint_score,
    _context_auxiliary_loss,
    _context_residual_metrics,
    _context_support_temperature_report,
    _drop_unused_model_arrays,
    _fit_temperature,
    _identity_context_support_bucket_ids,
    _prior_1vx_support_bucket_ids,
    _select_threshold,
    _sigmoid_np,
    _support_bucket_metrics,
    _threshold_accuracy,
    _validate_train_config,
)


def _split(labels: np.ndarray, support: np.ndarray | None) -> SplitData:
    n = int(labels.size)
    return SplitData(
        win_rate=np.zeros((n, 10), dtype=np.float32),
        matchup_1v1=np.zeros((n, 25), dtype=np.float32),
        synergy_2vx=np.zeros((n, 20), dtype=np.float32),
        p1_cnt=np.zeros((n, 10), dtype=np.float32),
        m1v1_cnt=np.zeros((n, 25), dtype=np.float32),
        s2vx_cnt=np.zeros((n, 20), dtype=np.float32),
        blue_win=labels.astype(np.float64, copy=False),
        identity_context_support=support,
    )


def test_temperature_scaling_is_report_only_and_keeps_raw_threshold_path() -> None:
    logits = np.array([-4.0, -2.0, -0.5, 0.5, 2.0, 4.0])
    labels = np.array([0, 0, 1, 0, 1, 1], dtype=np.float64)
    raw = _sigmoid_np(logits)
    temperature = _fit_temperature(logits, labels)
    calibrated = _sigmoid_np(logits, temperature=temperature)
    threshold, threshold_acc = _select_threshold(raw, labels)

    assert temperature > 0.0
    assert not np.allclose(raw, calibrated)
    assert threshold_acc == _threshold_accuracy(raw, labels, threshold)
    assert np.allclose(raw, _sigmoid_np(logits))


def test_threshold_and_temperature_are_fit_from_validation_arrays_only() -> None:
    val_logits = np.array([-3.0, -1.0, 1.0, 3.0])
    val_labels = np.array([0, 0, 1, 1], dtype=np.float64)
    test_labels = 1.0 - val_labels

    val_raw = _sigmoid_np(val_logits)
    threshold, threshold_acc = _select_threshold(val_raw, val_labels)
    temperature = _fit_temperature(val_logits, val_labels)

    assert threshold_acc == 1.0
    assert _threshold_accuracy(val_raw, test_labels, threshold) == 0.0
    assert temperature == _fit_temperature(val_logits, val_labels)


def test_identity_context_support_bucket_metrics_expose_context_gate_risk() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.float64)
    support = np.array(
        [
            [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
            [12, 14, 16, 18, 20, 22, 24, 26, 28, 29],
            [60, 80, 100, 120, 140, 160, 180, 190, 195, 199],
            [250, 300, 350, 400, 450, 500, 550, 600, 650, 700],
        ],
        dtype=np.float32,
    )
    split = _split(labels, support)
    ids = _identity_context_support_bucket_ids(split)
    assert [CONTEXT_SUPPORT_RISK_BUCKETS[int(i)] for i in ids] == [
        "zero_player",
        "min_1_29",
        "mean_30_199",
        "mean_200_plus",
    ]

    metrics = _support_bucket_metrics(np.array([0.8, 0.7, 0.6, 0.4]), split)
    risk = metrics["identity_context_support"]["risk_bucket"]
    assert risk["zero_player"]["n"] == 1
    assert risk["mean_200_plus"]["n"] == 1
    assert risk["zero_player"]["calibration_gap"] == 0.8


def test_prior_1vx_support_bucket_metrics_expose_variance_ablation_risk() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.float64)
    split = _split(labels, support=None)
    split.p1_cnt[:] = np.array(
        [
            [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
            [1, 2, 3, 4, 4, 9, 9, 9, 9, 9],
            [5, 6, 10, 20, 40, 50, 60, 70, 80, 90],
            [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        ],
        dtype=np.float32,
    )

    ids = _prior_1vx_support_bucket_ids(split)
    assert [PRIOR_1VX_SUPPORT_RISK_BUCKETS[int(i)] for i in ids] == [
        "zero_player",
        "min_1_4",
        "min_5_49",
        "min_50_plus",
    ]

    metrics = _support_bucket_metrics(np.array([0.8, 0.7, 0.6, 0.4]), split)
    risk = metrics["prior_1vx_support"]["risk_bucket"]
    assert risk["zero_player"]["n"] == 1
    assert risk["min_50_plus"]["n"] == 1
    assert risk["zero_player"]["calibration_gap"] == 0.8
    assert "identity_context_support" not in metrics


def test_default_model_config_drops_relationship_tables_before_tensor_cache() -> None:
    split = _split(
        np.array([0, 1], dtype=np.float64),
        np.full((2, 10), 100.0, dtype=np.float32),
    )

    dropped = _drop_unused_model_arrays(split, HGNNConfig())
    assert dropped.matchup_1v1 is None
    assert dropped.synergy_2vx is None
    assert dropped.m1v1_cnt is None
    assert dropped.s2vx_cnt is None

    kept = _drop_unused_model_arrays(
        split,
        HGNNConfig(use_relationship_integrations=True),
    )
    assert kept.matchup_1v1 is split.matchup_1v1
    assert kept.synergy_2vx is split.synergy_2vx


def test_context_support_temperature_report_fits_validation_buckets_only() -> None:
    val_logits = np.array([-3.0, -2.0, 2.0, 3.0, -1.0, -0.5, 0.5, 1.0])
    val_labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float64)
    test_logits = np.array([-4.0, -3.0, -2.0, 2.0, 3.0, 4.0])
    test_labels = 1.0 - np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    zero_support = np.zeros((4, 10), dtype=np.float32)
    high_support = np.full((4, 10), 300.0, dtype=np.float32)
    val_split = _split(val_labels, np.vstack([zero_support, high_support]))
    train_split = _split(val_labels, np.vstack([zero_support, high_support]))
    test_split = _split(
        test_labels,
        np.vstack([
            np.full((2, 10), 300.0, dtype=np.float32),
            np.zeros((4, 10), dtype=np.float32),
        ]),
    )
    raw_test = _sigmoid_np(test_logits).copy()

    report, scaled = _context_support_temperature_report(
        {"train": val_logits, "val": val_logits, "test": test_logits},
        {"train": train_split, "val": val_split, "test": test_split},
        min_bucket_size=4,
    )

    buckets = {str(row["bucket"]): row for row in report["buckets"]}
    assert report["fit_split"] == "val"
    assert report["report_only"] is True
    assert buckets["zero_player"]["n_val"] == 4
    assert buckets["mean_200_plus"]["n_val"] == 4
    assert buckets["zero_player"]["fit_source"] == "bucket_val"
    assert scaled["test"]["n"] == test_labels.size
    assert "identity_context_support" in scaled["test"]
    assert np.allclose(raw_test, _sigmoid_np(test_logits))


def test_context_support_calibration_min_bucket_fails_early() -> None:
    with pytest.raises(ValueError, match="context_support_calibration_min_bucket"):
        _validate_train_config(TrainConfig(context_support_calibration_min_bucket=0))


def test_context_auxiliary_loss_weight_fails_early_when_negative() -> None:
    with pytest.raises(ValueError, match="context_auxiliary_loss_weight"):
        _validate_train_config(TrainConfig(context_auxiliary_loss_weight=-0.1))


def test_auc_ranking_loss_config_fails_early_when_invalid() -> None:
    with pytest.raises(ValueError, match="auc_ranking_loss_weight"):
        _validate_train_config(TrainConfig(auc_ranking_loss_weight=-0.1))
    with pytest.raises(ValueError, match="auc_ranking_loss_pairs"):
        _validate_train_config(TrainConfig(auc_ranking_loss_pairs=0))


def test_checkpoint_metric_fails_early_when_unknown() -> None:
    with pytest.raises(ValueError, match="checkpoint_metric"):
        _validate_train_config(TrainConfig(checkpoint_metric="test_auc"))


def test_calibration_aware_checkpoint_metric_penalizes_ece() -> None:
    assert "val_nll_ece" in CHECKPOINT_METRICS

    score = _checkpoint_score(
        "val_nll_ece",
        val_metrics={
            "accuracy": 0.55,
            "auc": 0.60,
            "nll": 0.67,
            "ece": 0.03,
        },
        val_threshold_accuracy=0.58,
    )

    assert score == pytest.approx(-0.70)


def test_context_auxiliary_loss_detaches_base_logit() -> None:
    base = torch.tensor([0.2, -0.1], requires_grad=True)
    context = torch.tensor([0.0, 0.3], requires_grad=True)
    outputs = {
        "final_logit": base + context,
        "base_logit": base,
        "context_logit": context,
    }

    loss = _context_auxiliary_loss(
        outputs,
        torch.tensor([1.0, 0.0]),
        nn.BCEWithLogitsLoss(),
        weight=0.5,
    )
    loss.backward()

    assert base.grad is None
    assert context.grad is not None
    assert float(context.grad.abs().sum()) > 0.0


def test_auc_ranking_loss_rewards_positive_logits_above_negative_logits() -> None:
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    well_ranked = torch.tensor([3.0, 2.0, -1.0, -2.0])
    inverted = torch.tensor([-3.0, -2.0, 1.0, 2.0])

    good_loss = _auc_ranking_loss(well_ranked, labels, weight=0.5, max_pairs=16)
    bad_loss = _auc_ranking_loss(inverted, labels, weight=0.5, max_pairs=16)

    assert good_loss < bad_loss
    assert _auc_ranking_loss(well_ranked, labels, weight=0.0, max_pairs=16) == 0.0
    assert (
        _auc_ranking_loss(well_ranked, torch.ones_like(labels), weight=0.5, max_pairs=16)
        == 0.0
    )


def test_auc_ranking_loss_backpropagates_through_sampled_pairs() -> None:
    torch.manual_seed(7)
    logits = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

    loss = _auc_ranking_loss(logits, labels, weight=0.25, max_pairs=3)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_context_residual_metrics_are_machine_readable_logit_summaries() -> None:
    metrics = _context_residual_metrics(np.array([-2.0, 0.0, 1.0], dtype=np.float64))

    assert metrics["mean_logit"] == pytest.approx(-1.0 / 3.0)
    assert metrics["mean_abs_logit"] == pytest.approx(1.0)
    assert metrics["rms_logit"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert metrics["p95_abs_logit"] == pytest.approx(1.9)
    assert metrics["max_abs_logit"] == pytest.approx(2.0)
