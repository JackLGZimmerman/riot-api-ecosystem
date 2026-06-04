from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from app.ml.config import TrainConfig
from app.ml.context_audit_specs import eb_shrink_targets, group_audit_specs
from app.ml.dataset import SplitData
from app.ml.hgnn_model import HGNNConfig
from app.ml.train import (
    CHECKPOINT_METRICS,
    PRIOR_1VX_SUPPORT_RISK_BUCKETS,
    _attach_output_diagnostics,
    _auc_ranking_loss,
    _checkpoint_score,
    _drop_unused_model_arrays,
    _fit_temperature,
    _prior_1vx_support_bucket_ids,
    _select_threshold,
    _sigmoid_np,
    _support_bucket_metrics,
    _threshold_accuracy,
    _validate_split_targets,
    _validate_train_config,
)


def _split(labels: np.ndarray) -> SplitData:
    n = int(labels.size)
    return SplitData(
        win_rate=np.zeros((n, 10), dtype=np.float32),
        matchup_1v1=np.zeros((n, 25), dtype=np.float32),
        synergy_2vx=np.zeros((n, 20), dtype=np.float32),
        p1_cnt=np.zeros((n, 10), dtype=np.float32),
        m1v1_cnt=np.zeros((n, 25), dtype=np.float32),
        s2vx_cnt=np.zeros((n, 20), dtype=np.float32),
        blue_win=labels.astype(np.float64, copy=False),
        identity_static_sidecar=np.zeros((n, 10, 2), dtype=np.float32),
        identity_full_game_sidecar=np.zeros((n, 10, 3), dtype=np.float32),
        identity_temporal_sidecar=np.zeros((n, 10, 4), dtype=np.float32),
        identity_encoder_support=np.zeros((n, 10), dtype=np.float32),
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


def test_prior_1vx_support_bucket_metrics_expose_variance_ablation_risk() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.float64)
    split = _split(labels)
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


def test_default_model_config_drops_relationship_tables_before_tensor_cache() -> None:
    split = _split(np.array([0, 1], dtype=np.float64))

    dropped = _drop_unused_model_arrays(split, HGNNConfig())
    assert dropped.matchup_1v1 is None
    assert dropped.synergy_2vx is None
    assert dropped.m1v1_cnt is None
    assert dropped.s2vx_cnt is None
    assert dropped.identity_static_sidecar is None
    assert dropped.identity_full_game_sidecar is None
    assert dropped.identity_temporal_sidecar is None
    assert dropped.identity_encoder_support is None

    kept = _drop_unused_model_arrays(
        split,
        HGNNConfig(use_relationship_integrations=True),
    )
    assert kept.matchup_1v1 is split.matchup_1v1
    assert kept.synergy_2vx is split.synergy_2vx

    sidecar_kept = _drop_unused_model_arrays(
        split,
        HGNNConfig(use_identity_static_sidecar=True),
    )
    assert sidecar_kept.identity_static_sidecar is split.identity_static_sidecar
    assert sidecar_kept.identity_full_game_sidecar is None
    assert sidecar_kept.identity_temporal_sidecar is None
    assert sidecar_kept.identity_encoder_support is split.identity_encoder_support

    semantic_kept = _drop_unused_model_arrays(
        split,
        HGNNConfig(use_identity_semantic_context_head=True),
    )
    assert semantic_kept.identity_static_sidecar is split.identity_static_sidecar
    assert semantic_kept.identity_full_game_sidecar is split.identity_full_game_sidecar
    assert semantic_kept.identity_temporal_sidecar is split.identity_temporal_sidecar
    assert semantic_kept.identity_encoder_support is split.identity_encoder_support

    moe_kept = _drop_unused_model_arrays(
        split,
        HGNNConfig(use_learned_semantic_moe=True),
    )
    assert moe_kept.identity_static_sidecar is split.identity_static_sidecar
    assert moe_kept.identity_full_game_sidecar is split.identity_full_game_sidecar
    assert moe_kept.identity_temporal_sidecar is split.identity_temporal_sidecar
    assert moe_kept.identity_encoder_support is split.identity_encoder_support


def test_semantic_context_or_moe_head_fails_early_without_sidecar_cache_arrays() -> None:
    split = replace(
        _split(np.array([0, 1], dtype=np.float64)),
        identity_temporal_sidecar=None,
    )

    for config in (
        HGNNConfig(use_identity_semantic_context_head=True),
        HGNNConfig(use_learned_semantic_moe=True),
    ):
        with pytest.raises(ValueError, match="semantic context/MoE head requires cache arrays"):
            _drop_unused_model_arrays(split, config)


def test_output_diagnostics_include_logit_std_and_semantic_moe_stats() -> None:
    split_metrics: dict[str, dict[str, object]] = {"val": {}}
    _attach_output_diagnostics(
        split_metrics,
        {
            "val": {
                "base_logit": np.array([-1.0, 0.0, 1.0]),
                "context_logit": np.array([0.2, 0.0, -0.2]),
                "final_logit": np.array([-0.8, 0.0, 0.8]),
                "semantic_moe_expert_usage": np.array([0.2, 0.5, 0.3]),
                "semantic_moe_expert_selected_fraction": np.array([0.5, 1.0, 0.5]),
                "semantic_moe_router_entropy": np.array(0.6),
                "semantic_moe_factor_norm": np.array(2.0),
                "semantic_moe_balance_loss": np.array(0.01),
                "semantic_moe_entropy_loss": np.array(0.02),
                "semantic_moe_factor_orthogonality_loss": np.array(0.03),
                "semantic_moe_factor_variance_loss": np.array(0.04),
                "semantic_moe_factor_std_mean": np.array(0.5),
                "semantic_moe_factor_std_min": np.array(0.1),
                "semantic_moe_context_token_keep_fraction": np.array(1.0),
                "semantic_moe_delta_l2_loss": np.array(0.05),
                "semantic_moe_regularization_loss": np.array(0.06),
            }
        },
    )

    logit_diagnostics = split_metrics["val"]["logit_diagnostics"]
    assert isinstance(logit_diagnostics, dict)
    assert logit_diagnostics["base_logit_std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert logit_diagnostics["context_logit_std"] == pytest.approx(np.sqrt(0.08 / 3.0))
    assert logit_diagnostics["final_logit_std"] == pytest.approx(np.sqrt(1.28 / 3.0))

    moe = split_metrics["val"]["semantic_moe_diagnostics"]
    assert isinstance(moe, dict)
    assert np.allclose(moe["expert_usage"], [0.2, 0.5, 0.3])
    assert np.allclose(moe["expert_selected_fraction"], [0.5, 1.0, 0.5])
    assert moe["expert_usage_min"] == pytest.approx(0.2)
    assert moe["expert_usage_max"] == pytest.approx(0.5)
    assert moe["router_entropy_fraction_of_topk_max"] == pytest.approx(0.6 / np.log(2.0))
    assert moe["regularization_loss"] == pytest.approx(0.06)


def test_training_target_validation_catches_degenerate_cache_split() -> None:
    splits = {
        "train": _split(np.array([0, 1], dtype=np.float64)),
        "val": _split(np.zeros(3, dtype=np.float64)),
        "test": _split(np.array([0, 1], dtype=np.float64)),
    }

    with pytest.raises(ValueError, match="val split has degenerate blue_win labels"):
        _validate_split_targets(splits)


def test_auc_ranking_loss_config_fails_early_when_invalid() -> None:
    with pytest.raises(ValueError, match="auc_ranking_loss_weight"):
        _validate_train_config(TrainConfig(auc_ranking_loss_weight=-0.1))
    with pytest.raises(ValueError, match="auc_ranking_loss_pairs"):
        _validate_train_config(TrainConfig(auc_ranking_loss_pairs=0))


def test_checkpoint_metric_fails_early_when_unknown() -> None:
    with pytest.raises(ValueError, match="checkpoint_metric"):
        _validate_train_config(TrainConfig(checkpoint_metric="test_auc"))


def test_calibration_target_validation_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="semantic_context_calibration_target"):
        _validate_train_config(
            TrainConfig(semantic_context_calibration_target="bogus")
        )
    # valid families pass through validation.
    for target in ("champion_raw", "group_eb"):
        _validate_train_config(
            TrainConfig(semantic_context_calibration_target=target)
        )


def test_eb_shrink_targets_pulls_small_noisy_bins_toward_row_mean() -> None:
    counts = np.array([10_000.0, 50.0])
    means = np.array([0.50, 0.30])
    eb, eb_var = eb_shrink_targets(counts, means)
    mu = float(np.sum(counts * means) / counts.sum())
    sampling_var = means * (1.0 - means) / counts

    # the tiny, far-from-mean bin is shrunk hard toward the pooled mean.
    assert abs(eb[1] - mu) < 0.25 * abs(means[1] - mu)
    # the large, on-trend bin barely moves.
    assert abs(eb[0] - means[0]) < 1e-3
    # the EB target variance never exceeds the raw sampling variance (debiasing).
    assert np.all(eb_var <= sampling_var + 1e-12)
    assert eb_var[1] < sampling_var[1]


def test_eb_shrink_targets_collapses_to_mean_without_between_bin_signal() -> None:
    counts = np.array([1000.0, 1000.0, 1000.0])
    means = np.array([0.5, 0.5, 0.5])
    eb, eb_var = eb_shrink_targets(counts, means)
    assert np.allclose(eb, 0.5)
    assert np.allclose(eb_var, 0.0)


def test_group_audit_specs_only_use_known_build_labels() -> None:
    build_vocab = {
        "ability_power", "ad_off_tank", "ap_off_tank", "ar_tank", "attack_damage",
        "crit", "lethality", "mr_tank", "on_hit", "utility_enchanter",
        "utility_protection",
    }
    specs = group_audit_specs()
    assert specs
    for spec in specs:
        assert set(spec.builds) <= build_vocab, spec.title


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
