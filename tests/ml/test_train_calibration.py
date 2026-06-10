from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from app.ml.config import TrainConfig
from app.ml.context_examples_audit import AuditData
from app.ml.group_context_audit import build_group_rows, summarize
from app.ml.context_audit_specs import (
    eb_shrink_targets,
    group_audit_specs,
)
from app.ml.dataset import SplitData
from app.ml.hgnn_model import HGNNConfig
from app.ml.train import (
    PRIOR_1VX_SUPPORT_RISK_BUCKETS,
    _attach_output_diagnostics,
    _drop_unused_model_arrays,
    _fit_temperature,
    _prior_1vx_support_bucket_ids,
    _select_threshold,
    _semantic_context_gap_metrics,
    _semantic_group_eb_gap_metrics,
    _sigmoid_np,
    _support_bucket_metrics,
    _threshold_accuracy,
    _validate_split_targets,
    _validate_train_config,
)
from app.ml.semantic_group_features import SEMANTIC_GROUP_FEATURE_DIM


PRODUCTION_BUILD_VOCAB = (
    "ability_power",
    "ad_off_tank",
    "ap_off_tank",
    "ar_tank",
    "attack_damage",
    "crit",
    "lethality",
    "mr_tank",
    "on_hit",
    "utility_enchanter",
    "utility_protection",
)


def _split(labels: np.ndarray) -> SplitData:
    n = int(labels.size)
    return SplitData(
        win_rate=np.zeros((n, 10), dtype=np.float32),
        p1_cnt=np.zeros((n, 10), dtype=np.float32),
        blue_win=labels.astype(np.float64, copy=False),
        loadout_features=np.zeros((n, 10), dtype=np.float32),
        patch_features=np.zeros((n, 2), dtype=np.float32),
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


def test_prior_1vx_support_bucket_metrics_expose_variance_risk() -> None:
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


def test_default_model_config_drops_optional_model_arrays_before_tensor_cache() -> None:
    split = _split(np.array([0, 1], dtype=np.float64))

    dropped = _drop_unused_model_arrays(split, HGNNConfig())
    assert dropped.loadout_features is None
    assert dropped.patch_features is None
    assert dropped.identity_static_sidecar is None
    assert dropped.identity_full_game_sidecar is None
    assert dropped.identity_temporal_sidecar is None
    assert dropped.identity_encoder_support is None

    feature_kept = _drop_unused_model_arrays(
        split,
        HGNNConfig(loadout_feature_dim=10, patch_feature_dim=2),
    )
    assert feature_kept.loadout_features is split.loadout_features
    assert feature_kept.patch_features is split.patch_features

    sidecar_kept = _drop_unused_model_arrays(
        split,
        HGNNConfig(use_identity_static_sidecar=True),
    )
    assert sidecar_kept.identity_static_sidecar is split.identity_static_sidecar
    assert sidecar_kept.identity_full_game_sidecar is None
    assert sidecar_kept.identity_temporal_sidecar is None
    assert sidecar_kept.identity_encoder_support is split.identity_encoder_support

    moe_kept = _drop_unused_model_arrays(
        split,
        HGNNConfig(use_learned_semantic_moe=True),
    )
    assert moe_kept.identity_static_sidecar is split.identity_static_sidecar
    assert moe_kept.identity_full_game_sidecar is split.identity_full_game_sidecar
    assert moe_kept.identity_temporal_sidecar is split.identity_temporal_sidecar
    assert moe_kept.identity_encoder_support is split.identity_encoder_support


def test_train_config_rejects_unknown_raw_tensor_cache_device() -> None:
    with pytest.raises(ValueError, match="raw_tensor_cache_device"):
        _validate_train_config(
            replace(TrainConfig(), raw_tensor_cache_device="accelerator")
        )


def test_semantic_moe_head_fails_early_without_sidecar_cache_arrays() -> None:
    split = replace(
        _split(np.array([0, 1], dtype=np.float64)),
        identity_temporal_sidecar=None,
    )

    with pytest.raises(ValueError, match="semantic MoE head requires cache arrays"):
        _drop_unused_model_arrays(split, HGNNConfig(use_learned_semantic_moe=True))


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
    assert moe["router_entropy_fraction_of_topk_max"] == pytest.approx(
        0.6 / np.log(2.0)
    )
    assert moe["regularization_loss"] == pytest.approx(0.06)


def test_training_target_validation_catches_degenerate_cache_split() -> None:
    splits = {
        "train": _split(np.array([0, 1], dtype=np.float64)),
        "val": _split(np.zeros(3, dtype=np.float64)),
        "test": _split(np.array([0, 1], dtype=np.float64)),
    }

    with pytest.raises(ValueError, match="val split has degenerate blue_win labels"):
        _validate_split_targets(splits)


def test_freeze_loaded_parameters_requires_warm_start() -> None:
    with pytest.raises(ValueError, match="warm_start_model_path"):
        _validate_train_config(TrainConfig(freeze_warm_start_loaded_parameters=True))


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
        "ability_power",
        "ad_off_tank",
        "ap_off_tank",
        "ar_tank",
        "attack_damage",
        "crit",
        "lethality",
        "mr_tank",
        "on_hit",
        "utility_enchanter",
        "utility_protection",
    }
    specs = group_audit_specs()
    assert specs
    for spec in specs:
        assert set(spec.builds) <= build_vocab, spec.title


def test_semantic_context_metrics_expose_high_support_tail() -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    n = int(labels.size)
    champion_id = np.zeros((n, 10), dtype=np.int64)
    champion_id[:, 2] = 103
    build_id = np.full(
        (n, 10),
        PRODUCTION_BUILD_VOCAB.index("attack_damage"),
        dtype=np.int64,
    )
    build_id[:, 2] = PRODUCTION_BUILD_VOCAB.index("ability_power")
    build_id[:4, 5:] = PRODUCTION_BUILD_VOCAB.index("ar_tank")
    context_raw = np.zeros((n, 10, 14), dtype=np.float32)
    side_predictions = np.full((n, 10), 0.50, dtype=np.float64)
    side_predictions[:, 2] = np.array([0.90, 0.80, 0.85, 0.82, 0.70, 0.72])
    split = SplitData(
        win_rate=np.zeros((n, 10), dtype=np.float32),
        p1_cnt=np.zeros((n, 10), dtype=np.float32),
        blue_win=labels,
        champion_id=champion_id,
        build_id=build_id,
        context_raw=context_raw,
    )

    metrics = _semantic_context_gap_metrics(
        side_predictions,
        split,
        build_vocab=PRODUCTION_BUILD_VOCAB,
        min_count=1,
    )
    gated_metrics = _semantic_context_gap_metrics(
        side_predictions,
        split,
        build_vocab=PRODUCTION_BUILD_VOCAB,
        min_count=(n * 10) + 1,
    )

    assert metrics["context_populated_bins"] > 0
    assert (
        metrics["context_high_support_populated_bins"]
        == metrics["context_populated_bins"]
    )
    assert metrics["context_high_support_max_abs_gap"] == pytest.approx(
        metrics["context_max_abs_gap"]
    )
    assert np.isfinite(metrics["context_support_weighted_gap_mse"])
    assert gated_metrics["context_support_min_count"] == (n * 10) + 1
    assert gated_metrics["context_high_support_populated_bins"] == 0
    assert np.isnan(gated_metrics["context_high_support_max_abs_gap"])


def test_semantic_group_eb_metrics_are_zero_for_perfect_side_probabilities() -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    n = int(labels.size)
    side_labels = np.concatenate(
        [
            np.repeat(labels[:, None], 5, axis=1),
            np.repeat((1.0 - labels)[:, None], 5, axis=1),
        ],
        axis=1,
    )
    split = SplitData(
        win_rate=np.zeros((n, 10), dtype=np.float32),
        p1_cnt=np.zeros((n, 10), dtype=np.float32),
        blue_win=labels,
        champion_id=np.zeros((n, 10), dtype=np.int64),
        build_id=np.full(
            (n, 10), PRODUCTION_BUILD_VOCAB.index("ar_tank"), dtype=np.int64
        ),
        context_raw=np.zeros((n, 10, 14), dtype=np.float32),
        semantic_group_features=np.zeros(
            (n, 10, SEMANTIC_GROUP_FEATURE_DIM),
            dtype=np.float32,
        ),
    )

    metrics = _semantic_group_eb_gap_metrics(
        side_labels,
        split,
        build_vocab=PRODUCTION_BUILD_VOCAB,
    )

    assert metrics["group_n_bins"] > 0
    assert metrics["group_eb_gap_mse"] == pytest.approx(0.0)
    assert metrics["group_systematic_gap_mse"] == pytest.approx(0.0)


def test_train_group_eb_metrics_match_formal_group_audit_lens(tmp_path) -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    n = int(labels.size)
    champion_id = np.zeros((n, 10), dtype=np.int64)
    build_id = np.full(
        (n, 10),
        PRODUCTION_BUILD_VOCAB.index("ar_tank"),
        dtype=np.int64,
    )
    context_raw = np.zeros((n, 10, 14), dtype=np.float32)
    side_predictions = np.concatenate(
        [
            np.repeat(np.array([[0.70], [0.45], [0.62], [0.51]]), 5, axis=1),
            np.repeat(np.array([[0.30], [0.55], [0.38], [0.49]]), 5, axis=1),
        ],
        axis=1,
    )
    split = SplitData(
        win_rate=np.zeros((n, 10), dtype=np.float32),
        p1_cnt=np.zeros((n, 10), dtype=np.float32),
        blue_win=labels,
        champion_id=champion_id,
        build_id=build_id,
        context_raw=context_raw,
    )
    metrics = _semantic_group_eb_gap_metrics(
        side_predictions,
        split,
        build_vocab=PRODUCTION_BUILD_VOCAB,
    )

    (tmp_path / "cache_meta.json").write_text(
        json.dumps(
            {
                "n_games": n,
                "splits": {"train": n, "val": 0, "test": 0},
                "split_order": ["train", "val", "test"],
                "identity": {"build_vocab": list(PRODUCTION_BUILD_VOCAB)},
            }
        ),
        encoding="utf-8",
    )
    np.save(tmp_path / "blue_win.npy", labels)
    np.save(tmp_path / "champion_id.npy", champion_id)
    np.save(tmp_path / "build_id.npy", build_id)
    np.save(tmp_path / "identity_context_raw.npy", context_raw)
    formal = summarize(
        build_group_rows(
            AuditData(
                context_cache_dir=tmp_path,
                blue_probability=side_predictions,
                audit_split="all",
            ),
            group_audit_specs(),
        )
    )

    assert metrics["group_n_bins"] == formal["n_bins"]
    assert metrics["group_median_n"] == pytest.approx(formal["median_n"])
    assert metrics["group_min_n"] == formal["min_n"]
    assert metrics["group_raw_gap_mse"] == pytest.approx(formal["raw_gap_mse"])
    assert metrics["group_raw_floor"] == pytest.approx(formal["raw_floor"])
    assert metrics["group_eb_gap_mse"] == pytest.approx(formal["eb_gap_mse"])
    assert metrics["group_eb_floor"] == pytest.approx(formal["eb_floor"])
    assert metrics["group_systematic_gap_mse"] == pytest.approx(
        formal["systematic_gap_mse"]
    )
    assert metrics["group_systematic_gap_mse_clipped"] == pytest.approx(
        formal["systematic_gap_mse_clipped"]
    )
