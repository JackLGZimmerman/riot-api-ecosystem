from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ml.experiments.context_ablation import (
    Variant,
    _aggregate_repeated_rows,
    _leaderboard_row,
    _normalise_seeds,
    _validated_train_overrides,
    _variant_map,
    _write_leaderboard,
)


def test_leaderboard_row_includes_validation_semantic_summary_fields() -> None:
    summary = {
        "semantic_summary": {
            "aggregate": {
                "n_effects": 2,
                "mean_abs_delta_gap": 0.03,
                "max_abs_delta_gap": 0.10,
                "mean_abs_endpoint_gap": 0.02,
                "max_abs_endpoint_gap": 0.08,
            },
            "effects": [
                {"label": "smaller", "delta_gap": 0.04},
                {"label": "larger", "delta_gap": -0.10},
            ],
        }
    }

    row = _leaderboard_row(
        "candidate",
        {
            "val": {
                "threshold_accuracy": 0.58,
                "auc": 0.61,
                "nll": 0.67,
                "ece": 0.02,
                "brier": 0.24,
                "temperature_scaled": {"nll": 0.66, "ece": 0.01},
                "context_residual": {"mean_abs_logit": 0.02, "p95_abs_logit": 0.05},
            },
            "train_config": {
                "checkpoint_metric": "val_auc",
                "checkpoint_min_delta": 0.0001,
                "auc_ranking_loss_weight": 0.10,
                "auc_ranking_loss_pairs": 8192,
            },
            "model_config": {
                "use_1vx_posterior_variance": True,
            },
            "best_checkpoint_score": 0.61,
            "best_checkpoint_val_nll": 0.67,
            "best_checkpoint_val_ece": 0.02,
            "test": {
                "temperature_scaled": {"nll": 0.68, "ece": 0.03},
                "context_residual": {"mean_abs_logit": 0.03, "p95_abs_logit": 0.06},
                "support_buckets": {
                    "prior_1vx_support": {
                        "risk_bucket": {
                            "zero_player": {"n": 12, "auc": 0.55, "calibration_gap": 0.08},
                            "min_50_plus": {"n": 24, "auc": 0.62, "calibration_gap": -0.02},
                        },
                    },
                },
            },
        },
        semantic_summary=summary,
        semantic_summary_path=Path("candidate/semantic_summary_val.json"),
    )

    assert row["semantic_summary_path"] == "candidate/semantic_summary_val.json"
    assert row["val_semantic_n_effects"] == 2
    assert row["val_semantic_mean_abs_delta_gap"] == 0.03
    assert row["val_semantic_worst_effect"] == "larger"
    assert row["val_semantic_worst_effect_delta_gap"] == -0.10
    assert row["val_context_mean_abs_logit"] == 0.02
    assert row["test_context_p95_abs_logit"] == 0.06
    assert row["checkpoint_metric"] == "val_auc"
    assert row["checkpoint_min_delta"] == 0.0001
    assert row["auc_ranking_loss_weight"] == 0.10
    assert row["auc_ranking_loss_pairs"] == 8192
    assert row["best_checkpoint_score"] == 0.61
    assert row["best_checkpoint_val_nll"] == 0.67
    assert row["best_checkpoint_val_ece"] == 0.02
    assert row["val_ece"] == 0.02
    assert row["val_brier"] == 0.24
    assert row["val_temperature_scaled_nll"] == 0.66
    assert row["val_temperature_scaled_ece"] == 0.01
    assert row["test_temperature_scaled_ece"] == 0.03
    assert row["use_1vx_posterior_variance"] is True
    assert row["test_prior_1vx_support_max_abs_gap"] == 0.08
    assert row["test_prior_1vx_support_min_bucket_auc"] == 0.55


def test_leaderboard_ranking_uses_validation_metrics_not_test_tie_breakers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "leaderboard.json"
    rows = [
        {
            "variant": "better_test_only",
            "val_threshold_accuracy": 0.58,
            "checkpoint_metric": "val_auc",
            "val_auc": 0.60,
            "val_nll": 0.67,
            "test_auc": 0.99,
        },
        {
            "variant": "better_validation",
            "val_threshold_accuracy": 0.58,
            "val_auc": 0.61,
            "val_nll": 0.68,
            "test_auc": 0.50,
        },
    ]

    _write_leaderboard(path, rows)

    ranked = json.loads(path.read_text(encoding="utf-8"))
    assert [row["variant"] for row in ranked] == [
        "better_validation",
        "better_test_only",
    ]


def test_aggregate_repeated_rows_preserves_runs_and_exposes_mean_spread() -> None:
    rows = [
        {
            "variant": "candidate",
            "seed": 0,
            "val_threshold_accuracy": 0.58,
            "val_auc": 0.60,
            "val_nll": 0.67,
            "val_semantic_mean_abs_delta_gap": 0.04,
            "semantic_summary_path": "candidate/seed_0/semantic_summary_val.json",
        },
        {
            "variant": "candidate",
            "seed": 1,
            "val_threshold_accuracy": 0.60,
            "checkpoint_metric": "val_auc",
            "val_auc": 0.62,
            "val_nll": 0.65,
            "val_semantic_mean_abs_delta_gap": 0.02,
            "semantic_summary_path": "candidate/seed_1/semantic_summary_val.json",
        },
    ]

    aggregate = _aggregate_repeated_rows("candidate", rows, (0, 1))

    assert aggregate["variant"] == "candidate"
    assert aggregate["seeds"] == [0, 1]
    assert aggregate["n_repeats"] == 2
    assert aggregate["checkpoint_metric"] == [None, "val_auc"]
    assert [row["seed"] for row in aggregate["runs"]] == [0, 1]
    assert aggregate["semantic_summary_paths"] == [
        "candidate/seed_0/semantic_summary_val.json",
        "candidate/seed_1/semantic_summary_val.json",
    ]
    assert aggregate["val_threshold_accuracy"] == pytest.approx(0.59)
    assert aggregate["val_threshold_accuracy_mean"] == pytest.approx(0.59)
    assert aggregate["val_threshold_accuracy_std"] == pytest.approx(0.01)
    assert aggregate["val_threshold_accuracy_min"] == pytest.approx(0.58)
    assert aggregate["val_threshold_accuracy_max"] == pytest.approx(0.60)
    assert aggregate["val_semantic_mean_abs_delta_gap"] == pytest.approx(0.03)
    assert aggregate["test_auc"] is None
    assert aggregate["test_auc_std"] is None


def test_single_seed_aggregation_keeps_flat_leaderboard_shape() -> None:
    row = {"variant": "candidate", "val_threshold_accuracy": 0.58}

    aggregate = _aggregate_repeated_rows("candidate", [row], (7,))

    assert aggregate["variant"] == "candidate"
    assert aggregate["val_threshold_accuracy"] == 0.58
    assert aggregate["seed"] == 7
    assert aggregate["seeds"] == [7]
    assert aggregate["n_repeats"] == 1
    assert "runs" not in aggregate


def test_normalise_seeds_rejects_negative_or_duplicate_values() -> None:
    assert _normalise_seeds(SimpleNamespace(seed=3, seeds=None)) == (3,)
    assert _normalise_seeds(SimpleNamespace(seed=0, seeds=[0, 2])) == (0, 2)

    with pytest.raises(SystemExit, match="unique"):
        _normalise_seeds(SimpleNamespace(seed=0, seeds=[1, 1]))
    with pytest.raises(SystemExit, match=">= 0"):
        _normalise_seeds(SimpleNamespace(seed=0, seeds=[-1]))


def test_context_auxiliary_variant_is_explicit_train_override() -> None:
    variants = _variant_map()

    assert variants["current_mean_low_rank"].train_overrides == {}
    assert variants["low_rank_context_auxiliary"].train_overrides == {
        "context_auxiliary_loss_weight": 0.25
    }


def test_checkpoint_metric_variants_are_explicit_train_overrides() -> None:
    variants = _variant_map()

    assert (
        variants["low_rank_checkpoint_auc"].overrides
        == variants["current_mean_low_rank"].overrides
    )
    assert variants["low_rank_checkpoint_auc"].train_overrides == {
        "checkpoint_metric": "val_auc",
        "checkpoint_min_delta": 1.0e-4,
    }
    assert (
        variants["low_rank_checkpoint_nll"].overrides
        == variants["current_mean_low_rank"].overrides
    )
    assert variants["low_rank_checkpoint_nll"].train_overrides == {
        "checkpoint_metric": "val_nll",
        "checkpoint_min_delta": 1.0e-4,
    }
    assert (
        variants["low_rank_checkpoint_nll_ece"].overrides
        == variants["current_mean_low_rank"].overrides
    )
    assert variants["low_rank_checkpoint_nll_ece"].train_overrides == {
        "checkpoint_metric": "val_nll_ece",
        "checkpoint_min_delta": 1.0e-4,
    }
    assert (
        variants["low_rank_auc_ranking_loss"].overrides
        == variants["current_mean_low_rank"].overrides
    )
    assert variants["low_rank_auc_ranking_loss"].train_overrides == {
        "checkpoint_metric": "val_auc",
        "checkpoint_min_delta": 1.0e-4,
        "auc_ranking_loss_weight": 0.10,
        "auc_ranking_loss_pairs": 8192,
    }


def test_relationship_removal_is_recorded_on_default_leaderboard_rows() -> None:
    variants = _variant_map()

    assert "low_rank_no_relationship_auc" not in variants
    assert "low_rank_no_relationship_nll_ece" not in variants
    row = _leaderboard_row(
        "current_mean_low_rank",
        {
            "model_config": {"use_relationship_integrations": False},
            "train_config": {"checkpoint_metric": "val_threshold_accuracy"},
            "val": {},
            "test": {},
        },
    )
    assert row["use_relationship_integrations"] is False


def test_profile_detail_auc_variant_is_explicit_opt_in_sidecar_fusion() -> None:
    variants = _variant_map()

    assert variants["low_rank_profile_detail_auc"].overrides == {
        "use_identity_conditioned_context": True,
        "identity_context_conditioning_type": "low_rank",
        "context_set_encoder_type": "mean",
        "identity_profile_dim": "auto",
        "profile_include_ally_context": True,
        "profile_include_weighted_enemy_context": True,
        "profile_include_resistance_products": True,
        "profile_head_hidden": (32,),
        "m1v1_detail_dim": "auto",
    }
    assert variants["low_rank_profile_detail_auc"].train_overrides == {
        "checkpoint_metric": "val_auc",
        "checkpoint_min_delta": 1.0e-4,
        "auc_ranking_loss_weight": 0.10,
        "auc_ranking_loss_pairs": 8192,
    }


def test_wide_context_auc_variant_is_explicit_global_capacity() -> None:
    variants = _variant_map()

    assert variants["low_rank_wide_context_auc"].overrides == {
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
    }
    assert variants["low_rank_wide_context_auc"].train_overrides == {
        "checkpoint_metric": "val_auc",
        "checkpoint_min_delta": 1.0e-4,
        "auc_ranking_loss_weight": 0.10,
        "auc_ranking_loss_pairs": 8192,
    }


def test_slot_wide_context_auc_variant_is_explicit_role_order_capacity() -> None:
    variants = _variant_map()

    assert variants["low_rank_slot_wide_context_auc"].overrides == {
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
    }
    assert variants["low_rank_slot_wide_context_auc"].train_overrides == {
        "checkpoint_metric": "val_auc",
        "checkpoint_min_delta": 1.0e-4,
        "auc_ranking_loss_weight": 0.10,
        "auc_ranking_loss_pairs": 8192,
    }


def test_train_override_validation_rejects_hidden_fields() -> None:
    with pytest.raises(ValueError, match="unsupported train_overrides"):
        _validated_train_overrides(Variant("bad", {}, {"not_a_train_field": 1}))


def test_aggregate_repeated_rows_preserves_common_checkpoint_metric() -> None:
    rows = [
        {"variant": "candidate", "checkpoint_metric": "val_nll"},
        {"variant": "candidate", "checkpoint_metric": "val_nll"},
    ]

    aggregate = _aggregate_repeated_rows("candidate", rows, (0, 1))

    assert aggregate["checkpoint_metric"] == "val_nll"


def test_support_feature_variant_is_explicit_model_override() -> None:
    variants = _variant_map()

    assert variants["low_rank_support_features"].overrides == {
        "use_identity_conditioned_context": True,
        "identity_context_conditioning_type": "low_rank",
        "context_set_encoder_type": "mean",
        "identity_context_include_support_features": True,
    }
    assert variants["low_rank_support_features"].train_overrides == {}


def test_no_1vx_variance_variant_is_explicit_model_ablation() -> None:
    variants = _variant_map()

    assert variants["low_rank_no_1vx_variance"].overrides == {
        "use_identity_conditioned_context": True,
        "identity_context_conditioning_type": "low_rank",
        "context_set_encoder_type": "mean",
        "use_1vx_posterior_variance": False,
    }
    assert variants["low_rank_no_1vx_variance"].train_overrides == {}


def test_context_ablation_variants_do_not_reintroduce_prior_shortcut() -> None:
    for variant in _variant_map().values():
        assert "prior_shortcut_residual_hidden" not in variant.overrides
