from __future__ import annotations

import json

from app.ml.experiments.encoder_input_ablation import (
    ABLATION_BY_NAME,
    DEFAULT_EXPERIMENT_ROOT,
    PRODUCTION_AUDIT_TARGETS,
    PRODUCTION_MODEL_TARGETS,
    _commands_for_steps,
    audit_command,
    audit_json_path,
    build_sidecar_command,
    compare_runs,
    selected_specs,
    should_freeze_warm_start,
    sidecar_path,
    train_command,
)


def test_encoder_input_ablation_matrix_contains_expected_stage1_runs() -> None:
    names = {spec.name for spec in selected_specs(["stage1"])}

    assert {
        "fg_profile_only",
        "fg_raw_context",
        "fg_context_only",
        "fg_no_identity",
        "fg_support_log1p",
        "fg_soft_v2_w010",
        "fg_soft_v2_w020",
        "fg_pca_whitened",
        "fg_semantic_targets",
        "static_latent_32",
        "tmp_mask_flat",
        "tmp_mask_tcn",
        "tmp_mask_gru",
        "tmp_mask_standalone",
        "tmp_no_zero_unobserved",
        "mv_vicreg_w010",
        "mv_barlow_w010",
    } <= names

    for spec in selected_specs(["stage1"]):
        assert "--full-game-allow-outcome-metrics" not in spec.sidecar_flags


def test_diagnostic_latent_exports_are_not_promotion_eligible() -> None:
    assert ABLATION_BY_NAME["fg_semantic_targets"].diagnostic_only is True
    assert ABLATION_BY_NAME["fg_semantic_targets"].promotion_eligible is False
    assert ABLATION_BY_NAME["tmp_no_zero_unobserved"].diagnostic_only is True
    assert ABLATION_BY_NAME["tmp_no_zero_unobserved"].promotion_eligible is False


def test_temporal_ablation_reuses_rebuilt_control_static_full_game(tmp_path) -> None:
    spec = ABLATION_BY_NAME["tmp_mask_gru"]
    command = build_sidecar_command(spec, experiment_root=tmp_path)

    assert "--reuse-static-full-game-from" in command
    reuse_index = command.index("--reuse-static-full-game-from") + 1
    assert command[reuse_index] == str(
        tmp_path / "control_rebuilt" / "sidecar.npz"
    )
    assert command[-3:] == [
        "--temporal-mask-as-input",
        "--temporal-architecture",
        "gru",
    ]


def test_temporal_sidecar_steps_prepend_missing_rebuilt_control(tmp_path) -> None:
    spec = ABLATION_BY_NAME["tmp_mask_flat"]
    emitted: set[str] = set()

    commands = _commands_for_steps(
        spec,
        steps=("sidecar",),
        seed=4,
        experiment_root=tmp_path,
        freeze_mode="auto",
        refresh_predictions=False,
        bootstrap_samples=0,
        emitted_prerequisites=emitted,
    )
    repeated = _commands_for_steps(
        spec,
        steps=("sidecar",),
        seed=4,
        experiment_root=tmp_path,
        freeze_mode="auto",
        refresh_predictions=False,
        bootstrap_samples=0,
        emitted_prerequisites=emitted,
    )

    assert len(commands) == 2
    assert commands[0][commands[0].index("--output") + 1] == str(
        tmp_path / "control_rebuilt" / "sidecar.npz"
    )
    assert commands[1][commands[1].index("--output") + 1] == str(
        tmp_path / "tmp_mask_flat" / "sidecar.npz"
    )
    assert len(repeated) == 1
    assert repeated[0][repeated[0].index("--output") + 1] == str(
        tmp_path / "tmp_mask_flat" / "sidecar.npz"
    )


def test_static_latent_width_changes_hgnn_shape_and_freeze_auto() -> None:
    static_32 = ABLATION_BY_NAME["static_latent_32"]
    same_width = ABLATION_BY_NAME["fg_profile_only"]

    assert static_32.changes_hgnn_shape is True
    assert same_width.changes_hgnn_shape is False
    assert should_freeze_warm_start(static_32, "auto") is True
    assert should_freeze_warm_start(same_width, "auto") is False

    static_command = train_command(static_32, seed=4)
    same_width_command = train_command(same_width, seed=4)

    assert "--freeze-warm-start-loaded-parameters" in static_command
    assert "--freeze-warm-start-loaded-parameters" not in same_width_command


def test_sidecar_train_and_audit_paths_are_run_scoped(tmp_path) -> None:
    spec = ABLATION_BY_NAME["fg_soft_v2_w010"]

    sidecar_command = build_sidecar_command(spec, experiment_root=tmp_path)
    train = train_command(spec, seed=4, experiment_root=tmp_path)
    audit = audit_command(
        spec,
        seed=4,
        audit_split="validation",
        experiment_root=tmp_path,
        refresh_predictions=True,
    )

    assert str(tmp_path / spec.name / "sidecar.npz") in sidecar_command
    assert str(tmp_path / spec.name / "sidecar_summary.json") in sidecar_command
    assert "--full-game-semantic-target-mode" in sidecar_command
    assert "soft_v2" in sidecar_command
    assert "0.10" in sidecar_command
    assert str(tmp_path / spec.name / "model_seed4.pt") in train
    assert str(tmp_path / spec.name / "metrics_seed4.json") in train
    assert str(tmp_path / spec.name / "audit_focus_seed4.npy") in audit
    assert audit[audit.index("--audit-split") + 1] == "val"
    assert any("context_examples_audit_validation_seed4.json" in arg for arg in audit)
    assert "--refresh-predictions" in audit


def test_production_sidecar_control_reuses_checked_in_sidecar() -> None:
    spec = ABLATION_BY_NAME["production_sidecar_control"]

    assert spec.needs_sidecar_build is False
    assert sidecar_path(spec, experiment_root=DEFAULT_EXPERIMENT_ROOT) == (
        DEFAULT_EXPERIMENT_ROOT.parent / "semantic_identity_sidecar_compact.npz"
    )


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_runs_applies_validation_and_promotion_gates(tmp_path) -> None:
    spec = ABLATION_BY_NAME["fg_profile_only"]
    run_dir = tmp_path / spec.name
    _write_json(
        run_dir / "metrics_seed4.json",
        {
            "val": {
                "accuracy": 0.5790,
                "nll": 0.6720,
                "auc": 0.61,
            },
            "test": {
                "accuracy": 0.5740,
                "nll": 0.6750,
                "auc": 0.604,
            },
        },
    )
    _write_json(
        audit_json_path(
            spec,
            experiment_root=tmp_path,
            seed=4,
            audit_split="validation",
        ),
        {
            "split_summaries": [
                {
                    "split": "val",
                    "gap_mse": PRODUCTION_AUDIT_TARGETS["val"]["gap_mse"] * 0.90,
                "mean_abs_gap": PRODUCTION_AUDIT_TARGETS["val"]["mean_abs_gap"]
                * 0.90,
                    "max_abs_gap": 0.08,
                    "accuracy": 0.578,
                }
            ]
        },
    )
    _write_json(
        audit_json_path(spec, experiment_root=tmp_path, seed=4, audit_split="all"),
        {
            "splits": {
                "all": {
                    "summary": {
                        "gap_mse": PRODUCTION_AUDIT_TARGETS["all"][
                            "gap_mse"
                        ]
                        * 0.90,
                        "mean_abs_gap": PRODUCTION_AUDIT_TARGETS["all"][
                            "mean_abs_gap"
                        ]
                        * 0.90,
                        "max_abs_gap": PRODUCTION_AUDIT_TARGETS["all"][
                            "max_abs_gap"
                        ]
                        * 0.90,
                        "accuracy": PRODUCTION_AUDIT_TARGETS["all"][
                            "accuracy"
                        ],
                    }
                }
            },
            "split_summaries": [
                {
                    "split": "val",
                    "gap_mse": PRODUCTION_AUDIT_TARGETS["val"]["gap_mse"] * 0.90,
                    "mean_abs_gap": PRODUCTION_AUDIT_TARGETS["val"][
                        "mean_abs_gap"
                    ]
                    * 0.90,
                },
                {
                    "split": "test",
                    "gap_mse": PRODUCTION_AUDIT_TARGETS["test"]["gap_mse"] * 0.90,
                    "mean_abs_gap": PRODUCTION_AUDIT_TARGETS["test"][
                        "mean_abs_gap"
                    ]
                    * 0.90,
                },
            ],
        },
    )

    result = compare_runs([spec], seeds=[4], experiment_root=tmp_path)
    row = result["rows"][0]

    assert row["passes_validation_screen"] is True
    assert row["passes_promotion_gate"] is True


def test_compare_runs_requires_present_model_gate_metrics(tmp_path) -> None:
    spec = ABLATION_BY_NAME["fg_profile_only"]
    run_dir = tmp_path / spec.name
    _write_json(
        run_dir / "metrics_seed4.json",
        {
            "val": {
                "accuracy": PRODUCTION_MODEL_TARGETS["val_accuracy"],
            },
            "test": {
                "accuracy": PRODUCTION_MODEL_TARGETS["test_accuracy"],
                "nll": PRODUCTION_MODEL_TARGETS["test_nll"],
            },
        },
    )
    _write_json(
        audit_json_path(
            spec,
            experiment_root=tmp_path,
            seed=4,
            audit_split="validation",
        ),
        {
            "split_summaries": [
                {
                    "split": "val",
                    "gap_mse": PRODUCTION_AUDIT_TARGETS["val"]["gap_mse"] * 0.90,
                    "mean_abs_gap": PRODUCTION_AUDIT_TARGETS["val"]["mean_abs_gap"]
                    * 0.90,
                }
            ]
        },
    )
    _write_json(
        audit_json_path(spec, experiment_root=tmp_path, seed=4, audit_split="all"),
        {
            "splits": {
                "all": {
                    "summary": {
                        "gap_mse": PRODUCTION_AUDIT_TARGETS["all"]["gap_mse"] * 0.90,
                        "mean_abs_gap": PRODUCTION_AUDIT_TARGETS["all"][
                            "mean_abs_gap"
                        ]
                        * 0.90,
                        "max_abs_gap": PRODUCTION_AUDIT_TARGETS["all"][
                            "max_abs_gap"
                        ]
                        * 0.90,
                        "accuracy": PRODUCTION_AUDIT_TARGETS["all"]["accuracy"],
                    }
                }
            },
            "split_summaries": [
                {
                    "split": "val",
                    "gap_mse": PRODUCTION_AUDIT_TARGETS["val"]["gap_mse"] * 0.90,
                    "mean_abs_gap": PRODUCTION_AUDIT_TARGETS["val"][
                        "mean_abs_gap"
                    ]
                    * 0.90,
                },
                {
                    "split": "test",
                    "gap_mse": PRODUCTION_AUDIT_TARGETS["test"]["gap_mse"] * 0.90,
                    "mean_abs_gap": PRODUCTION_AUDIT_TARGETS["test"][
                        "mean_abs_gap"
                    ]
                    * 0.90,
                },
            ],
        },
    )

    result = compare_runs([spec], seeds=[4], experiment_root=tmp_path)
    row = result["rows"][0]

    assert row["passes_validation_screen"] is None
    assert row["passes_promotion_gate"] is None


def test_compare_runs_keeps_diagnostics_out_of_promotion(tmp_path) -> None:
    spec = ABLATION_BY_NAME["fg_semantic_targets"]

    result = compare_runs([spec], seeds=[4], experiment_root=tmp_path)
    row = result["rows"][0]

    assert row["passes_validation_screen"] is None
    assert row["passes_promotion_gate"] is False
